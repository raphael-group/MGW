import numpy as np
import torch

class NumpyPairLoader:
    """
    Minimal loader that expects .npy files:
      - s.npy (n x dim_e), X.npy (n x dim_f1)
      - t.npy (m x dim_e), Z.npy (m x dim_f2)
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load(self):
        s = np.load(f"{self.base_dir}/s.npy")  # (n, dim_e)
        X = np.load(f"{self.base_dir}/X.npy")  # (n, dim_f1)
        t = np.load(f"{self.base_dir}/t.npy")  # (m, dim_e)
        Z = np.load(f"{self.base_dir}/Z.npy")  # (m, dim_f2)
        return s, X, t, Z
import numpy as np

# ---------- helpers ----------
def hex_grid(nx=30, ny=20, spacing=1.0, jitter=0.0, rng=None):
    """Regular hex grid in [-1,1]^2-ish box, with optional jitter."""
    if rng is None: rng = np.random.default_rng()
    xs, ys = [], []
    for j in range(ny):
        xoff = 0.5 * (j % 2)
        for i in range(nx):
            xs.append((i + xoff) * spacing)
            ys.append(j * spacing * np.sin(np.pi/3))
    X = np.stack([np.array(xs), np.array(ys)], axis=-1)
    # center & scale to roughly [-1,1]^2
    X -= X.mean(0)
    X /= X.max()
    if jitter > 0:
        X += rng.normal(scale=jitter, size=X.shape)
    return X.astype(np.float32)

def soft_voronoi_mixture(coords, centers, temp=0.08):
    """Soft assignments π_k(x) from distances to K centers with softmax temperature."""
    d2 = ((coords[:, None, :] - centers[None, :, :])**2).sum(-1)  # (n,K)
    logits = -d2 / (2 * (temp**2))
    logits -= logits.max(axis=1, keepdims=True)
    pi = np.exp(logits); pi /= pi.sum(axis=1, keepdims=True)
    return pi  # (n,K)

def smooth_warp(coords, amp=0.05):
    """Small smooth displacement: W(x)=x+d(x). Same field used to generate slice-2."""
    x, y = coords[:,0], coords[:,1]
    dx = amp * ( np.sin(2*np.pi*x) * np.cos(np.pi*y) + 0.3*np.sin(3*np.pi*y) )
    dy = amp * ( 0.6*np.sin(2*np.pi*y) - 0.4*np.cos(2*np.pi*x) )
    return coords + np.stack([dx, dy], axis=-1)

def radial_u(coords, centers):
    """Per-type 1D latent u_k(x): angle around type center (wrapped to [-1,1])."""
    # returns (n,K) with u in [-1,1]
    diffs = coords[:,None,:] - centers[None,:,:]  # (n,K,2)
    ang = np.arctan2(diffs[...,1], diffs[...,0])  # (-pi,pi)
    u = ang / np.pi  # (-1,1)
    return u

def zscore(x, eps=1e-8):
    m = x.mean(0, keepdims=True); s = x.std(0, keepdims=True) + eps
    return (x - m)/s, (m, s)

# ---------- main generator ----------
def synthetic_multimodal_tissue(
    nx=30, ny=20, jitter=0.01,          # grid
    K=4, temp=0.08,                     # cell types, boundary softness
    warp_amp=0.06,                      # deformation from slice A -> slice B
    dA=6, dB=4,                         # #features per modality
    batch_amp=(0.2, 0.2),               # spatial low-rank batch effects for A/B
    noise=(0.03, 0.03),                 # observation noise for A/B
    missing=(0.0, 0.0),                 # missingness rates
    seed=0
):
    """
    Returns:
      s, t            : base coords for slice A/B (nearly identical grids)
      X, Z            : modality A/B features
      labels          : hard assignment (n,) in {0..K-1} for slice A (B shares same underlying map)
      pi_s, pi_t      : soft mixing weights (n,K) and (m,K)
      u_s, u_t        : within-type latent trajectories (n,K), (m,K)
      sigma           : index map pairing s[i] -> nearest in t (ground-truth alignment)
      phi_true, psi_true: callables that map base -> modality in noiseless form
    """
    rng = np.random.default_rng(seed)

    # 1) spatial grids (A and B)
    s = hex_grid(nx, ny, spacing=1.0, jitter=jitter, rng=rng)       # slice A
    t_ideal = s.copy()                                              # same grid
    t = smooth_warp(t_ideal, amp=warp_amp).astype(np.float32)       # warped slice B
    n, m = len(s), len(t)

    # 2) K cell-type centers (on A's grid for simplicity)
    #    pick centers by k-means++-like seeding for spread
    idx = rng.choice(n)
    centers = [s[idx]]
    for _ in range(1, K):
        d2 = ((s[:,None,:] - np.array(centers)[None,:,:])**2).sum(-1).min(axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        centers.append(s[rng.choice(n, p=probs)])
    centers = np.stack(centers, axis=0)  # (K,2)

    # 3) cell-type soft assignments (same latent regions for both slices)
    pi_s = soft_voronoi_mixture(s, centers, temp=temp)  # (n,K)
    # For t, evaluate assignments in the inverse-warp coordinates (align latent regions)
    t_inv = t - (smooth_warp(t, amp=warp_amp) - t)      # 1-step inverse approx
    pi_t = soft_voronoi_mixture(t_inv, centers, temp=temp)  # (m,K)

    # hard labels (argmax)
    labels = pi_s.argmax(1)

    # 4) within-type latent trajectories u_k(x) (same function, eval at s and t_inv)
    u_s = radial_u(s, centers)      # (n,K), in [-1,1]
    u_t = radial_u(t_inv, centers)  # (m,K)

    # 5) shared latent bank L(x): [pi, u, global coords]
    Ls = np.concatenate([pi_s, u_s, s], axis=1)         # (n, 2K+2)
    Lt = np.concatenate([pi_t, u_t, t_inv], axis=1)     # (m, 2K+2)

    # 6) modality-specific nonlinear maps (same latent, different views)
    # Random “loadings” to mix the latent bank differently across modalities
    WA = rng.normal(size=(Ls.shape[1], dA)) / np.sqrt(Ls.shape[1])
    WB = rng.normal(size=(Lt.shape[1], dB)) / np.sqrt(Lt.shape[1])

    def phi_true(x_bank):
        # A: periodic + tanh + interactions; smooth but rich
        h1 = np.sin(2*np.pi * x_bank @ WA[:, :dA//2])
        h2 = np.tanh(1.2 * x_bank @ WA[:, dA//2:])
        # mild interaction based on within-type u
        if K > 1:
            u_sum = x_bank[:, K:2*K].sum(1, keepdims=True)  # sum over u_k
        else:
            u_sum = x_bank[:, K:K+1]
        return np.concatenate([h1, h2], axis=1) + 0.2*u_sum

    def psi_true(x_bank):
        # B: tanh + polynomial + phasey radial terms
        g1 = np.tanh(1.8 * x_bank @ WB[:, :dB//2])
        g2 = (x_bank @ WB[:, dB//2:])**2 - 0.5
        # phase term from global coords (last 2 dims)
        xg = x_bank[:, -2:]
        phase = 0.15*np.sin(3*np.pi * (0.7*xg[:,0] + 0.3*xg[:,1]))[:,None]
        return np.concatenate([g1, g2], axis=1) + phase

    XA = phi_true(Ls)
    ZB = psi_true(Lt)

    # 7) low-rank spatial batch effects (smooth fields)
    def low_rank_field(coords, rank=2, amp=0.2):
        B = rng.normal(size=(coords.shape[1], rank))
        P = np.sin(2*np.pi*coords @ B) + 0.5*np.cos(np.pi*coords @ B)
        w = rng.normal(size=(rank, 1))
        return amp * (P @ w)  # (n,1)

    bA = low_rank_field(s, amp=batch_amp[0])
    bB = low_rank_field(t, amp=batch_amp[1])
    XA = XA + bA
    ZB = ZB + bB

    # 8) heteroskedastic noise (larger near mixture boundaries)
    edge_s = 1.0 - (pi_s**2).sum(1, keepdims=True)  # 0 in pure regions, ↑ near boundaries
    edge_t = 1.0 - (pi_t**2).sum(1, keepdims=True)
    XA_noisy = XA + noise[0] * (1 + 0.7*edge_s) * rng.standard_normal(XA.shape)
    ZB_noisy = ZB + noise[1] * (1 + 0.7*edge_t) * rng.standard_normal(ZB.shape)

    # 9) missingness (optional)
    if missing[0] > 0:
        mask = rng.random(XA_noisy.shape) < missing[0]
        XA_noisy[mask] = 0.0
    if missing[1] > 0:
        mask = rng.random(ZB_noisy.shape) < missing[1]
        ZB_noisy[mask] = 0.0

    # 10) ground-truth correspondence by nearest neighbor in space
    #     (since t is a small warp of s)
    from sklearn.neighbors import NearestNeighbors
    nbr = NearestNeighbors(n_neighbors=1).fit(t)
    sigma = nbr.kneighbors(s, return_distance=False).ravel()  # s[i] -> t[sigma[i]]

    return {
        's': s.astype(np.float32),
        't': t.astype(np.float32),
        'X': XA_noisy.astype(np.float32),
        'Z': ZB_noisy.astype(np.float32),
        'labels': labels.astype(np.int64),
        'pi_s': pi_s.astype(np.float32),
        'pi_t': pi_t.astype(np.float32),
        'u_s': u_s.astype(np.float32),
        'u_t': u_t.astype(np.float32),
        'sigma': sigma.astype(np.int64),
        'centers': centers.astype(np.float32),
        'phi_true': phi_true,
        'psi_true': psi_true,
    }