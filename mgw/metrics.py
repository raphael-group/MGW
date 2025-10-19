from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from mgw import plotting
from sklearn.metrics.pairwise import cosine_similarity


'''
def expression_cosine_similarity(X, Z, P=None, sample_pairs=50000, random_state=0):
    """
    Compute cosine similarity between expression profiles of aligned spots.
    Parameters
    ----------
    X, Z : np.ndarray
        Expression matrices of shape (n, d) and (m, d).
    P : np.ndarray
        Coupling matrix (n, m).
    sample_pairs : int
        Number of (i,j) pairs to sample.
    random_state : int
        Random seed for reproducibility.
    Returns
    -------
    mean_cosine : float
        Mean cosine similarity across sampled pairs.
    """
    
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    
    # Normalize feature vectors for cosine similarity
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    
    Pn = P / (P.sum() + 1e-12)
    
    px = Pn.sum(1)
    idx_x = rng.choice(X.shape[0], size=sample_pairs, p=px)
    idx_z = np.array([
        rng.choice(Z.shape[0], p=Pn[i,:] / (Pn[i,:].sum() + 1e-12))
        for i in idx_x
    ])
    
    # Compute cosine similarities
    cosines = np.einsum('ij,ij->i', Xn[idx_x], Zn[idx_z])
    mean_cosine = float(np.mean(cosines))
    
    print(f"Mean cosine similarity under coupling: {mean_cosine:.4f}")
    return mean_cosine'''

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy import sparse as sp

def evaluate_coupling(xs, xs2, P, A_labels, B_labels, metrics):
    """Return dict with migration and symmetric mean projected AMI for one coupling."""
    mig = float(migration_metrics(xs, xs2, P)['expected_disp'])
    ab = ami_on_projected_labels(P,   A_labels, B_labels)
    ba = ami_on_projected_labels(P.T, B_labels, A_labels)
    ami_mean = 0.5 * (ab["AMI"] + ba["AMI"])
    return {"migration": mig, "projAMI_mean": float(ami_mean)}

def _to_dense_col_norm(P_sub):
    """Return column-normalized dense matrix; safe for all-zero columns."""
    if sp.issparse(P_sub):
        P_sub = P_sub.tocsc(copy=True)
        col_sums = np.asarray(P_sub.sum(axis=0)).ravel()
        nz = col_sums > 0
        P_sub.data /= np.repeat(col_sums[nz], np.diff(P_sub.indptr)[nz])
        return P_sub.toarray(), col_sums
    else:
        col_sums = P_sub.sum(axis=0, keepdims=True)
        Pn = np.divide(P_sub, np.maximum(col_sums, 1e-12), where=True)
        return Pn, col_sums.ravel()

def ami_on_projected_labels(P, labels_A, labels_B, conf_thresh=0.0):
    """
    Project A->B using P, restricting to the common label set, then compute AMI/ARI/Acc.
    Returns dict with metrics and the projected labels/confidence for the evaluated B indices.
    """
    labels_A = np.asarray(labels_A)
    labels_B = np.asarray(labels_B)
    # common label universe
    common = np.intersect1d(np.unique(labels_A), np.unique(labels_B))
    maskA = np.isin(labels_A, common)
    maskB = np.isin(labels_B, common)
    if maskA.sum() == 0 or maskB.sum() == 0:
        raise ValueError("No overlap in label sets between A and B.")

    # restrict P to common labels
    P_sub = P[maskA][:, maskB]
    Pn, col_sums = _to_dense_col_norm(P_sub)  # shape (nA_common, nB_common)

    # map common labels to row-indices in A-subset
    common_to_rows = {lab: np.where(maskA & (labels_A == lab))[0] for lab in common}

    # per B column: label probabilities and prediction
    # Compute p(l | j) = sum_{i in A_l} Pn[i, j]
    nB = Pn.shape[1]
    probs = np.zeros((nB, common.size), dtype=float)
    for ell, lab in enumerate(common):
        probs[:, ell] = Pn[ np.ix_(np.isin(np.where(maskA)[0], common_to_rows[lab]), np.arange(nB)) ].sum(axis=0)

    pred_idx = probs.argmax(axis=1)
    pred_lab = common[pred_idx]
    conf = probs.max(axis=1)

    # filter by confidence threshold (optional)
    keep = conf >= conf_thresh
    true_lab = labels_B[maskB][keep]
    pred_lab = pred_lab[keep]

    # metrics
    ami = adjusted_mutual_info_score(true_lab, pred_lab)
    ari = adjusted_rand_score(true_lab, pred_lab)
    acc = accuracy_score(true_lab, pred_lab)

    return {
        "AMI": ami,
        "ARI": ari,
        "ACC": acc,
        "keep_mask_B_common": keep,
        "B_true_eval": true_lab,
        "B_pred_eval": pred_lab,
        "B_conf_eval": conf[keep],
        "common_labels": common
    }

def expression_cosine_similarity(X, Z, P, eps=1e-12):
    from numpy.linalg import norm
    Xn = X / (norm(X, axis=1, keepdims=True) + eps)
    Zn = Z / (norm(Z, axis=1, keepdims=True) + eps)
    C = Xn @ Zn.T  # cosine matrix
    return (P * C).sum() / (P.sum() + eps)

def Alignment_Clus_Metrics(X, Z, P, k=4, random_state=0):
    
    # Cluster each modality independently
    clx = KMeans(k, random_state=random_state).fit_predict(X)
    clz = KMeans(k, random_state=random_state).fit_predict(Z)
    
    # Compute weighted AMI under coupling P (normalized joint)
    P_norm = P / P.sum()
    
    # Sample from P to estimate AMI under coupling
    n_samp = 20000
    idx_x = np.random.choice(P.shape[0], size=n_samp, p=P_norm.sum(1))
    idx_z = [np.random.choice(P.shape[1], p = P_norm[i,:] / P_norm[i,:].sum() ) for i in idx_x]
    
    ami = adjusted_mutual_info_score(clx[idx_x], clz[idx_z])
    print("AMI(X,Z) under coupling:", ami)
    ari = adjusted_rand_score(clx[idx_x], clz[idx_z])
    print("ARI(X,Z) under coupling:", ari)
    
    return ami, ari

def _safe_row_probs(row):
    s = row.sum()
    return row / s if s > 0 else np.zeros_like(row)

def _weighted_joint_from_P(P, clx, clz, kx=None, kz=None):
    """Return joint J[a,b] = sum_{i,j} P_ij 1[clx_i=a] 1[clz_j=b]."""
    P = np.asarray(P, dtype=np.float64)
    n1, n2 = P.shape
    if kx is None: kx = int(clx.max()) + 1
    if kz is None: kz = int(clz.max()) + 1
    J = np.zeros((kx, kz), dtype=np.float64)
    for a in range(kx):
        ia = (clx == a)
        if not ia.any(): continue
        Pa = P[ia]                      # (|Ia|, n2)
        # accumulate by columns grouped by clz label
        for b in range(kz):
            jb = (clz == b)
            if jb.any():
                J[a, b] = Pa[:, jb].sum()
    return J

def _entropy(p):
    p = p[p > 0]
    return -(p * np.log(p)).sum()

def _weighted_mi_nmi_from_P(P, clx, clz):
    """Compute MI, NMI from coupling-weighted joint contingency (deterministic)."""
    P = P / P.sum()
    kx = int(clx.max()) + 1
    kz = int(clz.max()) + 1
    J  = _weighted_joint_from_P(P, clx, clz, kx, kz)      # shape (kx, kz)
    if J.sum() <= 0:
        return dict(MI=0.0, NMI=0.0, Hx=0.0, Hz=0.0, J=J)
    J  = J / J.sum()
    px = J.sum(axis=1, keepdims=True)                    # (kx,1)
    pz = J.sum(axis=0, keepdims=True)                    # (1,kz)
    # Mutual information
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = J / (px @ pz)
        ratio[~np.isfinite(ratio)] = 1.0
        MI = (J * np.log(ratio)).sum()
    Hx = _entropy(px.ravel())
    Hz = _entropy(pz.ravel())
    NMI = 0.0 if (Hx + Hz) == 0 else (2 * MI) / (Hx + Hz)
    return dict(MI=MI, NMI=NMI, Hx=Hx, Hz=Hz, J=J)

def cluster_and_scores(X, Z, P, k=4, n_init=10, sample_pairs=20000, random_state=0):
    """
    KMeans each modality; compute:
      - deterministic coupling-weighted NMI/MI,
      - Monte-Carlo AMI/ARI by sampling pairs from P.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X); Z = np.asarray(Z); P = np.asarray(P, dtype=np.float64)
    # cluster
    clx = KMeans(k, n_init=n_init, random_state=random_state).fit_predict(X)
    clz = KMeans(k, n_init=n_init, random_state=random_state).fit_predict(Z)
    # deterministic (exact) weighted NMI/MI
    det = _weighted_mi_nmi_from_P(P, clx, clz)
    # Monte-Carlo AMI/ARI
    Pn = P / P.sum()
    row_m = Pn.sum(axis=1)
    row_m = row_m / row_m.sum()
    rows = rng.choice(Pn.shape[0], size=sample_pairs, p=row_m)
    cols = np.fromiter(
        (rng.choice(Pn.shape[1], p=_safe_row_probs(Pn[i])) for i in rows),
        dtype=np.int64, count=sample_pairs
    )
    ami = adjusted_mutual_info_score(clx[rows], clz[cols])
    ari = adjusted_rand_score(clx[rows], clz[cols])
    return {
        "clusters_X": clx,
        "clusters_Z": clz,
        "NMI_weighted": det["NMI"],
        "MI_weighted": det["MI"],
        "Hx": det["Hx"], "Hz": det["Hz"],
        "AMI_mc": ami, "ARI_mc": ari
    }

from scipy.spatial.distance import cdist

def migration_metrics(xs, xt, P, use_procrustes=True, p=2):
    """
    Compute displacement summaries under P:
      - E[||s - t||^p] and its sqrt (if p=2),
      - barycentric drift per-row, with summary stats.
    """
    xs = np.asarray(xs, dtype=np.float64)
    xt = np.asarray(xt, dtype=np.float64)
    P  = np.asarray(P,  dtype=np.float64)
    Pn = P / P.sum()

    if use_procrustes:
        s_al, t_al, R, tvec = procrustes_from_coupling(xs, xt, Pn)
        S, T = s_al, t_al
    else:
        S, T = xs, xt

    # expected displacement (all pairs)
    D = cdist(S, T)  # Euclidean
    
    disp_p = ( (D ** p) * Pn).sum()
    disp = disp_p ** (1.0 / p) if p != 1 else disp_p

    # barycentric drift per-row: || s_i - sum_j P_ij t_j / sum_j P_ij ||
    row_m = Pn.sum(axis=1, keepdims=True)  # (n,1)
    row_m[row_m == 0] = 1.0
    t_bar = (Pn @ T) / row_m               # (n,2)
    drift = np.linalg.norm(S - t_bar, axis=1)
    stats = {
        "mean": float(drift.mean()),
        "median": float(np.median(drift)),
        "p90": float(np.quantile(drift, 0.90)),
        "p95": float(np.quantile(drift, 0.95)),
        "max": float(drift.max())
    }

    return {
        "expected_disp_p": float(disp_p),
        "expected_disp": float(disp),
        "p": p,
        "barycentric_drift_stats": stats,
        "aligned_coords": dict(S=S, T=T)  # handy for plotting
    }

import numpy as np

def procrustes_from_coupling(X, Y, P, eps=1e-12):
    """
    Weighted Procrustes using P as soft correspondences.
    Returns Xc, Yc_aligned, R, t such that Y ≈ R X + t in least-squares(P).
    X: (n,d), Y: (m,d), P: (n,m) nonnegative
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    # normalize once; avoid divisions by ~0
    mass = P.sum()
    if mass <= eps:
        raise ValueError("Coupling P has (near) zero mass.")
    P = P / mass

    # 1D marginals
    wx = P.sum(axis=1)      # (n,)
    wy = P.sum(axis=0)      # (m,)

    # weighted centroids (shape (1,d)), robust wrt shapes
    X_mean = (wx[:, None] * X).sum(axis=0, keepdims=True) / (wx.sum() + eps)
    Y_mean = (wy[:, None] * Y).sum(axis=0, keepdims=True) / (wy.sum() + eps)

    # center
    Xc = X - X_mean         # (n,d)
    Yc = Y - Y_mean         # (m,d)

    # cross-covariance H = Yc^T P^T Xc  (d×d)
    H = Yc.T @ P.T @ Xc

    # SVD → rotation
    U, _, Vt = np.linalg.svd(H, full_matrices=False)
    R = Vt.T @ U.T
    # enforce proper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # align Y to X: Y_aligned = (Y - Y_mean) @ R^T + t
    Yc_aligned = Yc @ R.T
    t = (Y_mean @ R.T) - X_mean  # translation to map X_mean to aligned Y_mean

    return Xc, Yc_aligned, R, t


