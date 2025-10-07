import numpy as np
import matplotlib.pyplot as plt

def scatter_colored(coords, values, s=10):
    plt.figure()
    plt.scatter(coords[:,0], coords[:,1], c=values, s=s)
    plt.axis('equal'); plt.title('Colored scatter'); plt.show()

def plot_alignment_lines_dense(xs, xt, P, k_per_point=3, alpha=0.05, lw=0.5):
    """
    Draw up to k_per_point connections per source x_i to target x_j
    with largest coupling weight. Covers all points but remains readable.
    """
    if not isinstance(P, np.ndarray):
        P = P.toarray() if hasattr(P, "toarray") else np.array(P)
    xs = np.asarray(xs)
    xt = np.asarray(xt)

    n1, n2 = P.shape
    offset = xs[:,0].max() - xt[:,0].min() + 1.25*(np.ptp(xs[:,0]) + np.ptp(xt[:,0]))

    plt.figure(figsize=(10,5))
    for i in range(n1):
        # find top-k_per_point targets
        j_top = np.argsort(-P[i,:])[:k_per_point]
        for j in j_top:
            w = P[i,j] / P[i,:].max()  # normalize per-row for transparency
            plt.plot([xs[i,0], xt[j,0]+offset],
                     [xs[i,1], xt[j,1]],
                     color="k", alpha=alpha*w, lw=lw)
    plt.scatter(xs[:,0], xs[:,1], s=8, c="C0", label="Source")
    plt.scatter(xt[:,0]+offset, xt[:,1], s=8, c="C1", label="Target")
    plt.axis("equal"); plt.axis("off")
    plt.title(f"GW Coupling: top {k_per_point} per point")
    plt.legend(frameon=False)
    plt.show()

def plot_alignment_lines(xs, xt, P, top=500, alpha=0.1):
    """
    Plot top-weight alignment lines between xs and xt, side-by-side.
    """
    if isinstance(P, np.ndarray):
        pass
    else:
        P = P.toarray() if hasattr(P, 'toarray') else np.array(P)

    n1, n2 = P.shape
    # pick top pairs
    flat = P.ravel()
    idx = np.argpartition(flat, -min(top, flat.size))[-min(top, flat.size):]
    idx = idx[np.argsort(-flat[idx])]

    xs = np.asarray(xs); xt = np.asarray(xt)
    offset = xs[:, 0].max() - xt[:, 0].min() + 1.25 * (np.ptp(xs[:, 0]) + np.ptp(xt[:, 0]))
    plt.figure(figsize=(10,5))
    for k in idx:
        i = k // n2; j = k % n2
        plt.plot([xs[i,0], xt[j,0]+offset], [xs[i,1], xt[j,1]], 'k-', alpha=alpha, lw=0.5)
    plt.scatter(xs[:,0], xs[:,1], s=10, c='C0')
    plt.scatter(xt[:,0]+offset, xt[:,1], s=10, c='C1')
    plt.axis('equal'); plt.axis('off'); plt.title('Top aligned pairs'); plt.show()

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def _edge_len_local(x_i, x_j, Gi, Gj):
    # symmetrized local arc length under pull-back metrics Gi, Gj
    dx_i = torch.as_tensor(x_j - x_i, dtype=Gi.dtype, device=Gi.device)
    dx_j = torch.as_tensor(x_i - x_j, dtype=Gj.dtype, device=Gj.device)
    li = torch.sqrt(torch.clamp(dx_i @ Gi @ dx_i, min=0.0))
    lj = torch.sqrt(torch.clamp(dx_j @ Gj @ dx_j, min=0.0))
    return 0.5 * (li + lj)

def geodesics_with_predecessors(coords_np, G_tensors, knn_csr):
    """
    coords_np: (n,2) numpy
    G_tensors: (n,2,2) torch (or numpy convertible) SPD tensors for each point
    knn_csr:   sparse kNN adjacency on coords (unweighted)
    Returns: (dist_matrix, predecessors)
    """
    if isinstance(G_tensors, np.ndarray):
        G_tensors = torch.from_numpy(G_tensors)
    device = G_tensors.device
    coords_t = torch.from_numpy(coords_np).to(device=device, dtype=G_tensors.dtype)

    rows, cols = knn_csr.nonzero()
    weights = []
    for i, j in zip(rows, cols):
        w = _edge_len_local(coords_np[i], coords_np[j], G_tensors[i], G_tensors[j])
        weights.append(float(w.detach().cpu().item()))
    W = csr_matrix((weights, (rows, cols)), shape=knn_csr.shape)
    D, P = shortest_path(W, directed=False, unweighted=False, return_predecessors=True)
    return D, P

def reconstruct_path(predecessors, i, j):
    """Return list of node indices along shortest path i -> j."""
    path = [j]
    k = j
    while k != i and k != -9999:
        k = predecessors[i, k]
        if k == -9999:
            return []  # no path
        path.append(k)
    return path[::-1]

def plot_geodesic_on_base(coordinates, predecessors, start, end, values=None, title='Geodesic path'):
    """
    coordinates: (n,2) numpy or torch
    predecessors: predecessor matrix from shortest_path
    values: optional scalar field to color the points (same length n)
    """
    coords = np.asarray(coordinates)
    #coordinates.detach().cpu().numpy() if hasattr(coordinates, 'device') else np.asarray(coordinates)
    path_idx = reconstruct_path(predecessors, start, end)

    plt.figure(figsize=(6,6))
    if values is not None:
        plt.scatter(coords[:,0], coords[:,1], c=np.asarray(values), s=6, cmap='viridis')
        plt.colorbar()
    else:
        plt.scatter(coords[:,0], coords[:,1], s=6, c='lightgray')

    if path_idx:
        P = coords[path_idx]
        plt.plot(P[:,0], P[:,1], '-o', ms=3, lw=1.5, c='k')
        plt.scatter([coords[start,0], coords[end,0]],
                    [coords[start,1], coords[end,1]],
                    c=['C3','C2'], s=40, marker='*')

    plt.title(title); plt.axis('equal'); plt.show()

import numpy as np, torch, matplotlib.pyplot as plt
from scipy.interpolate import griddata

def predict_on_model(model, arr_np):
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    with torch.no_grad():
        return model(torch.from_numpy(arr_np).to(device, dtype)).cpu().numpy()

def field_to_grid(coords, values, nx=200, ny=160, method='linear'):
    x, y = coords[:,0], coords[:,1]
    gx = np.linspace(x.min(), x.max(), nx); gy = np.linspace(y.min(), y.max(), ny)
    GX, GY = np.meshgrid(gx, gy)
    GZ = griddata(np.c_[x,y], values, (GX, GY), method=method)
    if np.isnan(GZ).any():
        GZ = np.where(np.isnan(GZ), griddata(np.c_[x,y], values, (GX, GY), method='nearest'), GZ)
    return GX, GY, GZ

def plot_fit_on_cloud(coords, y_true, y_pred, title_true='raw', title_pred='pred'):
    lo = np.percentile(np.concatenate([y_true, y_pred]), 5)
    hi = np.percentile(np.concatenate([y_true, y_pred]), 95)

    GX, GY, GT = field_to_grid(coords, y_true)
    _,  _, GP = field_to_grid(coords, y_pred)

    fig, axes = plt.subplots(1,3, figsize=(13,4), constrained_layout=True)
    im0 = axes[0].imshow(GT, origin='lower',
                         extent=[GX.min(), GX.max(), GY.min(), GY.max()],
                         cmap='viridis', vmin=lo, vmax=hi, aspect='equal')
    axes[0].set_title(title_true); axes[0].axis('off')

    im1 = axes[1].imshow(GP, origin='lower',
                         extent=[GX.min(), GX.max(), GY.min(), GY.max()],
                         cmap='viridis', vmin=lo, vmax=hi, aspect='equal')
    axes[1].set_title(title_pred); axes[1].axis('off')

    im2 = axes[2].imshow(GP-GT, origin='lower',
                         extent=[GX.min(), GX.max(), GY.min(), GY.max()],
                         cmap='coolwarm', aspect='equal')
    axes[2].set_title('residual (pred - raw)'); axes[2].axis('off')

    fig.colorbar(im1, ax=axes[:2], fraction=0.03, pad=0.02)
    fig.colorbar(im2, ax=[axes[2]], fraction=0.03, pad=0.02)
    plt.show()

def fit_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1); y_pred = y_pred.reshape(-1)
    mse = np.mean((y_pred - y_true)**2)
    var = np.var(y_true) + 1e-12
    r2  = 1 - mse/var
    corr = np.corrcoef(y_true, y_pred)[0,1]
    return {'mse': mse, 'r2': r2, 'corr': corr}

def procrustes_from_coupling(X, Y, P, ensure_rotation=True):
    """
    Align Y to X using a weighted Procrustes fit with weights from coupling P.

    X: (n,d)  source coords
    Y: (m,d)  target coords
    P: (n,m)  coupling (sum(P) ~ 1 or any positive mass)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    P = np.asarray(P, dtype=float)

    m = P.sum()
    if m <= 0:
        raise ValueError("Coupling has zero mass.")

    # weights over points
    w_x = P.sum(axis=1)          # (n,)
    w_y = P.sum(axis=0)          # (m,)

    # weighted centroids
    x_bar = (w_x @ X) / m        # (d,)
    y_bar = (w_y @ Y) / m        # (d,)

    # center
    Xc = X - x_bar
    Yc = Y - y_bar

    # cross-covariance H = Yc^T P^T Xc  (d x m) @ (m x n) @ (n x d) -> (d x d)
    H = Yc.T @ P.T @ Xc

    # SVD -> rotation
    U, _, Vt = np.linalg.svd(H, full_matrices=False)
    R = Vt.T @ U.T
    if ensure_rotation and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # apply: align Y onto X’s frame
    Y_aligned = Yc @ R + x_bar
    # translation that maps Y -> Y_aligned is: y ↦ (y - y_bar) R + x_bar
    t = x_bar - y_bar @ R

    # X was centered then re-shifted, so return the original X for plotting
    return X, Y_aligned, R, t
