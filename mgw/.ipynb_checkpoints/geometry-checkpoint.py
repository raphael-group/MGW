import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra  # fastest in SciPy for sparse graphs

def _jacobian_pointwise(phi, x_point):
    x_point = x_point.detach().requires_grad_(True)
    y = phi(x_point[None, :]).squeeze(0)  # (dim_f,)
    # Compute Jacobian dy/dx: dim_f x dim_e
    J = []
    for i in range(y.shape[0]):
        grad = torch.autograd.grad(y[i], x_point, retain_graph=True, create_graph=False)[0]
        J.append(grad)
    J = torch.stack(J, dim=0)  # (dim_f, dim_e)
    return J

def pullback_metric_field(phi, X, eps=1e-9):
    """
    For each x in X (N x dim_e), compute g(x) = J_phi(x)^T J_phi(x) + eps I
    Returns a tensor G of shape (N, dim_e, dim_e)
    """
    device = X.device
    N, dim_e = X.shape
    G = torch.zeros((N, dim_e, dim_e), dtype=X.dtype, device=device)
    I = torch.eye(dim_e, dtype=X.dtype, device=device)
    for i in range(N):
        J = _jacobian_pointwise(phi, X[i])
        G[i] = J.T @ J + eps * I
    return G

def knn_graph(coords: np.ndarray, k:int=10):
    nn = NearestNeighbors(n_neighbors=min(k+1, len(coords)), algorithm="kd_tree")
    nn.fit(coords)
    dists, idxs = nn.kneighbors(coords)
    # discard self neighbor (first)
    dists = dists[:, 1:]; idxs = idxs[:, 1:]
    rows, cols, data = [], [], []
    for i in range(len(coords)):
        for d, j in zip(dists[i], idxs[i]):
            rows.append(i); cols.append(j); data.append(d)
            rows.append(j); cols.append(i); data.append(d)  # symmetric
    n = len(coords)
    G = csr_matrix((data, (rows, cols)), shape=(n, n))
    return G

def _edge_length_local(x_i, x_j, Gi, Gj):
    """
    Symmetrized local length under metrics Gi, Gj
    """
    dx_i = torch.from_numpy(x_j - x_i).to(Gi.device).to(Gi.dtype)
    dx_j = torch.from_numpy(x_i - x_j).to(Gj.device).to(Gj.dtype)
    li = torch.sqrt(torch.clamp(dx_i @ Gi @ dx_i, min=0.0))
    lj = torch.sqrt(torch.clamp(dx_j @ Gj @ dx_j, min=0.0))
    return 0.5 * (li + lj)

def geodesic_distances(coords: np.ndarray, G_tensors: torch.Tensor, knn_csr: csr_matrix):
    """
    Build a weighted graph with edge weights equal to local Riemannian arc lengths,
    then compute all-pairs shortest paths.
    """
    device = G_tensors.device
    coords_t = torch.from_numpy(coords).to(device=device, dtype=G_tensors.dtype)
    rows, cols = knn_csr.nonzero()
    weights = []
    for i, j in zip(rows, cols):
        w = _edge_length_local(coords[i], coords[j], G_tensors[i], G_tensors[j])
        weights.append(float(w.detach().cpu().item()))
    W = csr_matrix((weights, (rows, cols)), shape=knn_csr.shape)
    dist = shortest_path(W, directed=False, unweighted=False)
    return dist  # numpy array (n x n)

def geodesic_distances_fast(coords: np.ndarray,
                            G_tensors: torch.Tensor,  # (n,2,2) on GPU or CPU
                            knn_csr: csr_matrix,
                            make_symmetric: bool = True,
                            return_predecessors: bool = False):
    """
    Vectorized edge construction + Dijkstra APSP.
    coords: (n,2) numpy
    G_tensors: (n,2,2) torch tensor (ideally on CUDA)
    knn_csr: sparse adjacency (bool/0-1 or weighted); only the sparsity pattern is used.
    """
    # 1) Vectorized edge weights on the device of G_tensors
    device = G_tensors.device
    dtype  = G_tensors.dtype
    
    rows, cols = knn_csr.nonzero()               # numpy arrays
    rows_t = torch.from_numpy(rows).to(device)
    cols_t = torch.from_numpy(cols).to(device)
    
    x = torch.from_numpy(coords).to(device=device, dtype=dtype)  # (n,2)
    
    # Differences per edge: Δx = x_j - x_i
    dx = x[cols_t] - x[rows_t]                                  # (E,2)

    # Midpoint metric: G_mid = 0.5*(G_i + G_j)
    Gi = G_tensors[rows_t]                                      # (E,2,2)
    Gj = G_tensors[cols_t]                                      # (E,2,2)
    Gm = 0.5 * (Gi + Gj)                                        # (E,2,2)
    
    # Quadratic form Δx^T Gm Δx, then sqrt
    # einsum: (E,2),(E,2,2),(E,2) -> (E,)
    q = torch.einsum('ei,eij,ej->e', dx, Gm, dx).clamp_min(0.0)
    w = torch.sqrt(q)
    
    # Move once to CPU
    weights = w.detach().cpu().numpy()
    
    # 2) Build sparse weight matrix in one go
    W = csr_matrix((weights, (rows, cols)), shape=knn_csr.shape)
    if make_symmetric:
        # ensure symmetry cheaply (prefer min to avoid double counting)
        W = W.minimum(W.T)
    
    # 3) All-pairs shortest paths (Dijkstra is the fastest in SciPy)
    dist = dijkstra(W, directed=False, return_predecessors=return_predecessors)
    
    return dist  # (n,n) or (predecessors, dist) if requested

def pairwise_squared_geodesic(dist_matrix: np.ndarray):
    return dist_matrix ** 2

