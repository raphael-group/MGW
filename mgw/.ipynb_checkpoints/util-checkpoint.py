
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import anndata as ad
import numpy as np, torch, scipy.sparse as sp
from scipy.spatial.distance import cdist
import scanpy as sc
from .gw import solve_gw_ott

import anndata as ad
from sklearn.utils import sparsefuncs as sf
from sklearn.preprocessing import normalize as l2_normalize_rows

def _barycentric_right(P, Y, a=None, eps=1e-12):
    P = np.asarray(P, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = P.shape
    if a is None:
        a = P.sum(axis=1)  # row marginals
    denom = (a + eps)[:, None]
    return (P @ Y) / denom

def project_informative_features(
    path_X, path_Z,
    *,
    X_layer=None, Z_layer=None,
    PCA_comp=20, CCA_comp=5,
    n_downsample=10000,
    log1p_X=True,
    log1p_Z=True,
    feeler_epsilon=1e-3,
    verbose=True,
    adata_X=None, adata_Z=None,
    use_pca_X=True,
    use_pca_Z=True,
    spatial_key="spatial",
    spatial_only=True,
    feature_only=False
):
    if adata_X is not None and adata_Z is not None:
        st_xy = adata_X.obsm[spatial_key]
        sm_xy = adata_Z.obsm[spatial_key]
    else:
        st_xy, adata_X = _ann_load(path_X, X_layer)
        sm_xy, adata_Z = _ann_load(path_Z, Z_layer)
    
    xs = normalize_coords_to_unit_square(st_xy)
    xt = normalize_coords_to_unit_square(sm_xy)
    
    '''
    X_pca = get_or_compute_pca(adata_X, layer=X_layer, n_comps=PCA_comp, log1p=log1p_features)
    Z_pca = get_or_compute_pca(adata_Z, layer=Z_layer, n_comps=PCA_comp, log1p=log1p_features)
    '''
    if use_pca_X:
        X_pca = get_or_compute_pca(adata_X, layer=X_layer, n_comps=PCA_comp, log1p=log1p_X)
    else:
        X_raw = adata_X.X.toarray() if sp.issparse(adata_X.X) else np.asarray(adata_X.X)
        if log1p_X:
            X_raw = np.log1p(X_raw)
        if verbose:
            print(f"Using raw (non-PCA) features on X→{X_pca.shape}")
        X_pca = X_raw
    if use_pca_Z:
        Z_pca = get_or_compute_pca(adata_Z, layer=Z_layer, n_comps=PCA_comp, log1p=log1p_Z)
    else:
        Z_raw = adata_Z.X.toarray() if sp.issparse(adata_Z.X) else np.asarray(adata_Z.X)
        if log1p_Z:
            Z_raw = np.log1p(Z_raw)
        if verbose:
            print(f"Using raw (non-PCA) features on Z→{Z_pca.shape}")
        Z_pca = Z_raw
    
    if verbose:
        print("Feature shapes -> X:", X_pca.shape, "Z:", Z_pca.shape)
    
    nX = xs.shape[0]; nZ = xt.shape[0]
    
    if n_downsample is not None:
        rng = np.random.default_rng(42)
        
        kZ = min([n_downsample, nZ])
        idxZ = rng.choice(nZ, size=kZ, replace=False)
        xt_gw   = xt[idxZ]
        Zpca_gw = Z_pca[idxZ]
        
        kX = min([n_downsample, nX])
        idxX = rng.choice(nX, size=kX, replace=False)
        xs_gw   = xs[idxX]
        Xpca_gw = X_pca[idxX]
        
        if verbose:
            print(f"Feeler GW sizes: X-side {xs_gw.shape[0]}, Z-side {xt_gw.shape[0]}")
    else:
        xs_gw, Xpca_gw = xs, X_pca
        xt_gw, Zpca_gw = xt, Z_pca
        idxZ = np.arange(nZ)
    
    # ----------------------------
    # Fused intra-domain costs (Hadamard: spatial × feature)
    # ----------------------------
    def _norm(C):
        q = np.quantile(C[np.triu_indices_from(C, 1)], 0.99)
        return C/(q + 1e-12)
    
    if spatial_only:
        # Normalize costs to comparable scale (quantile) to help solvers.
        Dxs, Dxt  = cdist(xs_gw, xs_gw), cdist(xt_gw, xt_gw)
        C1,C2 = _norm(Dxs), _norm(Dxt)
    elif feature_only:
        Dfx, Dfz  = cdist(Xpca_gw, Xpca_gw), cdist(Zpca_gw, Zpca_gw)
        C1, C2 = _norm(Dfx), _norm(Dfz)
    else:
        Dxs  = cdist(xs_gw, xs_gw)
        Dxt  = cdist(xt_gw, xt_gw)
        Dfx  = cdist(Xpca_gw, Xpca_gw)
        Dfz  = cdist(Zpca_gw, Zpca_gw)
        C1 = _norm(Dxs) * _norm(Dfx)
        C2 = _norm(Dxt) * _norm(Dfz)
    
    print('Solving feeler feature alignment.')
    gw_params = dict(verbose=True, inner_maxit=2000, outer_maxit=2000,
                 inner_tol=1e-6, outer_tol=1e-6, epsilon=feeler_epsilon)
    
    P = solve_gw_ott(C1, C2, **gw_params)
    Z_bary_on_X = _barycentric_right(P, Zpca_gw)   # shape (|X_gw|, dZ)
    
    X_scaler = StandardScaler(with_mean=True, with_std=True).fit(Xpca_gw)
    Z_scaler = StandardScaler(with_mean=True, with_std=True).fit(Z_bary_on_X)
    
    X_std = X_scaler.transform(Xpca_gw)
    Z_std = Z_scaler.transform(Z_bary_on_X)
    
    n_comp = min(CCA_comp, X_std.shape[1], Z_std.shape[1])
    
    print('Computing CCA Components')
    cca = CCA(n_components=n_comp, max_iter=1000, scale=False)
    cca.fit(X_std, Z_std)
    
    X_cca_full = X_scaler.transform(X_pca) @ cca.x_weights_
    Z_cca_full = Z_scaler.transform(Z_pca) @ cca.y_weights_
    
    if verbose:
        print("CCA dims:", X_cca_full.shape[1], " (applied to full sets)")
    
    return dict(
        xs=xs, xt=xt,
        X_pca=X_pca, Z_pca=Z_pca,
        P_feeler=P,                          # on (X_all, Z_ds) shapes
        Z_bary_on_X=Z_bary_on_X,             # size n_X × PCA_comp
        X_cca_full=X_cca_full,               # size n_X × CCA_comp
        Z_cca_full=Z_cca_full,               # size n_Z × CCA_comp
        cca=cca,
        scalers=dict(X_scaler=X_scaler, Z_scaler=Z_scaler),
        meta=dict(
            idxZ_feeler=idxZ,
            PCA_comp=PCA_comp, CCA_comp=n_comp,
            feeler_epsilon=feeler_epsilon
        )
    )

def normalize_range(ys_t):
    ys_t  = ys_t - ys_t.mean(0, keepdim=True)
    ys_t /= ys_t.std(0, keepdim=True) + 1e-8
    return ys_t

def normalize_range_np(ys):
    ys  = ys - ys.mean(0)
    ys /= ys.std(0) + 1e-8
    return ys

'''
----
'''

def downsample_per_slice(adata, max_obs=10_000, random_state=42):
    """Return a copy of `adata` with ≤max_obs observations."""
    if adata.n_obs <= max_obs:
        return adata.copy()
    rng = np.random.default_rng(random_state)
    idx = rng.choice(adata.n_obs, size=max_obs, replace=False)
    return adata[idx].copy()

def cca_from_unpaired(sp_xy_A, sp_xy_B, feats_A, feats_B, n_components=2, K=8):
    """Return (A_cca, B_cca, cca, sc_A, sc_B) given coords + neural feats.
       Pairs B→A via barycentric KNN on coords, fits CCA, and projects ALL points."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import CCA
    import numpy as np

    nbrs = NearestNeighbors(n_neighbors=K).fit(sp_xy_B)
    dist, ind = nbrs.kneighbors(sp_xy_A)
    sigma = np.median(dist) + 1e-12
    W = np.exp(-(dist**2)/(2*sigma**2)); W /= (W.sum(1, keepdims=True) + 1e-12)
    B_on_A = (W[..., None] * feats_B[ind]).sum(axis=1)
    
    sc_A = StandardScaler().fit(feats_A)
    sc_B = StandardScaler().fit(B_on_A)
    XA = sc_A.transform(feats_A)
    YB = sc_B.transform(B_on_A)
    
    cca = CCA(n_components=n_components, max_iter=5000)
    cca.fit(XA, YB)

    A_cca = ((feats_A - sc_A.mean_) / sc_A.scale_) @ cca.x_weights_
    B_cca = ((feats_B - sc_B.mean_) / sc_B.scale_) @ cca.y_weights_
    return A_cca, B_cca, cca, sc_A, sc_B

def _top_var_indices(X, k):
    """Return indices of the top-k most variable features (columns), sparse-safe."""
    if k is None or k >= X.shape[1]:
        return np.arange(X.shape[1])

    if sp.issparse(X):
        # mean/var across axis=0 (features) for CSR/CSC
        # returns np arrays of shape (n_features,)
        mean, var = sf.mean_variance_axis(X, axis=0)
        var = np.asarray(var).ravel()
    else:
        var = np.var(np.asarray(X), axis=0)

    k = max(int(k), 1)
    return np.argsort(var)[-k:]

def pca_from(adata, *, layer=None, n_comps=30, log1p=True, scale_if_dense=True, key="X_pca"):
    """
    Return n_comps PCA coords from an AnnData (handles sparse vs dense), caching to .obsm[key].
    """
    import scipy.sparse as sp
    ad_tmp = adata.copy()
    if layer is not None:
        if layer not in ad_tmp.layers:
            raise KeyError(f"Layer {layer!r} not found.")
        ad_tmp.X = ad_tmp.layers[layer]

    if log1p:
        sc.pp.log1p(ad_tmp)

    is_sparse = sp.issparse(ad_tmp.X)
    zero_center = not is_sparse  # TruncatedSVD path if sparse

    if scale_if_dense and zero_center:
        sc.pp.scale(ad_tmp, max_value=10)

    sc.tl.pca(
        ad_tmp,
        n_comps=n_comps,
        zero_center=zero_center,
        svd_solver="arpack" if zero_center else "auto",
    )
    Xp = np.asarray(ad_tmp.obsm["X_pca"], dtype=np.float64)
    adata.obsm[key] = Xp  # cache
    return Xp[:, :n_comps]

'''
def pca_from(adata, *, layer=None, n_comps=30, log1p=True, scale_if_dense=True):
    """Return n_comps PCA coords from an AnnData (handles sparse vs dense)."""
    ad_tmp = adata.copy()
    if layer is not None and layer in ad_tmp.layers:
        ad_tmp.X = ad_tmp.layers[layer]

    # optional log1p
    if log1p:
        sc.pp.log1p(ad_tmp)

    # sparse matrices can't be mean-centered in memory-friendly way; use zero_center=False (TruncatedSVD path)
    is_sparse = sp.issparse(ad_tmp.X)
    zero_center = not is_sparse

    # scale to unit variance if dense (common before PCA)
    if scale_if_dense and zero_center:
        sc.pp.scale(ad_tmp, max_value=10)  # clip heavy tails a bit

    # run PCA
    sc.tl.pca(
        ad_tmp,
        n_comps=n_comps,
        zero_center=zero_center,           # False for sparse -> TruncatedSVD-style
        svd_solver="arpack" if zero_center else "auto"
    )
    return np.asarray(ad_tmp.obsm["X_pca"], dtype=np.float64)
'''

def _to_dense(X):
    if sp.issparse(X): return X.toarray()
    return np.asarray(X)

def normalize_coords_to_unit_square(Xxy):
    Xxy = np.asarray(Xxy, dtype=np.float64)
    m = Xxy.min(0, keepdims=True)
    M = Xxy.max(0, keepdims=True)
    return (Xxy - m) / (M - m + 1e-12)

def log1p_if(X):
    return np.log1p(X) if log1p_features else X

def zscore_features(X):
    if not zscore_per_feature: return X
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu)/sd

def top_var_idx(X, n_keep):
    if (n_keep is None) or (n_keep >= X.shape[1]): 
        return np.arange(X.shape[1])
    # unbiased sample variance per feature
    v = X.var(axis=0, ddof=1)
    return np.argsort(v)[-n_keep:]

def _ann_load(path, layer=None):
    adata = ad.read_h5ad(path)
    if "spatial" not in adata.obsm:
        raise ValueError(f"{path} missing .obsm['spatial']")
    XY = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    return XY, adata

def ann_load(path, layer=None):
    adata = ad.read_h5ad(path)
    if "spatial" not in adata.obsm:
        raise ValueError(f"{path} missing .obsm['spatial']")
    X = _to_dense(adata.layers[layer]) if (layer is not None and layer in adata.layers) else _to_dense(adata.X)
    XY = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    return XY, X, adata

def get_or_compute_pca(adata, layer=None, n_comps=30, log1p=True, key="X_pca"):
    """Return existing PCA if present; otherwise compute it."""
    # Check if PCA already exists in adata.obsm
    if key in adata.obsm:
        print(f"Using precomputed PCA ({adata.obsm[key].shape[1]} comps).")
        X_pca = np.asarray(adata.obsm[key][:, :n_comps])
    else:
        print(f"Computing PCA ({n_comps} comps)...")
        X_pca = pca_from(adata, layer=layer, n_comps=n_comps, log1p=log1p)
        adata.obsm[key] = X_pca  # optionally cache it
    #X_pca = normalize(X_pca)
    return X_pca
