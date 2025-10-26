# mgw.py
import os, numpy as np, torch, scipy.sparse as sp
from typing import Optional, Dict, Any
from scipy.spatial.distance import cdist
from mgw import util, models, geometry, plotting
from mgw.gw import solve_gw_ott
from mgw.plotting import fit_metrics
from sklearn.preprocessing import normalize

# ----------------------------
# Small helpers
# ----------------------------
def _to_unit_square(x: np.ndarray) -> np.ndarray:
    return util.normalize_coords_to_unit_square(np.asarray(x, dtype=float))

def _norm_geod(D: np.ndarray) -> np.ndarray:
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    q = np.quantile(D[np.triu_indices_from(D, 1)], 0.99)
    return D / (q + 1e-12)

def _maybe_subsample(adata, max_obs: Optional[int], seed: int = 42):
    if max_obs is None or adata.n_obs <= max_obs:
        return adata.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=max_obs, replace=False)
    return adata[idx].copy()

def mgw_align(
    A, B,
    *,
    # features / embeddings
    PCA_comp: int = 30,
    CCA_comp: int = 3,
    use_cca_feeler: bool = True,
    feeler_downsample: Optional[int] = 8000,   # downsample size per slice for the CCA "feeler"
    log1p_X: bool = True,
    log1p_Z: bool = True,
    use_pca_X: bool = True,
    use_pca_Z: bool = True,
    
    # neural fields
    widths: tuple = (128, 256, 256, 128),
    lr: float = 1e-3,
    niter: int = 20_000,
    print_every: int = 1_000,
    
    # metric + graph
    knn_k: int = 12,
    geodesic_eps: float = 1e-2,
    
    # GW solver
    gw_params: Optional[Dict[str, Any]] = None,
    
    # device / dtype
    device: Optional[str] = None,          # "cuda" / "cpu" / None→auto
    torch_default_dtype: torch.dtype = torch.float64,
    
    # caching
    save_dir: Optional[str] = None,        # if set, save models & P here
    tag: Optional[str] = None,             # suffix for filenames
    
    # speed/control
    max_obs_A: Optional[int] = None,       # subsample A to at most this many spots (None=keep all)
    max_obs_B: Optional[int] = None,       # subsample B to at most this many spots
    verbose: bool = True,
    plot_net: bool = False,
    
    # feeler alignment for CCA
    spatial_only=True,
    feature_only=False
):
    """
    Run Manifold-GW alignment end-to-end and return a dict with the coupling and intermediates.

    Inputs:
      A, B: AnnData objects with .obsm['spatial'] and expression in .X (or layers you set earlier).

    Returns:
      dict with keys:
        P                : (nA, nB) GW coupling (numpy)
        xs, xs2          : spatial coords (unit-square) used
        X_pca, Z_pca     : PCA features (numpy)
        X_rep, Z_rep     : features used to train φ, ψ  (CCA-processed if use_cca_feeler else PCA)
        phi, psi         : trained torch models
        G_M, G_N         : pullback metric tensors (CPU numpy/torch tensors)
        C_M, C_N         : squared normalized geodesic costs fed to GW
        config           : resolved config you ran with
    """

    # ----------------------------
    # Setup
    # ----------------------------
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_dtype(torch_default_dtype)
    if verbose:
        print(f"[mgw] device={device} default_dtype={torch.get_default_dtype()}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        suffix = ("" if tag is None else f"_{tag}")
    else:
        suffix = ""

    # Optional subsampling for tractability
    A_ = _maybe_subsample(A, max_obs_A)
    B_ = _maybe_subsample(B, max_obs_B)

    # ----------------------------
    # 1) Spatial coordinates (unit-square)
    # ----------------------------
    xs  = _to_unit_square(A_.obsm["spatial"])
    xs2 = _to_unit_square(B_.obsm["spatial"])

    # ----------------------------
    # 2) Feature preprocessing: PCA then (optional) CCA feeler
    # ----------------------------
    # Reuse existing PCA if present, else compute
    def _get_pca(adata, n_comps, log1p):
        if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= n_comps:
            return np.asarray(adata.obsm["X_pca"][:, :n_comps], dtype=np.float64)
        # compute PCA fresh
        return util.pca_from(adata, n_comps=n_comps, log1p=log1p)
    
    if use_pca_X:
        X_pca = _get_pca(A_, n_comps=PCA_comp, log1p=log1p_X)
    else:
        X_pca = A_.X.toarray() if sp.issparse(A_.X) else np.asarray(A_.X)
    
    if use_pca_Z:
        Z_pca = _get_pca(B_, n_comps=PCA_comp, log1p=log1p_Z)
    else:
        Z_pca = B_.X.toarray() if sp.issparse(B_.X) else np.asarray(B_.X)
        if log1p_Z:
            Z_pca = np.log1p(Z_pca)
        if verbose:
            print(f"[mgw] using raw features (no PCA): X={X_pca.shape}, Z={Z_pca.shape}")
    
    if use_cca_feeler:
        if verbose:
            print("[mgw] running CCA feeler (PCA → GW on fused costs → barycenter → CCA)")
        feeler = util.project_informative_features(
            path_X=None, path_Z=None,
            adata_X=A_, adata_Z=B_,
            PCA_comp=PCA_comp, CCA_comp=CCA_comp,
            n_downsample=feeler_downsample,
            log1p_X=log1p_X,
            log1p_Z=log1p_Z,
            verbose=verbose
        )
        X_rep = feeler["X_cca_full"]
        Z_rep = feeler["Z_cca_full"]
    
    else:
        if verbose:
            print("[mgw] skipping CCA; using PCA features directly")
        X_rep, Z_rep = X_pca, Z_pca
    
    xs_t  = torch.from_numpy(xs).to(device)
    xs2_t = torch.from_numpy(xs2).to(device)
    
    ys_t  = torch.from_numpy(normalize(X_rep)).to(device) #util.normalize_range(torch.from_numpy(X_rep).to(device))
    ys2_t = torch.from_numpy(normalize(Z_rep)).to(device) #util.normalize_range(torch.from_numpy(Z_rep).to(device))

    dim_e, dim_f_M, dim_f_N = 2, ys_t.shape[1], ys2_t.shape[1]
    if verbose:
        print(f"[mgw] dims: E={dim_e}, F_M={dim_f_M}, F_N={dim_f_N}")

    # ----------------------------
    # 3) Train neural fields φ, ψ : (coords)->(features)
    # ----------------------------
    # (cache if requested)
    phi_path = psi_path = None
    if save_dir:
        phi_path = os.path.join(save_dir, f"phi{suffix}.pt")
        psi_path = os.path.join(save_dir, f"psi{suffix}.pt")

    phi = models.PhiModel(dim_e, dim_f_M, widths=widths).to(device)
    psi = models.PhiModel(dim_e, dim_f_N, widths=widths).to(device)

    if save_dir and os.path.isfile(phi_path) and os.path.isfile(psi_path):
        if verbose: print("[mgw] loading cached φ, ψ")
        phi.load_state_dict(torch.load(phi_path, map_location=device))
        psi.load_state_dict(torch.load(psi_path, map_location=device))
        phi.eval(); psi.eval()
    else:
        if verbose: print("[mgw] training φ, ψ")
        phi = models.train_phi(phi, xs_t,  ys_t,  lr=lr, niter=niter, print_every=print_every, device=device)
        psi = models.train_phi(psi, xs2_t, ys2_t, lr=lr, niter=niter, print_every=print_every, device=device)
        phi.eval(); psi.eval()
        if save_dir:
            torch.save(phi.state_dict(), phi_path)
            torch.save(psi.state_dict(), psi_path)
    
    if plot_net:
        rng = np.random.default_rng(0)
        for k in rng.choice(dim_f_M, size=min(5, dim_f_M), replace=False):
            X_pred = plotting.predict_on_model(phi, xs)
            plotting.plot_fit_on_cloud(xs, ys_t[:,k].cpu().numpy(), X_pred[:,k],
                                       title_true=f'ST feat{k} (true)', title_pred=f'φ(xs) feat{k} (pred)')
            print('φ feat', k, fit_metrics(ys_t[:,k].cpu().numpy(), X_pred[:,k]))
            
            Z_pred = plotting.predict_on_model(psi, xs2)
            plotting.plot_fit_on_cloud(xs2,ys2_t[:,k].cpu().numpy(), Z_pred[:,k],
                                       title_true=f'SM feat{k} (true)', title_pred=f'ψ(xs2) feat{k} (pred)')
            print('ψ feat', k, fit_metrics(ys2_t[:,k].cpu().numpy(), Z_pred[:,k]))
    
    # ----------------------------
    # 4) Pullback metric fields & geodesics
    # ----------------------------
    if verbose: print("[mgw] computing pullback metrics")
    G_M = geometry.pullback_metric_field(phi, torch.from_numpy(xs).to(device),  eps=geodesic_eps).cpu()
    G_N = geometry.pullback_metric_field(psi, torch.from_numpy(xs2).to(device), eps=geodesic_eps).cpu()

    if verbose: print("[mgw] building kNN graphs")
    Gs = geometry.knn_graph(xs,  k=knn_k)
    Gt = geometry.knn_graph(xs2, k=knn_k)

    if verbose: print("[mgw] geodesic distances")
    D_M = geometry.geodesic_distances_fast(xs,  G_M, Gs)  # (nA,nA)
    D_N = geometry.geodesic_distances_fast(xs2, G_N, Gt)  # (nB,nB)

    D_Mn, D_Nn = _norm_geod(D_M), _norm_geod(D_N)
    C_M, C_N = D_Mn**2, D_Nn**2
    C_M = C_M / (C_M.max() + 1e-12)
    C_N = C_N / (C_N.max() + 1e-12)

    # ----------------------------
    # 5) Solve GW on squared geodesic costs
    # ----------------------------
    if gw_params is None:
        gw_params = dict(verbose=True, inner_maxit=3000, outer_maxit=3000,
                         inner_tol=1e-7,   outer_tol=1e-7,   epsilon=1e-4)
    if verbose:
        print(f"[mgw] solving GW with params={gw_params}")

    P = solve_gw_ott(C_M, C_N, **gw_params)
    if verbose:
        print(f"[mgw] coupling: shape={P.shape}, mass={P.sum():.6f}")

    if save_dir:
        np.save(os.path.join(save_dir, f"P{suffix}.npy"), P)

    return dict(
        P=P,
        xs=xs, xs2=xs2,
        X_pca=X_pca, Z_pca=Z_pca,
        X_rep=X_rep, Z_rep=Z_rep,
        phi=phi, psi=psi,
        G_M=G_M, G_N=G_N,
        C_M=C_M, C_N=C_N,
        config=dict(
            PCA_comp=PCA_comp, CCA_comp=CCA_comp, use_cca_feeler=use_cca_feeler,
            widths=widths, lr=lr, niter=niter, knn_k=knn_k, geodesic_eps=geodesic_eps,
            gw_params=gw_params, device=device, save_dir=save_dir, tag=tag,
            max_obs_A=max_obs_A, max_obs_B=max_obs_B
        )
    )