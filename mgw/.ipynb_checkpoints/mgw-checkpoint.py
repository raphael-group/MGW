# mgw.py
from typing import Optional, Dict, Any, Tuple
import numpy as np, torch, os, scipy.sparse as sp
from mgw import util, models, geometry, plotting
from mgw.gw import solve_gw_ott
from sklearn.preprocessing import normalize

def mgw_preprocess(
    A, B,
    *,
    # feature config
    PCA_comp: int = 30,
    CCA_comp: int = 3,
    use_cca_feeler: bool = True,
    feeler_downsample: Optional[int] = 8000,
    log1p_X: bool = True,
    log1p_Z: bool = True,
    use_pca_X: bool = True,
    use_pca_Z: bool = True,
    # sampling
    max_obs_A: Optional[int] = None,
    max_obs_B: Optional[int] = None,
    # feeler flavors
    spatial_only: bool = True,
    feature_only: bool = False,
    # misc
    verbose: bool = True,
) -> Dict[str, Any]:
    """Preprocess A,B into unit-square coords + representation features."""
    A_ = _maybe_subsample(A, max_obs_A)
    B_ = _maybe_subsample(B, max_obs_B)

    xs  = _to_unit_square(A_.obsm["spatial"])
    xs2 = _to_unit_square(B_.obsm["spatial"])

    def _get_pca(adata, n_comps, log1p):
        if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= n_comps:
            return np.asarray(adata.obsm["X_pca"][:, :n_comps], dtype=np.float64)
        return util.pca_from(adata, n_comps=n_comps, log1p=log1p)

    # Base PCA or raw features
    if use_pca_X:
        X_pca = _get_pca(A_, n_comps=PCA_comp, log1p=log1p_X)
    else:
        X_pca = A_.X.toarray() if sp.issparse(A_.X) else np.asarray(A_.X)
        X_pca = X_pca.astype(np.float64)
        if log1p_X: X_pca = np.log1p(X_pca)

    if use_pca_Z:
        Z_pca = _get_pca(B_, n_comps=PCA_comp, log1p=log1p_Z)
    else:
        Z_pca = B_.X.toarray() if sp.issparse(B_.X) else np.asarray(B_.X)
        Z_pca = Z_pca.astype(np.float64)
        if log1p_Z: Z_pca = np.log1p(Z_pca)

    if verbose:
        print(f"[mgw.pre] PCA/raw shapes: X={X_pca.shape}  Z={Z_pca.shape}")

    # Optional CCA feeler
    feeler = None
    if use_cca_feeler:
        if verbose:
            print("[mgw.pre] CCA feeler: PCA → small GW (fused) → barycentric → CCA")
        feeler = util.project_informative_features(
            path_X=None, path_Z=None,
            adata_X=A_, adata_Z=B_,
            PCA_comp=PCA_comp, CCA_comp=CCA_comp,
            n_downsample=feeler_downsample,
            log1p_X=log1p_X, log1p_Z=log1p_Z,
            verbose=verbose,
            spatial_only=spatial_only,
            feature_only=feature_only,
        )
        X_rep = feeler["X_cca_full"]
        Z_rep = feeler["Z_cca_full"]
    else:
        X_rep, Z_rep = X_pca, Z_pca

    # L2 row-norm for model targets
    X_pca = normalize(X_pca)
    Z_pca = normalize(Z_pca)
    
    # L2 row-norm for model targets
    X_rep = normalize(X_rep)
    Z_rep = normalize(Z_rep)
    
    return dict(
        A_=A_, B_=B_,
        xs=xs, xs2=xs2,
        X_feat=X_pca, Z_feat=Z_pca,
        X_rep=X_rep, Z_rep=Z_rep,
        feeler=feeler,
        config=dict(
            PCA_comp=PCA_comp, CCA_comp=CCA_comp, use_cca_feeler=use_cca_feeler,
            feeler_downsample=feeler_downsample, log1p_X=log1p_X, log1p_Z=log1p_Z,
            use_pca_X=use_pca_X, use_pca_Z=use_pca_Z,
            spatial_only=spatial_only, feature_only=feature_only,
            max_obs_A=max_obs_A, max_obs_B=max_obs_B,
        ),
    )

def mgw_align_core(
    pre: Dict[str, Any],
    *,
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
    device: Optional[str] = None,
    torch_default_dtype: torch.dtype = torch.float64,
    # caching
    save_dir: Optional[str] = None,
    tag: Optional[str] = None,
    # misc
    verbose: bool = True,
    plot_net: bool = False,
) -> Dict[str, Any]:
    """Core: learn φ,ψ; pullback metrics; GW; return coupling + intermediates."""
    xs, xs2 = pre["xs"], pre["xs2"]
    X_rep, Z_rep = pre["X_rep"], pre["Z_rep"]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_dtype(torch_default_dtype)
    if verbose: print(f"[mgw.core] device={device} dtype={torch.get_default_dtype()}")

    suffix = "" if not save_dir else ("" if tag is None else f"_{tag}")
    if save_dir: os.makedirs(save_dir, exist_ok=True)

    xs_t, xs2_t = torch.from_numpy(xs).to(device), torch.from_numpy(xs2).to(device)
    ys_t, ys2_t = torch.from_numpy(X_rep).to(device), torch.from_numpy(Z_rep).to(device)
    dim_e, dim_f_M, dim_f_N = 2, ys_t.shape[1], ys2_t.shape[1]
    if verbose: print(f"[mgw.core] dims: E=2, F_M={dim_f_M}, F_N={dim_f_N}")

    # φ, ψ
    phi = models.PhiModel(2, dim_f_M, widths=widths).to(device)
    psi = models.PhiModel(2, dim_f_N, widths=widths).to(device)

    phi_path = psi_path = None
    if save_dir:
        phi_path = os.path.join(save_dir, f"phi{suffix}.pt")
        psi_path = os.path.join(save_dir, f"psi{suffix}.pt")

    if phi_path and os.path.isfile(phi_path) and os.path.isfile(psi_path):
        if verbose: print("[mgw.core] loading cached φ, ψ")
        phi.load_state_dict(torch.load(phi_path, map_location=device))
        psi.load_state_dict(torch.load(psi_path, map_location=device))
        phi.eval(); psi.eval()
    else:
        if verbose: print("[mgw.core] training φ, ψ")
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
            Z_pred = plotting.predict_on_model(psi, xs2)
            plotting.plot_fit_on_cloud(xs2, ys2_t[:,k].cpu().numpy(), Z_pred[:,k],
                                       title_true=f'SM feat{k} (true)', title_pred=f'ψ(xs2) feat{k} (pred)')

    if verbose: print("[mgw.core] pullback metrics & geodesics")
    G_M = geometry.pullback_metric_field(phi, torch.from_numpy(xs).to(device),  eps=geodesic_eps).cpu()
    G_N = geometry.pullback_metric_field(psi, torch.from_numpy(xs2).to(device), eps=geodesic_eps).cpu()

    Gs = geometry.knn_graph(xs,  k=knn_k)
    Gt = geometry.knn_graph(xs2, k=knn_k)
    D_M = geometry.geodesic_distances_fast(xs,  G_M, Gs)
    D_N = geometry.geodesic_distances_fast(xs2, G_N, Gt)

    def _norm_geod(D):
        D = np.maximum(D, 0.0); np.fill_diagonal(D, 0.0)
        q = np.quantile(D[np.triu_indices_from(D, 1)], 0.99)
        return D / (q + 1e-12)

    C_M = _norm_geod(D_M)**2; C_M /= (C_M.max() + 1e-12)
    C_N = _norm_geod(D_N)**2; C_N /= (C_N.max() + 1e-12)

    if gw_params is None:
        gw_params = dict(verbose=True, inner_maxit=3000, outer_maxit=3000,
                         inner_tol=1e-7,   outer_tol=1e-7,   epsilon=1e-4)
    if verbose: print(f"[mgw.core] solving GW with {gw_params}")
    P = solve_gw_ott(C_M, C_N, **gw_params)
    if verbose: print(f"[mgw.core] coupling: shape={P.shape}, mass={P.sum():.6f}")

    if save_dir:
        np.save(os.path.join(save_dir, f"P{suffix}.npy"), P)

    return dict(
        P=P, xs=xs, xs2=xs2,
        X_rep=X_rep, Z_rep=Z_rep,
        X_feat=pre.get("X_feat"), Z_feat=pre.get("Z_feat"),
        phi=phi, psi=psi,
        G_M=G_M, G_N=G_N,
        C_M=C_M, C_N=C_N,
        feeler=pre.get("feeler"),
        config=dict(
            **pre["config"],
            widths=widths, lr=lr, niter=niter,
            knn_k=knn_k, geodesic_eps=geodesic_eps,
            gw_params=gw_params, device=device,
            save_dir=save_dir, tag=tag,
        )
    )

def mgw_align(
    A, B,
    *,
    # preprocess knobs
    PCA_comp: int = 30, CCA_comp: int = 3, use_cca_feeler: bool = True,
    feeler_downsample: Optional[int] = 8000,
    log1p_X: bool = True, log1p_Z: bool = True,
    use_pca_X: bool = True, use_pca_Z: bool = True,
    max_obs_A: Optional[int] = None, max_obs_B: Optional[int] = None,
    spatial_only: bool = True, feature_only: bool = False,
    # core knobs
    widths: tuple = (128, 256, 256, 128), lr: float = 1e-3,
    niter: int = 20_000, print_every: int = 1_000,
    knn_k: int = 12, geodesic_eps: float = 1e-2,
    gw_params: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None, torch_default_dtype: torch.dtype = torch.float64,
    save_dir: Optional[str] = None, tag: Optional[str] = None,
    verbose: bool = True, plot_net: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper: preprocess → core."""
    pre = mgw_preprocess(
        A, B,
        PCA_comp=PCA_comp, CCA_comp=CCA_comp, use_cca_feeler=use_cca_feeler,
        feeler_downsample=feeler_downsample,
        log1p_X=log1p_X, log1p_Z=log1p_Z,
        use_pca_X=use_pca_X, use_pca_Z=use_pca_Z,
        max_obs_A=max_obs_A, max_obs_B=max_obs_B,
        spatial_only=spatial_only, feature_only=feature_only,
        verbose=verbose,
    )
    return mgw_align_core(
        pre,
        widths=widths, lr=lr, niter=niter, print_every=print_every,
        knn_k=knn_k, geodesic_eps=geodesic_eps,
        gw_params=gw_params,
        device=device, torch_default_dtype=torch_default_dtype,
        save_dir=save_dir, tag=tag,
        verbose=verbose, plot_net=plot_net,
    )

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

