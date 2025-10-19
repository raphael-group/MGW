import anndata as ad
import numpy as np, torch, scipy.sparse as sp
from . import plotting, models, geometry
from . import pullback_metric_field, knn_graph
import scanpy as sc
from . import util
from gw import solve_gw_ott
from metrics import Alignment_Clus_Metrics as acm
from scipy.spatial.distance import cdist


def _normalize_geodesics(D: np.ndarray) -> np.ndarray:
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    q = np.quantile(D[np.triu_indices_from(D, k=1)], 0.99)
    return D / (q + 1e-12)


# ----------------------------
# Class 1: OurModelGW (pull-back metric + GW on squared geodesic costs)
# ----------------------------
class ManifoldGW:
    def __init__(
        self,
        knn_k: int = 12,
        geodesic_eps: float = 1e-6,
        phi_widths=(128, 256, 256, 128),
        psi_widths=(128, 256, 256, 128),
        phi_lr: float = 1e-3,
        psi_lr: float = 1e-3,
        phi_niter: int = 10000,
        psi_niter: int = 10000,
        print_every: int = 500,
        gw_params: dict = None,
        K: int = 50,
        use_pca: bool = False
    ):
        self.knn_k = knn_k
        self.geodesic_eps = geodesic_eps
        self.phi_widths = phi_widths
        self.psi_widths = psi_widths
        self.phi_lr = phi_lr
        self.psi_lr = psi_lr
        self.phi_niter = phi_niter
        self.psi_niter = psi_niter
        self.print_every = print_every
        self.K = K
        self.use_pca = use_pca
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.gw_params = dict(
            verbose=True,
            inner_maxit=2000, outer_maxit=2000,
            inner_tol=1e-8, outer_tol=1e-8,
            epsilon=1e-3,
        )
        if gw_params is not None:
            self.gw_params.update(gw_params)

    def run(self, results: dict) -> dict:
        """
        `results` must contain:
          xs, xt  (normalized spatial coords, arrays of shape (n,2), (m,2))
          If use_pca=False:  X_cca_full, Z_cca_full
          If use_pca=True:   X_pca, Z_pca
          Always provide:    X_pca, Z_pca (used for clustering metrics)
        """
        xs, xs2 = results["xs"], results["xt"]

        if self.use_pca:
            ys, ys2 = results["X_pca"], results["Z_pca"]
        else:
            ys, ys2 = results["X_cca_full"], results["Z_cca_full"]

        # Torch tensors (coords)
        xs_t  = torch.from_numpy(xs).to(self.device)
        xs2_t = torch.from_numpy(xs2).to(self.device)

        # Normalize feature ranges to [0,1]
        ys_t  = util.normalize_range(torch.from_numpy(ys).to(self.device))
        ys2_t = util.normalize_range(torch.from_numpy(ys2).to(self.device))

        dim_e   = 2
        dim_f_M = ys_t.shape[1]
        dim_f_N = ys2_t.shape[1]

        # Learn φ, ψ : (coords)->(features)
        phi = models.PhiModel(dim_e, dim_f_M, widths=self.phi_widths).to(self.device)
        psi = models.PhiModel(dim_e, dim_f_N, widths=self.psi_widths).to(self.device)

        phi = models.train_phi(phi, xs_t, ys_t,
                               lr=self.phi_lr, niter=self.phi_niter,
                               print_every=self.print_every, device=self.device)
        psi = models.train_phi(psi, xs2_t, ys2_t,
                               lr=self.psi_lr, niter=self.psi_niter,
                               print_every=self.print_every, device=self.device)
        phi.eval(); psi.eval()

        # Pull-back metric tensor fields g^M, g^N
        print('Computing metric tensor field')
        G_M = pullback_metric_field(phi, torch.from_numpy(xs).to(self.device), eps=self.geodesic_eps).cpu()  # (n,2,2)
        G_N = pullback_metric_field(psi, torch.from_numpy(xs2).to(self.device), eps=self.geodesic_eps).cpu() # (m,2,2)
        print('Computed pull-back Jacobian fields')

        # kNN graphs
        G_s = knn_graph(xs,  k=self.knn_k)
        G_t = knn_graph(xs2, k=self.knn_k)
        print('Built kNN graphs')

        # Geodesic distances
        D_M = geometry.geodesic_distances_fast(xs,  G_M, G_s)  # (n,n)
        D_N = geometry.geodesic_distances_fast(xs2, G_N, G_t)  # (m,m)
        print('Geodesics computed')

        # Normalize; cost = squared geodesics
        D_Mn = _normalize_geodesics(D_M)
        D_Nn = _normalize_geodesics(D_N)
        C_M  = D_Mn ** 2
        C_N  = D_Nn ** 2

        # Solve GW
        P = solve_gw_ott(C_M, C_N, **self.gw_params)
        print("Coupling:", P.shape, "sum:", P.sum())

        return P


# ----------------------------
# Class 2: SpatialGW (GW on squared Euclidean spatial costs)
# ----------------------------
class SpatialGW:
    """
    GW on squared Euclidean **spatial** costs only.
    You specify which coordinates/features to read from `results`.
    """
    def __init__(
        self,
        xs_key="xs", xs2_key="xt",
        metric_feat_key_M="X_pca", metric_feat_key_N="Z_pca",
        gw_params=None, K=50
    ):
        self.xs_key = xs_key; self.xs2_key = xs2_key
        self.metric_feat_key_M = metric_feat_key_M; self.metric_feat_key_N = metric_feat_key_N
        self.K = K
        self.gw_params = dict(verbose=True, inner_maxit=2000, outer_maxit=2000,
                              inner_tol=1e-8, outer_tol=1e-8, epsilon=1e-3)
        if gw_params: self.gw_params.update(gw_params)

    def run(self, results: dict):
        xs  = results[self.xs_key]
        xs2 = results[self.xs2_key]

        C1 = cdist(xs,  xs)**2
        C2 = cdist(xs2, xs2)**2

        P = solve_gw_ott(C1, C2, **self.gw_params)

        return P

# ----------------------------
# Class 3: SCOTGW (GW on squared feature costs only)
# ----------------------------
class SCOTGW:
    """
    GW on squared **feature** costs only (SCOT-style).
    You specify which feature matrices to read from `results`.
    """
    def __init__(
        self,
        feat_key_M="X_pca", feat_key_N="Z_pca",
        gw_params=None, K=50, normalize=True
    ):
        self.feat_key_M = feat_key_M; self.feat_key_N = feat_key_N
        self.K = K; self.normalize = normalize
        self.gw_params = dict(verbose=True, inner_maxit=2000, outer_maxit=2000,
                              inner_tol=1e-8, outer_tol=1e-8, epsilon=1e-3)
        if gw_params: self.gw_params.update(gw_params)

    def run(self, results: dict):
        Y  = results[self.feat_key_M]
        Y2 = results[self.feat_key_N]

        C1 = cdist(Y,  Y)**2
        C2 = cdist(Y2, Y2)**2
        if self.normalize:
            C1 = C1 / (C1.max() + 1e-12)
            C2 = C2 / (C2.max() + 1e-12)

        P = solve_gw_ott(C1, C2, **self.gw_params)
        return P
