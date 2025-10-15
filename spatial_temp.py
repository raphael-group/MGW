import anndata as ad
import numpy as np, torch, scipy.sparse as sp
from mgw import plotting, models, geometry
from mgw import pullback_metric_field, knn_graph
import importlib; importlib.reload(geometry)
import scanpy as sc
from mgw import util
from mgw.gw import solve_gw_ott
from mgw.metrics import Alignment_Clus_Metrics as acm
from scipy.spatial.distance import cdist
import json
import pandas as pd

#xs, xs_t normalized coordinates of adata1
#xt, xs2, xs2_t normalized coordinates of adata2
#ys,ys_t X_pca or X_cca_full
#yt,ys2_t Z_pca or Z_cca_full


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
print("Device:", device)


ST_LAYER = None
SM_LAYER = None

# Preprocessing
log1p_features = True
zscore_per_feature = True
n_features_st = 256   # set None to keep all
n_features_sm = 256   # set None to keep all

# Graph / geodesic params
knn_k = 12
geodesic_eps = 1e-6   # used inside pullback metric calc

gw_params = dict(verbose=True, inner_maxit=2000, outer_maxit=2000,inner_tol=1e-8, outer_tol=1e-8, epsilon=1e-3)
        
def run_our_model(results, pca=False, K=50):
    xs, xs2 = results["xs"], results["xt"]
    if pca:
        ys, ys2 = results["X_pca"], results["Z_pca"]
    else:
        ys, ys2 = results["X_cca_full"], results["Z_cca_full"]

    xs_t  = torch.from_numpy(xs).to(device)
    xs2_t = torch.from_numpy(xs2).to(device)

    ys_t  = util.normalize_range(torch.from_numpy(ys).to(device))
    ys2_t = util.normalize_range(torch.from_numpy(ys2).to(device))

    dim_e   = 2
    dim_f_M = ys_t.shape[1]
    dim_f_N = ys2_t.shape[1]

    # ----------------------------
    # Learn φ, ψ : (coords)->(features)
    # ----------------------------
    phi = models.PhiModel(dim_e, dim_f_M, widths=(128,256,256,128)).to(device)
    psi = models.PhiModel(dim_e, dim_f_N, widths=(128,256,256,128)).to(device)

    phi = models.train_phi(phi, xs_t, ys_t, lr=1e-3, niter=10000, print_every=500, device=device)
    psi = models.train_phi(psi, xs2_t, ys2_t, lr=1e-3, niter=10000, print_every=500, device=device)
    phi.eval(); psi.eval()
    # ---------------------------------------
    # Pull-back metric tensor fields g^M, g^N
    # ---------------------------------------
    print('Computing metric tensor field')
    G_M = pullback_metric_field(phi, torch.from_numpy(xs).to(device), eps=geodesic_eps).cpu()   # (n,2,2)
    G_N = pullback_metric_field(psi, torch.from_numpy(xs2).to(device), eps=geodesic_eps).cpu()   # (m,2,2)
    print('Computed pull-back Jacobian fields')

    # ----------------------------
    # kNN graphs + geodesics
    # ----------------------------
    G_s = knn_graph(xs,  k=knn_k)
    G_t = knn_graph(xs2,  k=knn_k)
    print('Built kNN graphs')

    D_M = geometry.geodesic_distances_fast(xs,  G_M, G_s)  # (n,n)
    D_N = geometry.geodesic_distances_fast(xs2,  G_N, G_t)  # (m,m)
    print('Geodesics computed')

    def normalize_geodesics(D):
        D = np.maximum(D, 0.0)
        np.fill_diagonal(D, 0.0)
        q = np.quantile(D[np.triu_indices_from(D, k=1)], 0.99)
        return D / (q + 1e-12)

    D_Mn = normalize_geodesics(D_M)
    D_Nn = normalize_geodesics(D_N)
    C_M  = D_Mn**2
    C_N  = D_Nn**2

    # ----------------------------
    # Solve GW on squared geodesic costs
    # ----------------------------

    #gw_params = dict(verbose=True, inner_maxit=2000, outer_maxit=2000,inner_tol=1e-8, outer_tol=1e-8, epsilon=1e-3)

    P = solve_gw_ott(C_M, C_N, **gw_params)
    print("Coupling:", P.shape, "sum:", P.sum())

    # ----------------------------
    # Visualize alignment (Procrustes on coordinates)
    # ----------------------------

    #s_aligned, t_aligned, R, tvec = plotting.procrustes_from_coupling(xs, xs2, P)
    #plotting.plot_alignment_lines_dense(s_aligned, t_aligned, P, alpha=0.05)

    Y, Y2 = results["X_pca"], results["Z_pca"]

    ami, ari = acm(Y, Y2, P, k=K)

    return {"AMI": ami, "ARI": ari}

def run_spatial_GW(xs, xs2 ,Y ,Y2 ,K=50):

    # Spatial GW baseline
    C1_s  = cdist(xs, xs)**2
    C2_s  = cdist(xs2, xs2)**2

    P_s = solve_gw_ott(C1_s, C2_s, **gw_params)
    #s_aligned, t_aligned, R, tvec = plotting.procrustes_from_coupling(xs, xs2, P_s)
    #plotting.plot_alignment_lines_dense(s_aligned, t_aligned, P_s, alpha=0.05)
    ami, ari = acm(Y, Y2, P_s, k=K)
    return {"AMI": ami, "ARI": ari}
    

def run_SCOT(Y ,Y2, K=50):
    # SCOT Alignment baseline

    C1_f  = cdist(Y, Y)**2
    C2_f  = cdist(Y2, Y2)**2

    C1_f = C1_f / C1_f.max()
    C2_f = C2_f / C2_f.max()

    P_f = solve_gw_ott(C1_f, C2_f, **gw_params)
    #s_aligned, t_aligned, R, tvec = plotting.procrustes_from_coupling(xs, xs2, P_f)
    #plotting.plot_alignment_lines_dense(s_aligned, t_aligned, P_f, alpha=0.05)
    ami, ari = acm(Y, Y2, P_f, k=K)

    return {"AMI": ami, "ARI": ari}

default_path = "/scratch/gpfs/ph3641/mouse_embryo/"
pairs = [
    {"pair_id": "E9.5_vs_E10.5", "adata1_path": default_path+"E9.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E10.5_E1S1.MOSTA.h5ad"},
    {"pair_id": "E10.5_vs_E11.5", "adata1_path": default_path+"E10.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E11.5_E1S1.MOSTA.h5ad"},
    {"pair_id": "E11.5_vs_E12.5", "adata1_path": default_path+"E11.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E12.5_E1S1.MOSTA.h5ad"},
    {"pair_id": "E12.5_vs_E13.5", "adata1_path": default_path+"E12.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E13.5_E1S1.MOSTA.h5ad"},
    {"pair_id": "E13.5_vs_E14.5", "adata1_path": default_path+"E13.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E14.5_E1S1.MOSTA.h5ad"},
    {"pair_id": "E14.5_vs_E15.5", "adata1_path": default_path+"E14.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E15.5_E1S1.MOSTA.h5ad"},
    {"pair_id": "E15.5_vs_E16.5", "adata1_path": default_path+"E15.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E16.5_E1S1.MOSTA.h5ad"},
]

records = {}  # collect rows here

for p in pairs:
    pair_id = p["pair_id"]
    

    # Load (or reuse in-memory adatas if you have them)
    #adata1 = sc.read_h5ad(p["adata1_path"])
    #adata2 = sc.read_h5ad(p["adata2_path"])

    records[pair_id] = {}
    results = util.project_informative_features(p["adata1_path"], p["adata2_path"], PCA_comp=30, CCA_comp=3, spatial_only=True, feature_only=False)

    model1_record = run_our_model(results, pca=False)
    records[pair_id]["ManifoldGW_CCA"] = model1_record

    model2_record = run_our_model(results, pca=True)
    records[pair_id]["ManifoldGW_PCA"] = model2_record

    model3_record = run_spatial_GW(xs = results["xs"], xs2 = results["xt"], Y = results["X_pca"], Y2=results["Z_pca"])
    records[pair_id]["SpatialGW_OT"] = model3_record

    model4_record = run_SCOT(Y = results["X_pca"], Y2=results["Z_pca"])
    records[pair_id]["Features_GW_OT"] = model4_record

def _to_native(o):
    try:
        # cast numpy scalars to float/int
        return float(o)
    except Exception:
        return o
    
with open("metrics_raw.json", "w") as f:
    json.dump(records, f, indent=2, default=_to_native)

rows = []
for pair_id, models in records.items():
    for model_name, metrics in models.items():
        for metric_name, value in metrics.items():
            rows.append({
                "pair_id": pair_id,
                "model": model_name,
                "metric": metric_name,
                "value": float(value),
            })

df_long = pd.DataFrame(rows).sort_values(["pair_id", "model", "metric"])
df_long.to_csv("metrics_long.csv", index=False)