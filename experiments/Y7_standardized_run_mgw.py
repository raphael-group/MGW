import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import anndata as ad
import numpy as np
import os, random, numpy as np, torch
import mgw.mgw as mgw
import scanpy as sc
from mgw.baselines import SpatialGW, SCOTGW
from moscot.problems.cross_modality import TranslationProblem
import squidpy as sq
from validation.run_methods import run_scot_v2, run_scot
import scipy.sparse as sp


GLOBAL_SEED = 68
PCA_componet = 30
CCA_componet = 24
MSI_Top = PCA_componet
P_PATH = f'/scratch/gpfs/BRAPHAEL/ST_SM/P_npy/seed{GLOBAL_SEED}/'

ST_PATH= "/scratch/gpfs/BRAPHAEL/ST_SM/adata_ST_Y7_T_raw.h5ad"
MSI_PATH= "/scratch/gpfs/BRAPHAEL/ST_SM/adata_SM_Y7_T_raw.h5ad"

st = ad.read_h5ad(ST_PATH)
msi = ad.read_h5ad(MSI_PATH)
gw_params = dict(verbose=True, inner_maxit=3000, outer_maxit=3000, inner_tol=1e-7,   outer_tol=1e-7,   epsilon=1e-4)

def seed_everything(seed: int = 42):
    """Set global seed for reproducibility across common libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass  # torch not installed or not used

    print(f"✅ Global seed set to {seed}")
seed_everything(GLOBAL_SEED)

K = MSI_Top

sc.pp.normalize_total(msi, target_sum=1e3)

sq.gr.spatial_neighbors(msi, key_added='spatial')
sq.gr.spatial_autocorr(
    msi,
    mode="moran",
    genes=None,
    layer=None,
    n_jobs=8,
)
top_moran = (
    msi.uns["moranI"]
    .sort_values("I", ascending=False)
    .head(K)
)
msi.var["highly_variable_moranI"] = msi.var_names.isin(top_moran.index)

print(msi.uns["moranI"].head())       
moran_df = msi.uns["moranI"]
if "names" in moran_df.columns:         
    ranked = moran_df.sort_values("I", ascending=False)["names"].head(K)
else:
    ranked = moran_df.sort_values("I", ascending=False).index[:K]

msi.var["highly_variable_moranI"] = msi.var_names.isin(ranked)

n_total = msi.shape[1]
n_keep  = int(msi.var["highly_variable_moranI"].sum())
print(f"Before: {msi.shape}  |  marked spatially variable: {n_keep}/{n_total}")

msi = msi[:, msi.var["highly_variable_moranI"]].copy()

print("After: ", msi.shape)

pre = mgw.mgw_preprocess(
    st, msi,
    PCA_comp=PCA_componet,
    CCA_comp=CCA_componet,
    use_cca_feeler=True,
    use_pca_X=True,
    use_pca_Z=False,
    log1p_X=True,
    log1p_Z=False,
    verbose=True,
)


print("run SCOT...")
P_scotv1 = run_scot(pre["X_feat"],pre["Z_feat"])
P_scotv2 = run_scot_v2(pre["X_feat"],pre["Z_feat"])

# sp.save_npz(P_PATH+'P_scotv1_Y7.npz', P_scotv1)
# sp.save_npz(P_PATH+'P_scotv2_Y7.npz', P_scotv2)

print("run ManifoldGW...")
out = mgw.mgw_align_core(
    pre,
    widths=(128,256,256,128),
    lr=1e-3,
    niter=20_000,
    knn_k=12,
    geodesic_eps=1e-2,
    save_dir=f"/scratch/gpfs/BRAPHAEL/ST_SM/experiments/seed{GLOBAL_SEED}",
    tag=f'Y7_{CCA_componet}',
    verbose=True,
    plot_net=False,
    use_cca = True,
    gw_params = gw_params
)

P_cca_manifoldgw = out["P"]

print("run SpatialGW...")
model = SpatialGW(gw_params = gw_params)
P_spatialgw = model.run(xs=pre["xs"],xs2=pre["xs2"])

print("run FeatureGW...")
model = SCOTGW(gw_params = gw_params)
P_featuregw = model.run(Y=pre["X_feat"],Y2=pre["Z_feat"])

print("run MOSCOT...")
adata_st = pre["A_"]
adata_st.obsm["X_cca"] = pre["X_rep"]

adata_sm = pre["B_"]
adata_sm.obsm["counts"] = adata_sm.X.toarray().copy()
adata_sm.obsm["X_cca"] = pre["Z_rep"]
tp = TranslationProblem(adata_src=adata_st, adata_tgt=adata_sm)
tp = tp.prepare(src_attr="X_pca", tgt_attr="counts")

tp = tp.solve(alpha=1.0, epsilon=1e-3,max_iterations=20_0000,threshold=5e-4)

out = tp.solutions
ott_output = out[("src", "tgt")]
P_moscot_pca = ott_output.transport_matrix

# np.save(P_PATH+f'P_cca_manifoldgw_Y7_{CCA_componet}.npy', P_cca_manifoldgw)
# np.save(P_PATH+'P_spatialgw_pca_Y7.npy', P_spatialgw)
# np.save(P_PATH+'P_featuregw_pca_Y7.npy', P_featuregw)
# np.save(P_PATH+'P_moscot_pca_Y7.npy', P_moscot_pca)