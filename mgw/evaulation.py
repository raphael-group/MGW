import scanpy as sc
import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr, zscore
import scipy.sparse as sp
import scanpy as sc
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import pandas as pd
import sys
sys.path.append("../SpatialMETA")
import spatialmeta as smt
import matplotlib.pyplot as plt
from pathlib import Path
from benchmark.multi_benmark_function import run_modality_benchmark
from .util import _barycentric_right

def _to_numpy(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)
    
def to_dense_row(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)

def bary_proj(adata_st, adata_sm, P_csr, first_tag="ST", second_tag ="SM", eps=1e-12):
    """
    adata_st.X: (n_st, g)
    adata_sm.X: (n_sm, m)
    P_csr: csr_matrix, shape (n_st, n_sm)
    returns joint AnnData on ST grid with X=[genes | projected metabolites]
    """
    assert P_csr.shape == (adata_st.n_obs, adata_sm.n_obs)

    X_st = adata_st.X.toarray()
    X_sm = adata_sm.X.toarray()

    # Barycentric projection: SM -> ST
    # (n_st x n_sm) @ (n_sm x m) => (n_st x m)
    X_sm_to_st = _barycentric_right(P_csr, X_sm, eps=eps)

    X_joint = sp.hstack([X_st, X_sm_to_st], format='csr') if (sp.issparse(X_st) or sp.issparse(X_sm_to_st)) \
              else np.hstack([to_dense_row(X_st), to_dense_row(X_sm_to_st)])

    var_st = adata_st.var.copy()
    var_sm = adata_sm.var.copy()
    var_st["type"] = "ST"
    var_sm["type"] = "SM"

    var_st = var_st.copy()
    var_st.index = ["g:" + str(i) for i in var_st.index]
    var_sm = var_sm.copy()
    var_sm.index = ["m:" + str(i) for i in var_sm.index]
    var_joint = pd.concat([var_st, var_sm])

    # New AnnData on ST grid
    adata_joint = ad.AnnData(
        X=X_joint,
        obs=adata_st.obs.copy(),
        var=var_joint,
        obsm=adata_st.obsm.copy(),
        obsp=adata_st.obsp.copy() if hasattr(adata_st, "obsp") else None,
        uns=adata_st.uns.copy()
    )

    adata_joint.uns["joint_base"] = "ST"
    adata_joint.uns["features"] = {"genes_prefix": "g:", "metabolites_prefix": "m:"}
    adata_joint.obsm[f"coupling_to_{second_tag}"] = P_csr  # shape: (n_ST, n_SM)

    return adata_joint

def evaluate_pipeline_one_dataset(merge_adata=None, adata_path = None, dataset_name = None, method_name=None, dir_path="./log"):
    if (method_name is not None) and (dataset_name is not None):
        out_dir = Path(dir_path) / dataset_name / method_name
    else:
        out_dir = Path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # --- IO & setup ---
    if merge_adata is None:
        merge_adata = sc.read_h5ad(adata_path)

    # --- raw counts snapshot ---
    merge_adata.layers["counts"] = merge_adata.X.copy()

    # --- joint normalization (SM/ST) ---
    smt.pp.normalize_total_joint_adata_sm_st(
        merge_adata,
        target_sum_SM=1e4,
        target_sum_ST=1e4,
    )
    merge_adata.layers["normalized"] = merge_adata.X.copy()

    # --- feature selection (spatially variable) ---
    smt.pp.spatial_variable_joint_adata_sm_st(
        merge_adata,
        n_top_genes=2500,
        n_top_metabolites=1000,
        min_samples=3,
        min_frac=0.9,
        min_logfc=3,
    )
    if "highly_variable_moranI" in merge_adata.var:
        merge_adata = merge_adata[:, merge_adata.var.highly_variable_moranI]

    # --- model ---
    model = smt.model.ConditionalVAESTSM(
        merge_adata,
        n_latent=10,
        device="cuda:0",
        # batch_keys=["sample"],
        # batch_embedding="embedding",
        reconstruction_method_sm="g",
        reconstruction_method_st="zinb",
    )

    loss_dict = model.fit(
        max_epoch=100,
        lr=1e-3,
        kl_loss_reduction="mean",
        mode="multi",
    )

    # --- training curves plot ---
    plt.rcParams["font.family"] = "DejaVu Sans"
    n_keys = len(loss_dict)
    fig, axes = plt.subplots(n_keys, 1, figsize=(8, 3 * max(n_keys, 1)), sharex=True)
    if n_keys == 1:
        axes = [axes]

    for ax, (key, values) in zip(axes, loss_dict.items()):
        epochs = list(range(1, len(values) + 1))
        ax.plot(epochs, values, marker="o", linewidth=1.5)
        ax.set_title(key.replace("_", " "), fontsize=12)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Epoch", fontsize=11)
    fig.suptitle("Training Metrics over Epochs", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "training_metrics.png", dpi=150)
    plt.close(fig)

    # --- model outputs ---
    Z = model.get_latent_embedding()              # (n_cells, n_latent)
    X = model.get_normalized_expression()         # reconstruction or norm expr (per your API)
    C = model.get_modality_contribution()         # per-cell contributions (SM/ST)

    merge_adata.layers['reconstruction'] = X
    merge_adata.obsm['X_emb']=Z
    merge_adata.obsm['contribution_st']=C
    merge_adata.obsm['contribution_sm']=1-C

    merge_adata.obsm["multi_emb"] = merge_adata.obsm["X_emb"]
    assert "multi_emb" in merge_adata.obsm       
    assert "spatial" in merge_adata.obsm            
    assert "type" in merge_adata.var.columns       
    assert "counts" in merge_adata.layers          

    emb_key = "multi_emb" if "multi_emb" in merge_adata.obsm else "X_emb"
    assert emb_key in merge_adata.obsm, "No embedding found in .obsm['multi_emb'] or .obsm['X_emb']"

    # --- neighbors, Leiden, UMAP ---
    sc.pp.neighbors(merge_adata, use_rep=emb_key, n_neighbors=20, metric="euclidean")
    sc.tl.umap(merge_adata)  # required before plotting UMAP
    sc.tl.leiden(merge_adata, resolution=1.0, key_added="leiden_cluster")

    # UMAP plot colored by the actual cluster column
    fig, ax = plt.subplots(figsize=(5, 5))
    sc.pl.umap(
        merge_adata,
        color=["leiden_cluster"],
        palette=sc.pl.palettes.default_28,
        show=False,
        size=10,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "umap_leiden.png", dpi=150)
    plt.close(fig)
    
    df = pd.DataFrame({
        "barcode": merge_adata.obs_names.astype(str),
        "leiden_cluster": merge_adata.obs["leiden_cluster"].astype(str),
    })
    df.to_parquet(out_dir / "leiden_clusters_1.parquet", index=False)

    merge_adata.write(out_dir / "merged_after_VAE.h5ad")

    results = run_modality_benchmark(merge_adata, out_dir)
    import json
    results_path = out_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")
    return results

def coupling_pearson_metrics(adata1, adata2, P):
    a1, a2 = adata1.copy(), adata2.copy()
    a1.var_names_make_unique()
    a2.var_names_make_unique()

    sc.pp.log1p(a1); sc.pp.log1p(a2)
    sc.pp.scale(a1);  sc.pp.scale(a2)

    common = a1.var_names.intersection(a2.var_names)
    if len(common) == 0:
        raise ValueError("No shared genes.")
    a1 = a1[:, common].copy()
    a2 = a2[:, common].copy()

    X1 = a1.X.toarray() if sp.issparse(a1.X) else np.asarray(a1.X)
    X2 = a2.X.toarray() if sp.issparse(a2.X) else np.asarray(a2.X)
    P = P.toarray() if sp.issparse(P) else np.asarray(P)

    if P.shape != (X2.shape[0], X1.shape[0]):
        raise ValueError(f"P shape {P.shape} mismatches {(X2.shape[0], X1.shape[0])}.")

    X1 = zscore(X1, axis=1)
    X2 = zscore(X2, axis=1)

    corr = (X1 @ X2.T) / X1.shape[1]

    mask = P > 0
    unweighted = float(corr[mask].mean())
    weighted   = float((corr[mask] * P[mask]).sum() / P[mask].sum())

    return unweighted, weighted


def coupling_similarity_overall(
    adata1, adata2, P,
    log1p=True, scale=True,
    eps=1e-12
):
    """
    Compute overall cosine similarity and Pearson correlation between
    adata1.X and the barycentrically projected adata2.X via coupling P.

    Returns:
        dict with keys 'cosine' and 'pearson'
    """
    a1, a2 = adata1.copy(), adata2.copy()
    a1.var_names_make_unique()
    a2.var_names_make_unique()

    if log1p:
        sc.pp.log1p(a1); sc.pp.log1p(a2)
    if scale:
        sc.pp.scale(a1); sc.pp.scale(a2)

    common = a1.var_names.intersection(a2.var_names)
    if len(common) == 0:
        raise ValueError("No shared genes between datasets.")
    common = common.sort_values()
    a1 = a1[:, common].copy()
    a2 = a2[:, common].copy()

    X1 = _to_numpy(a1.X)
    X2 = _to_numpy(a2.X)
    P = P.toarray() if sp.issparse(P) else np.asarray(P)

    if P.shape != (X2.shape[0], X1.shape[0]):
        P = P.T

    X1_proj = _barycentric_right(P, X1, eps=eps)

    x1_flat = X1_proj.flatten()
    x2_flat = X2.flatten()

    cos_sim = 1 - cosine(x1_flat, x2_flat)
    pear_corr = pearsonr(x1_flat, x2_flat)[0]

    return {"cosine": float(cos_sim), "pearson": float(pear_corr)}
