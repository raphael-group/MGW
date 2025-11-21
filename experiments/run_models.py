from mgw.baselines import ManifoldGW, SpatialGW, SCOTGW
from mgw import util
import json
import pandas as pd

default_path = "/scratch/gpfs/ph3641/mouse_embryo/"
pairs = [
    {"pair_id": "E9.5_vs_E10.5", "adata1_path": default_path+"E9.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E10.5_E1S1.MOSTA.h5ad"},
    #{"pair_id": "E10.5_vs_E11.5", "adata1_path": default_path+"E10.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E11.5_E1S1.MOSTA.h5ad"},
    #{"pair_id": "E11.5_vs_E12.5", "adata1_path": default_path+"E11.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E12.5_E1S1.MOSTA.h5ad"},
    #{"pair_id": "E12.5_vs_E13.5", "adata1_path": default_path+"E12.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E13.5_E1S1.MOSTA.h5ad"},
    #{"pair_id": "E13.5_vs_E14.5", "adata1_path": default_path+"E13.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E14.5_E1S1.MOSTA.h5ad"},
    #{"pair_id": "E14.5_vs_E15.5", "adata1_path": default_path+"E14.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E15.5_E1S1.MOSTA.h5ad"},
    #{"pair_id": "E15.5_vs_E16.5", "adata1_path": default_path+"E15.5_E1S1.MOSTA.h5ad", "adata2_path": default_path+"E16.5_E1S1.MOSTA.h5ad"},
]

records = {} 

for p in pairs:
    pair_id = p["pair_id"]
    

    # Load (or reuse in-memory adatas if you have them)
    #adata1 = sc.read_h5ad(p["adata1_path"])
    #adata2 = sc.read_h5ad(p["adata2_path"])

    records[pair_id] = {}
    results = util.project_informative_features(p["adata1_path"], p["adata2_path"], PCA_comp=30, CCA_comp=3, spatial_only=True, feature_only=False)

    manifoldGW_with_cca = ManifoldGW(use_pca=False)  # use CCA
    P_manifold_cca = manifoldGW_with_cca.run(results)
    # can do some metrics here on coupling matrix and add to dict
    print("ManifoldGW CCA:", P_manifold_cca .shape())
    records[pair_id]["ManifoldGW_CCA"] = {"P shape": P_manifold_cca.shape()}

    manifoldGW_with_pca = ManifoldGW(use_pca=True)  # !!!use PCA
    P_manifold_pca = manifoldGW_with_pca.run(results)
    # can do some metrics here on coupling matrix and add to dict
    print("ManifoldGW PCA:", P_manifold_pca .shape())
    records[pair_id]["ManifoldGW_PCA"] = {"P shape": P_manifold_pca.shape()}

    # Spatial GW baseline
    spgw = SpatialGW()
    P_spatial = spgw.run(results)
    # can do some metrics here on coupling matrix and add to dict
    print("SpatialGW:", P_spatial.shape())
    records[pair_id]["SpatialGW"] = {"P shape": P_spatial.shape()}


    # SCOT (feature-only GW) baseline
    scot = SCOTGW()
    P_features = scot.run(results)
    # can do some metrics here on coupling matrix and add to dict
    print("SCOTGW:", P_features.shape())
    records[pair_id]["SCOTGW"] = {"P shape": P_features.shape()}


def _to_native(o):
    try:
        # cast numpy scalars to float/int
        return float(o)
    except Exception:
        return o
    
with open("metrics_raw_test.json", "w") as f:
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
df_long.to_csv("metrics_long_test.csv", index=False)