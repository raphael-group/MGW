# Manifold Gromov-Wasserstein (MGW): Riemannian Metric Learning for Alignment of Spatial Multiomics

This the repository for **["Riemannian Metric Learning for Alignment of Spatial Multiomics."]()** a technique which:
 1. Performs Riemannian metric learning across spatial modalities (multimoics, transcriptomics, and so on) using the Riemannian pull-back metric.
 2. Infers Riemannian (geodesic) distances.
 3. Aligns Riemannian distances with Gromov-Wasserstein Optimal Transport.

In the section below, we detail the usage of MGW which complements the simple example notebook:
```markdown
- [demo_mgw_y7.ipynb](demo_mgw_y7.ipynb)
```

## Contents
`mgw/`
  - `mgw.py` — main solver/class for MGW
  - `geometry.py` — metric-tensor, geodesic distance, k-NN graph, APSP utilities
  - `models.py` — neural field models (MLP)
  - `metric.py` — evaluation metrics (e.g. migration, AMI, cosine similarity(
  - `plotting.py` — visualization utilities
  - `utils.py` — miscellaneous helpers, barycentric projection

`validation/`
  - `dopamine.py` — validation utilities for dopamine experiments (AUROC, AUPRC)
  - `run_methods.py` — code for running other methods (moscot Translation, SCOT, SCOTv2, PASTE2 FGW spatial, POT FGW spatial only)

`demos/`
  - `demo_mgw_y7.ipynb` — demo notebook for running MGW on the Y_7 ccRCC slice (Hu '24)
  - `riemannian_mouse_geodesics.ipynb` — code for visualization of the geodesics in the Riemannian pull-back metric of E9.5-10.5 mouse embryo (Chen '22)

- Reproducible experimental notebooks on Stereo-Seq Mouse Embryo, Visium-Xenium alignment of colorectal cancer, MALDI-MSI metabolomics and Visium transcriptomics alignment of human striatum, AFADESI-MSI and Visium alignment of renal cancer

## **Getting Started**

### **1. Load the two multiomic datasets **
Load two AnnData objects such as spatial transcriptomics (`st`) and spatial metabolomics (`msi`) after appropriate filtering.
```python
import anndata as ad
st = ad.read_h5ad(ST_PATH)
msi = ad.read_h5ad(MSI_PATH)
```
### **1. Running MGW's pre-processing (optional) the two multiomic datasets **
Call `mgw.mgw_preprocess` on two AnnDatas. 

You can run PCA (will default to pre-computed PCA if already done) with `PCA_comp` components, and an additional `CCA` step for multimodal data. Set `use_cca_feeler=True` for this CCA step, which involves basic/coarse feeler alignment (`spatial_only: bool = True` to do a spatial-only feeler, `feature_only = True` to do a feature-only feeler, or if both `False` a basic spatial-feature feeler). This subsets feature dimensions which are correlated across modalities, and you can specify the number of final CCA dimensions with `CCA_comp`.

To keep `st.X` and `msi.X` as-is without processing, set `use_cca_feeler=False`, `use_pca_X/Z=False`, and `log1p_X/Z=False`.

```python
import mgw.mgw as mgw
pre = mgw.mgw_preprocess(
    st, msi,
    PCA_comp=PCA_componet,
    CCA_comp=CCA_componet,
    use_cca_feeler=True, 
    use_pca_X=True,
    use_pca_Z=False, #False if the features from second modality are intensities which doesn't make sense to run pca on
    log1p_X=True,
    log1p_Z=False, #False if the features from second modality are not counts which doesn't make sense to run log1p on
    verbose=True
)
```

### **2. Run MGW **

Next, we run `mgw.mgw_align_core` on the data **pre** to both infer the neural fields, learn metric tensors, and align the result with Gromov-Wasserstein.
```python

PHI_ARC = (128,256,256,128)
KNN_K= 12
DEFAULT_GW_PARAMS = dict(verbose=True, inner_maxit=3000, outer_maxit=3000, inner_tol=1e-7,   outer_tol=1e-7,   epsilon=1e-4)
DEFAULT_LR = 1e-3
DEFAULT_EPS = 1e-2
DEFAULT_ITER = 20_000
EXP_PATH = "your_path"
EXP_TAG = "your_tag"

out = mgw.mgw_align_core(
        pre,
        widths=PHI_ARC,
        lr=DEFAULT_LR,
        niter=DEFAULT_ITER,
        knn_k=KNN_K,
        geodesic_eps=DEFAULT_EPS,
        save_dir=EXP_PATH, 
        tag=EXP_TAG, 
        verbose=True,
        plot_net=True, # zoom in to visually check if the two modalities shown similar pattern
        use_cca = True, #for multi-modal, we recommend setting to TRUE
        gw_params = DEFAULT_GW_PARAMS
    )
```
Here, the key parameters are
- `PHI_ARC`: Layers of the MLP
- `KNN_K`: Resolution of the K nearest neighbor graph used for Riemannian geodesics
- `DEFAULT_GW_PARAMS`: Default parameters for the optimal transport solver of **ott jax**
- `DEFAULT_LR`: Learning-rate for the network.
- `DEFAULT_ITER`: Number of training iterations for the network.
- `save_dir`: Where to save outputs
- `tag`: Tag for generated files.

### **3. Return alignment and project across modalities **
We have a number of variables which can be accessed from `out`.
- `P`: MGW coupling/alignment
- `xs`: Spatial coordinates 1 (normalized)
- `xs2`: Spatial coordinates 2 (normalized)
- `phi`: Neural field mapping into modality 1
- `psi`: Neural field mapping into modality 2
- `G_M`/`G_N`: Pull-back metric tensor field over coordinates
- `C_M`/`C_N`: MGW Riemannian distance matrices

As an example, let us return the alignment and barycentrically project across modalities.

```python
P = out["P"]
from mgw.evaulation import bary_proj
adata_sm2st = bary_proj(st, msi, P)
adata_st2sm = bary_proj(msi, st, P.T)
```

`P` represents the **MGW** alignment, `adata_sm2st` is the metabolomics to transcriptomics projection (added to `st` as metabolite annotation), and `adata_st2sm` is the transcriptomics to metabolomics projection (added to `msi` as metabolite annotation).

