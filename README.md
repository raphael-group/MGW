# Manifold Gromov-Wasserstein (MGW): Riemannian Metric Learning for Alignment of Spatial Multiomics

This the repository for **["Riemannian Metric Learning for Alignment of Spatial Multiomics."]()** a technique which performs Riemannian metric learning across spatial modalities (multimoics, transcriptomics, and so on) using the Riemannian pull-back metric, infers Riemannian (geodesic) distances, and aligns Riemannian distances with Gromov-Wasserstein Optimal Transport.

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

### **1. Pre-process the two multiomic datasets **
Call **mgw.mgw_preprocess** on two AnnDatas, such as spatial transcriptomics (**st**) and spatial metabolomics (**msi**).
```python
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

### **2. Run MGW: Fit Neural Fields, learn metric tensors, and align with Gromov-Wasserstein **
```python
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
        use_cca = True, #to improve the alignment, we recommend to set it to TRUE
        gw_params = DEFAULT_GW_PARAMS
    )
```
### **3. Return alignment **
Return the alignment

```python
P = out["P"]
```
