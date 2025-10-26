
import numpy as np, pandas as pd, scipy.sparse as sp, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.neighbors import KDTree, NearestNeighbors
from mgw import util
from sklearn.utils import check_array
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import anndata as ad

import moscot.plotting as mtp
from moscot import datasets
from moscot.problems.cross_modality import TranslationProblem
import mgw.mgw as mgw

import sys
sys.path.insert(1, '/home/ph3641/Packages/SCOT/src')
import scotv1, scotv2

def _to_csr(P):
    return P if sp.isspmatrix_csr(P) else sp.csr_matrix(P)

def _to_1d(x):
    if sp.issparse(x): return x.toarray().ravel().astype(float)
    x = np.asarray(x); return (x.reshape(x.shape[0], -1)[:,0] if x.ndim>1 else x).astype(float)

def _rownorm(P):
    P = _to_csr(P)
    rs = np.asarray(P.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0/rs) @ P

def _colnorm(P):
    P = _to_csr(P)
    cs = np.asarray(P.sum(axis=0)).ravel()
    cs[cs == 0] = 1.0
    return P @ sp.diags(1.0/cs)

def translate_with_P(P, X_src=None, Y_tgt=None, *, direction="src->tgt", row_normalize=True, col_normalize=False):
    """
    Generic barycentric translation using a coupling P (shape n_src x n_tgt).

    direction:
      - "src->tgt": return X_src mapped into target space using target coordinates Y_tgt (n_tgt x d)
                    result is (n_src x d): for each source i, weighted avg of target Y by P[i,:].
      - "tgt->src": return Y_tgt mapped into source space using source coordinates X_src (n_src x d)
                    result is (n_tgt x d): for each target j, weighted avg of source X by P[:,j].
    """
    P = _to_csr(P)
    if direction == "src->tgt":
        if Y_tgt is None:
            raise ValueError("Need Y_tgt (target coordinates) for src->tgt translation.")
        Pn = _rownorm(P) if row_normalize else P
        return Pn @ check_array(Y_tgt, dtype=float)
    elif direction == "tgt->src":
        if X_src is None:
            raise ValueError("Need X_src (source coordinates) for tgt->src translation.")
        Pn = _colnorm(P) if col_normalize else P  # normalize columns if you want barycentric over source
        return Pn.T @ check_array(X_src, dtype=float)
    else:
        raise ValueError("direction must be 'src->tgt' or 'tgt->src'.")

def run_moscot(
    st, msi,
    Z_s, Z_m,
    alpha=1.0, max_iter=5_000, tol=1e-7
):
    import numpy as np, scipy.sparse as sp
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, normalize
    from sklearn.neighbors import NearestNeighbors
    from moscot.problems import TranslationProblem
    
    # Put into .obsm for MOSCOT
    st_tmp, msi_tmp = st.copy(), msi.copy()
    st_tmp.obsm["GEX_X_pca"] = Z_s
    msi_tmp.obsm["MSI_X_pca"] = Z_m
    
    # --- 1) MOSCOT translate MSI -> RNA latent ---
    tp = TranslationProblem(adata_src=st_tmp, adata_tgt=msi_tmp)
    tp = tp.prepare(src_attr="GEX_X_pca", tgt_attr="MSI_X_pca")
    tp = tp.solve(alpha=alpha)
    sub = tp.problems[("src", "tgt")]
    P = np.array(sub.solution.transport_matrix)
    
    return sp.csr_matrix(P)

def run_scot( Z_s, Z_m, 
             # Using recommended settings: https://rsinghlab.github.io/SCOT/tutorial/
             k= 50, epsilon= 0.005, norm=True
            ):
    
    try:
        domain1 = Z_s
        domain2 = Z_m
        scot_aligner=scotv1.SCOT(domain1, domain2)
        aligned_domain1, aligned_domain2= scot_aligner.align(k=k, e=epsilon, normalize=norm)
        P = scot_aligner.coupling
        P = np.array(P)
        return sp.csr_matrix(P)
    
    except Exception as e:
        print(f"[skip] SCOT: {e}")
        return None

def run_scot_v2(Z_s, Z_m,
            # Using recommended settings: https://rsinghlab.github.io/SCOT/tutorial/
             k= 50, epsilon= 1e-3, norm=True):
    try:
        domain1 = Z_s
        domain2 = Z_m
        scot_aligner=scotv2.SCOTv2([domain1, domain2])
        aligned_domain1, aligned_domain2= scot_aligner.align(normalize=True, 
                                                             k=50, 
                                                             eps=0.005, 
                                                             rho=0.1, 
                                                             projMethod="barycentric")
        P = scot_aligner.couplings[0]
        P = np.array(P)
        return sp.csr_matrix(P)
    except Exception as e:
        print(f"[skip] SCOT: {e}")
        return None

def run_paste_spatial_only(st, msi):
    import numpy as np, pandas as pd, scipy.sparse as sp
    def _spatial_only_copy(src):
        xy = np.asarray(src.obsm["spatial"], dtype=np.float64)
        xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0) + 1e-12)
        ad_tmp = ad.AnnData(
            X=xy,                                  # (n, 2)
            var=pd.DataFrame(index=["sp_x","sp_y"])
        ).copy()
        ad_tmp.obsm["spatial"] = np.asarray(src.obsm["spatial"], dtype=float)  # the original coords
        print(ad_tmp)
        return ad_tmp
    
    st_tmp  = _spatial_only_copy(st)
    msi_tmp = _spatial_only_copy(msi)
    
    try:
        from paste2 import PASTE2
        P = PASTE2.partial_pairwise_align(st_tmp, msi_tmp, s=0.7)
        return sp.csr_matrix(P)
    except Exception as e:
        print(f"paste.pairwise_align failed: {e}")

def run_pot_fgw_spatial(st, msi):
    
    xy_st, xy_msi = st.obsm["spatial"].astype(float), msi.obsm["spatial"].astype(float)
    try:
        import ot
        Cs = ot.utils.dist(xy_st, xy_st);  Cs **= 2
        Cm = ot.utils.dist(xy_msi, xy_msi); Cm **= 2
        P = ot.gromov.gromov_wasserstein(Cs, Cm, loss_fun='square_loss', epsilon=5e-2, max_iter=500, verbose=False)
        return sp.csr_matrix(P)
        
    except Exception as e:
        print(f"[skip] POT-FGW: {e}")
        return None




