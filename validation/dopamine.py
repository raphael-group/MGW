import numpy as np, pandas as pd, scipy.sparse as sp, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from mgw import util

# ---------- helpers ----------
def to_1d(x):
    if sp.issparse(x): return x.toarray().ravel().astype(float)
    x = np.asarray(x);  return (x.reshape(x.shape[0], -1)[:,0] if x.ndim>1 else x).astype(float)

def rownorm(M):
    M = M.tocsr() if not sp.isspmatrix_csr(M) else M
    rs = np.array(M.sum(axis=1)).ravel()
    return sp.diags(1.0/np.maximum(rs,1e-12)) @ M

def orient_P(P, n_st, n_msi):
    return P if P.shape==(n_st,n_msi) else (P.T if P.shape==(n_msi,n_st) else (_ for _ in ()).throw(ValueError(f"Bad P shape {P.shape}")))

def eval_proj(P_st_msi, mz_target, ppm=25, pos=["dopamine_Cd"]):
    '''
    AUROC: how does the rank order of the target metabolite's projection under P separate dopaminergic GT label or not.
    (AUPRC: more sensitive if positives are rare)
    '''
    # "CI"
    # Evaluate mz values
    mz = msi.var["mz"].values.astype(float)
    # Identify diff from target, pick optimal index for that target
    j  = int(np.argmin(np.abs(mz - mz_target)))
    if 1e6*abs(mz[j]-mz_target)/mz_target > ppm: print(f"[warn] {mz_target} -> {mz[j]:.5f} (> {ppm} ppm)")
    # Intensity of target m/z channel
    inten = to_1d(msi.X[:, j])
    # normalize
    inten /= np.percentile(inten,99)+1e-9
    # Barycentric projection of m/z onto ST spots via P
    s = to_1d(rownorm(P_st_msi) @ inten)
    # Whether ST dopaminergic or not
    y = st.obs["dopamine"].astype(str).isin(pos).astype(int).to_numpy()
    if y.sum() in (0, len(y)): return np.nan, np.nan, s
    # Scale, fit-transform s
    s = MinMaxScaler().fit_transform(s.reshape(-1,1)).ravel()
    # Evaluate AUROC, AUPRC scores 
    return roc_auc_score(y,s), average_precision_score(y,s), s

def plot_curves(y, s, title):
    fpr,tpr,_ = roc_curve(y,s); prec,rec,_ = precision_recall_curve(y,s)
    plt.figure(figsize=(5,4.2)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--',lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {title}"); plt.grid(alpha=.2); plt.show()
    plt.figure(figsize=(5,4.2)); plt.plot(rec,prec); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — {title}"); plt.grid(alpha=.2); plt.show()

def plot_spatial(xy, c, title, cmap="inferno", s=10):
    plt.figure(figsize=(6,6)); plt.scatter(xy[:,0], xy[:,1], c=c, s=s, cmap=cmap)
    plt.gca().invert_yaxis(); plt.axis("equal"); plt.colorbar(label="scaled"); plt.title(title); plt.show()
