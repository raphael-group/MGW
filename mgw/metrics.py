from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np

def Alignment_Clus_Metrics(X, Z, P, k=4, random_state=0):
    
    # Cluster each modality independently
    clx = KMeans(k, random_state=random_state).fit_predict(X)
    clz = KMeans(k, random_state=random_state).fit_predict(Z)
    
    # Compute weighted AMI under coupling P (normalized joint)
    P_norm = P / P.sum()
    
    # Sample from P to estimate AMI under coupling
    n_samp = 20000
    idx_x = np.random.choice(P.shape[0], size=n_samp, p=P_norm.sum(1))
    idx_z = [np.random.choice(P.shape[1], p = P_norm[i,:] / P_norm[i,:].sum() ) for i in idx_x]
    
    ami = adjusted_mutual_info_score(clx[idx_x], clz[idx_z])
    print("AMI(X,Z) under coupling:", ami)
    ari = adjusted_rand_score(clx[idx_x], clz[idx_z])
    print("ARI(X,Z) under coupling:", ari)
    
    return ami, ari

