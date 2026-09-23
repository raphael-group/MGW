import numpy as np
import torch

class NumpyPairLoader:
    """
    Minimal loader that expects .npy files:
      - s.npy (n x dim_e), X.npy (n x dim_f1)
      - t.npy (m x dim_e), Z.npy (m x dim_f2)
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load(self):
        s = np.load(f"{self.base_dir}/s.npy")  # (n, dim_e)
        X = np.load(f"{self.base_dir}/X.npy")  # (n, dim_f1)
        t = np.load(f"{self.base_dir}/t.npy")  # (m, dim_e)
        Z = np.load(f"{self.base_dir}/Z.npy")  # (m, dim_f2)
        return s, X, t, Z
