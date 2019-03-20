import warnings
import numpy as np
from sklearn.decomposition import PCA


class IdentityLSH:
    def __init__(self):
        pass
    
    def fit(self, X, n_iter=None):
        return self
    
    def fit_transform(self, X, n_iter=None):
        return X
    
    def transform(self, X):
        return X

    
class PcaLSH(PCA):
    def __init__(self, *args, **kwargs):
        super(PcaLSH, self).__init__(*args, **kwargs)
    
    def fit(self, X, n_iter=None):
        super().fit(X)
        return self
    
    def fit_transform(self, X, n_iter=None):
        return super().fit_transform(X)
    
    def transform(self, X):
        return super().transform(X)
    
    
class RotationLSH:
    def __init__(self, binary):
        self.R = None
        self.binary = binary
    
    
    def fit(self, X, n_iter=20):
        assert X.ndim == 2
        m, k = X.shape
        
        # construct initial orthonormal matrix
        temp = np.random.randn(k,k)
        R, _, _ = np.linalg.svd(temp)
        B = np.empty((m,k))
        
        for i in range(n_iter):
            B = np.sign(X @ R)
            U, _, Vt = np.linalg.svd(B.T @ X)
            R = U @ Vt
        self.R = R
        return self
    
    
    def fit_transform(self, X, n_iter=20):
        self.fit(X, n_iter)
        return self.transform(X)
    
    
    def transform(self, X):
        X_comp = X @ self.R
        if self.binary:
            X_comp = np.sign(X_comp)
        return X_comp
