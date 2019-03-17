import warnings
import numpy as np
from sklearn.decomposition import PCA


class IdentityLSH:
    def __init__(self):
        pass
    
    def fit(self, X, n_iter=None):
        return self
    
    def fit_transform(self, X, n_iter=None, binary=None):
        if binary == True:
            warnings.warn("'binary=True' is only meaningful for RotationLSH")
        return X
    
    def transform(self, X, binary=None):
        if binary == True:
            warnings.warn("'binary=True' is only meaningful for RotationLSH")
        return X

    
class PcaLSH(PCA):
    def __init__(self, *args, **kwargs):
        super(PcaLSH, self).__init__(*args, **kwargs)
    
    def fit(self, X, n_iter=None):
        super().fit(X)
        return self
    
    def fit_transform(self, X, n_iter=None, binary=None):
        if binary == True:
            warnings.warn("'binary=True' is only meaningful for RotationLSH")
        return super().fit_transform(X)
    
    def transform(self, X, binary=None):
        if binary == True:
            warnings.warn("'binary=True' is only meaningful for RotationLSH")
        return super().transform(X)
    
    
class RotationLSH:
    def __init__(self):
        self.R = None
    
    def fit(self, X, n_iter=5):
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
    
    def fit_transform(self, X, n_iter=20, binary=False):
        self.fit(X, n_iter)
        return self.transform(X, binary)
    
    def transform(self, X, binary=False):
        X_comp = X @ self.R
        if binary:
            X_comp = np.sign(X_comp)
        return X_comp
