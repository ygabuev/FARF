import numpy as np
from sklearn.decomposition import PCA


class IdentityLSH:
    def __init__(self):
        pass
    
    def fit_transform(self, X, n_iter=None):
        return X
    
    def fit(self, X, n_iter=None):
        return self
    
    def transform(self, X):
        return X

    
class PcaLSH(PCA):
    def __init__(self, *args, **kwargs):
        super(PcaLSH, self).__init__(*args, **kwargs)
    
    
class RotationLSH:
    def __init__(self):
        self.R = None
    
    def fit_transform(self, X, n_iter=5):
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
        return X @ R
    
    def fit(self, X, n_iter=5):
        self.fit_transform(X, n_iter)
        return self
    
    def transform(self, X, binary=False):
        X_comp = X @ self.R
        if binary:
            X_comp = np.sign(X_comp)
        return X_comp
