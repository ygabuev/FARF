import numpy as np

from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import Lasso

class RandomForestClustRegressor(RandomTreesEmbedding):
    def __init__(self,
                 n_estimators=10,
                 max_depth=15,
                 min_samples_split=2,  # can be replaced by min_samples_leaf
                 min_samples_leaf=64,
                 max_leaf_nodes=None,
                 n_jobs=None,
                 random_state=7,
                 warm_start=False,
                 bootstrap=True,
                 max_features=1,
                 alpha=0.1):
        super(RandomForestClustRegressor, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            n_jobs=n_jobs,
            random_state=random_state,
            warm_start=warm_start)
        
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.alpha = alpha # LASSO coefficient
        
        self.k_in = None
        self.k_out = None
        
        self.forest_is_fitted = False
        self.reg_is_fitted = False
        self.leaves_to_data_map = None
        self.leaves_to_reg_map = None
    
    
    def _fit_forest(self, X, *args):
        super().fit(X, *args)
        self.forest_is_fitted = True
        data_to_leaves_map = self.apply(X) # data_to_leaves_map[i,j] = index of leaf for tree j for sample i
        self._map_leaves_to_data(data_to_leaves_map)
    
    
    def _map_leaves_to_data(self, data_to_leaves_map):
        # construct dict {tree_i: {leaf_j: X_inds}}
        # TODO: make faster
        map = {}
        for i, tree in enumerate(data_to_leaves_map.T):
            map_tree = {}
            leaves = np.unique(tree)
            for leaf in leaves:
                inds = np.where(tree == leaf)
                map_tree[leaf] = inds
            map[i] = map_tree
        self.leaves_to_data_map = map
    
    
    def _fit_leaf_regressions(self, X, Y):
        assert self.forest_is_fitted
        
        map = {}
        for tree, leaves in self.leaves_to_data_map.items():
            tree_map = {}
            for leaf, inds in leaves.items():
                X_ = X[inds]
                Y_ = Y[inds]
                reg = Lasso(alpha=self.alpha, fit_intercept=True)
                reg.fit(X_, Y_)
                tree_map[leaf] = reg
            map[tree] = tree_map
        self.leaves_to_reg_map = map
        self.reg_is_fitted = True
    
    
    def fit(self, X, X_comp, Y, *args):
        # X      - the full augmented features
        # X_comp - compressed features
        
        self.k_in = X.shape[1]
        self.k_out = Y.shape[1]
        
        self._fit_forest(X_comp, *args)
        self._fit_leaf_regressions(X, Y)
        return self
    
    
    def predict(self, X, X_comp):
        assert self.forest_is_fitted and self.reg_is_fitted
        m = X.shape[0]
        
        data_to_leaves_map = self.apply(X_comp)
        Y_pred = np.empty((m, self.k_out))
        
        for i, (x, leaves) in enumerate(zip(X, data_to_leaves_map)):
            x = x.reshape(1, -1)
            y_pred = np.zeros(self.k_out)
            for tree, leaf in enumerate(leaves):
                reg = self.leaves_to_reg_map[tree][leaf]
                y_pred_tree = reg.predict(x).flatten()
                y_pred += y_pred_tree
            y_pred /= self.n_estimators
            Y_pred[i] = y_pred
        return Y_pred
