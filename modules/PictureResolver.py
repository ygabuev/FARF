import numpy as np
import skimage.util
import utils

class PictureResolver:
    # TODO: make size-informed and shape-agnostic
    # currently based on 9x9 patches
    
    def __init__(self, rf, lsh):
        self.rf = rf
        self.lsh = lsh
    
    def resolve(self, img_in, patch_size=(9,9), step=6):
        assert img_in.ndim == 2
        
        features = utils.get_features(img_in)
        n_features = features.shape[-1]
        patches = self._split_into_patches(features, (*patch_size, n_features), step=6)
        patches_arr_size = patches.shape[0:2]

        X = np.reshape(patches, (np.prod(patches_arr_size), -1))
        X_comp = self.lsh.transform(X)
        Y_pred = self.rf.predict(X, X_comp)                # MOST EXPENSIVE ACTION
        
        patches = np.reshape(Y_pred, (*patches_arr_size, *patch_size))
        
        img_out_delta = np.zeros(img_in.shape)
        div_coef = np.zeros(img_in.shape)
        for i in range(patches_arr_size[0]):
            for j in range(patches_arr_size[1]):
                patch = patches[i,j]
                p = patch_size[0]
                img_out_delta[step*i:step*i+p, step*j:step*j+p] += patch
                div_coef[step*i:step*i+p, step*j:step*j+p] += np.ones(patch_size)
        img_out_delta /= div_coef
        return img_in + img_out_delta
    
    def _split_into_patches(self, img_array, patch_size, step=6):
        patches = skimage.util.view_as_windows(img_array, patch_size, step=step)
        return patches
