import numpy as np
import skimage.util
import scipy.signal


def get_features(patch):
    """
    Extract the following features from the input patch:
      - first-order gradients  [-1,0,1] in both directions
      - second-order gradients [1,0,-2,0,1] in both directions
      - first-order gradient magnitude
      - second-order gradient magnitude
    """
    assert patch.ndim == 2
    
    k1 = np.array([[-1,0,1]])
    k2 = k1.T
    k3 = np.array([[1,0,-2,0,1]])
    k4 = k3.T
    
    Fs = []
    for k in [k1,k2,k3,k4]:
        F = scipy.signal.convolve2d(patch, k, mode='same')
        Fs.append(F)
    Fs.append(np.sqrt(Fs[0]**2 + Fs[1]**2))
    Fs.append(np.sqrt(Fs[2]**2 + Fs[3]**2))
    Fs = np.dstack(Fs)
    return Fs


class PictureResolver:
    # TODO: make size-informed and shape-agnostic
    # currently based on 9x9 patches and 6 features
    
    def __init__(self, rf, lsh):
        self.rf = rf
        self.lsh = lsh
    
    def resolve(self, img_in, patch_size=(9,9), step=6):
        assert img_in.ndim == 2
        
        features = get_features(img_in)
        patches = self._split_into_patches(features, (*patch_size, 6), step=6)
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
