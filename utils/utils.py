import PIL.Image as Image
import numpy as np
import scipy.signal


def load_image(filename, luminance=True) :
    img = Image.open(filename)
    img.load()
    img_array = np.asarray(img, dtype="int32")
    if luminance:
        img_array = img_array.mean(axis=2)
    return img_array


def resize(img_array, s=0.5, interpolation=Image.BICUBIC):
    """
    :param s - desired scaling factor
    :param interpolation - see https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize
    """
    size = (s * np.array(img_array.shape[0:2])).astype(np.int)
    return np.array(Image.fromarray(img_array).resize(size, interpolation))

def get_features(patch, augmented=True):
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
    if augmented:
        Fs.append(np.sqrt(Fs[0]**2 + Fs[1]**2))
        Fs.append(np.sqrt(Fs[2]**2 + Fs[3]**2))
    Fs = np.dstack(Fs)
    return Fs
