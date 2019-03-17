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
