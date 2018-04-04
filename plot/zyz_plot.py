from zyz_core import *
from scipy.misc import bytescale



def scale_image(image):
    """
    Rescale a two-dimensional image to fill the 0--255 range. 
    """
    along_track = image.shape[0]
    cross_track = image.shape[1]

    x = np.array([0, 30, 60, 120, 190, 255], dtype=np.uint8)
    y = np.array([0, 110, 160, 210, 240, 255], dtype=np.uint8)

    scaled = np.zeros((along_track, cross_track), dtype=np.uint8)
    for i in range(len(x) - 1):
        x1 = x[i]
        x2 = x[i + 1]
        y1 = y[i]
        y2 = y[i + 1]
        m = (y2 - y1) / float(x2 - x1)
        b = y2 - (m * x2)
        mask = ((image >= x1) & (image < x2))
        scaled = scaled + mask * np.asarray(m * image + b, dtype=np.uint8)

    mask = image >= x2
    scaled = scaled + (mask * 255)
    return scaled


