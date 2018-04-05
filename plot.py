
# -*- coding: utf-8 -*-

from . import np, bytescale


def enhance_rgb(rgb_array, coeff=None):
    """
    Inputs: a rgb array with band dimension at last;
            a coeff used to normalize all three bands, default is the maximum value of blue band
    Output: an enhanced rgb array.
    """

    r = rgb_array[:, :, 0]
    g = rgb_array[:, :, 1]
    b = rgb_array[:, :, 2]
    
    if coeff == None:
        coeff = b.max() # Maximum blue radiance

    rgb = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = scale_image_2d(bytescale(r / coeff))
    rgb[:, :, 1] = scale_image_2d(bytescale(g / coeff))
    rgb[:, :, 2] = scale_image_2d(bytescale(b / coeff))
    
    return rgb


def scale_image_2d(image):
    """
    Rescale a two-dimensional image to fill the 0--255 range.
    Input: a 2-D image array, which should be np.uint8
    Output: a scaled 2-D image array.
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