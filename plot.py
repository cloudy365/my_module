
# -*- coding: utf-8 -*-

from . import np, bytescale


def enhance_rgb(rgb_array, scale_method=['discrete']*3, scale_factors=[1]*3, cmins=None, cmaxs=None):
    """
    
    Inputs:
    1) rgb_array:      a rgb 3-D array with band dimension at last;
    2) scale_method:   either 'discrete' (default) or 'RLT';
    3) scale_factors:  used when method=='RLT;
    4) cmins:          the min values of radiances --> digital counts conversion for each band;
    5) cmaxs:          the max values of radiances --> digital counts conversion for each band;
            
    Output:     
    1) a rescaled enhanced 3-D rgb digital counts array.
    """

    r = rgb_array[:, :, 0]
    g = rgb_array[:, :, 1]
    b = rgb_array[:, :, 2]
    
    if cmaxs == None:
        cmaxs = [r.max(), g.max(), b.max()]
    if cmins == None:
        cmins = [r.min(), g.min(), b.min()]

    rgb = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = scale_image_2d(bytescale(r, cmin=cmins[0], cmax=cmaxs[0]), scale_method[0], scale_factors[0])
    rgb[:, :, 1] = scale_image_2d(bytescale(g, cmin=cmins[1], cmax=cmaxs[1]), scale_method[1], scale_factors[1])
    rgb[:, :, 2] = scale_image_2d(bytescale(b, cmin=cmins[2], cmax=cmaxs[2]), scale_method[2], scale_factors[2])
        
    return rgb



def scale_image_2d(image_array, method, scale_factor):
    """
    Rescale a two-dimensional image within the 0--255 range.
    
    Inputs:
    1) image_array:   a 2-D image array, which should be np.uint8;
    2) method:        'discrete' (default) or 'RLT;
    3) scale_factor:  used when method=='RLT';
    
    Output: 
    1) a rescaled 2-D image array.
    """
    
    image = bytescale(image_array)
    
    if method == 'discrete':
        along_track = image.shape[0]
        cross_track = image.shape[1]

        # test on normal rgb
        x = np.array([0, 30, 60, 120, 190, 255], dtype=np.uint8)
        y = np.array([0, 110, 160, 210, 240, 255], dtype=np.uint8)
        
        # test on band-6
        # x = np.array([0, 15, 30, 60, 120, 190, 255], dtype=np.uint8)
        # y = np.array([0, 70, 110, 160, 210, 240, 255], dtype=np.uint8)

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
    
    elif method == 'RLT':
        val_min = 0
        val_max = 255
        scaled = 255*(1.0*image/(val_max-val_min))**(1.0/scale_factor)

    elif method == 'RLT2':
        val_min = 0
        val_max = 255
        scaled = 255*(1.0*image/(val_max-val_min))**(1.0/scale_factor)
        
    return scaled
