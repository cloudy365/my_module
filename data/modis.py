


from .. import np, SD


###########
#  MODIS  #
###########


# functions used to process MODIS MOD02 data
def MODIS_bands_specification():
    # retrieve MODIS021KM radiances from one specific channel
    bands_250 = np.array([1, 2])
    bands_500 = np.array([3, 4, 5, 6, 7])
    bands_1km_RefSB = np.array([8, 9, 10, 11, 12, 13, 13.5, 14, 14.5, 15, 16, 17, 18, 19, 26])
    bands_1km_Emissive = np.array([20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])
    print "SW bands (250 m):{}".format(bands_250)
    print "SW bands (500 m):{}".format(bands_500)
    print "SW bands (1 km):{}".format(bands_1km_RefSB)
    print "LW bands (1 km):{}".format(bands_1km_Emissive)
    

def MOD02_retrieve_radiance(mod02_file, iband, iref):
    # retrieve MODIS021KM radiances from one specific channel
    bands_250 = np.array([1, 2])
    bands_500 = np.array([3, 4, 5, 6, 7])
    bands_1km_RefSB = np.array([8, 9, 10, 11, 12, 13, 13.5, 14, 14.5, 15, 16, 17, 18, 19, 26])
    bands_1km_Emissive = np.array([20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])
    
    mfile = SD(mod02_file)
    if iband in bands_250:
        mdata = mfile.select('EV_250_Aggr1km_RefSB')
        if iref == False:
            scales = mdata.attributes()['radiance_scales']
            offset = mdata.attributes()['radiance_offsets']
        else:
            scales = mdata.attributes()['reflectance_scales']
            offset = mdata.attributes()['reflectance_offsets']
        
        idx = np.where(bands_250==iband)[0][0]
        rad = (mdata.get()[idx] - offset[idx]) * scales[idx] 
        
    elif iband in bands_500:
        mdata = mfile.select('EV_500_Aggr1km_RefSB')
        if iref == False:
            scales = mdata.attributes()['radiance_scales']
            offset = mdata.attributes()['radiance_offsets']
        else:
            scales = mdata.attributes()['reflectance_scales']
            offset = mdata.attributes()['reflectance_offsets']
        
        idx = np.where(bands_500==iband)[0][0]
        rad = (mdata.get()[idx] - offset[idx]) * scales[idx] 
        
    elif iband in bands_1km_RefSB:
        mdata = mfile.select('EV_1KM_RefSB')
        if iref == False:
            scales = mdata.attributes()['radiance_scales']
            offset = mdata.attributes()['radiance_offsets']
        else:
            scales = mdata.attributes()['reflectance_scales']
            offset = mdata.attributes()['reflectance_offsets']
        
        idx = np.where(bands_1km_RefSB==iband)[0][0]
        rad =(mdata.get()[idx] - offset[idx]) * scales[idx] 
        
    elif iband in bands_1km_Emissive:
        mdata = mfile.select('EV_1KM_Emissive')
        rad_scales = mdata.attributes()['radiance_scales']
        rad_offset = mdata.attributes()['radiance_offsets']
        
        idx = np.where(bands_1km_Emissive==iband)[0][0]
        rad = (mdata.get()[idx] - rad_offset[idx]) * rad_scales[idx] 
    return rad
    

def MOD02_select_chs(mod02_file, chs, iref=False):
    mod_array = []
    for ich in chs:
        mod_array.append( MOD02_retrieve_radiance(mod02_file, ich, iref) )
    mod_array = np.rollaxis(np.array(mod_array), 0, 3)
    return mod_array


def MOD02_RGB_channels(mod02_file, iref=False):
    rgb_chs = [1, 4, 3]
    rgb = MOD02_select_chs(mod02_file, rgb_chs, iref)
    if iref:
        rgb_out = rgb
    else:
        rgb_out = rgb / np.nanmax(rgb[:, :, -1])
    return rgb_out
        

def MOD02_17_channels(mod02_file):
    rad_band_1 = MOD02_retrieve_radiance(mod02_file, 1) 
    rad_band_4 = MOD02_retrieve_radiance(mod02_file, 4)
    rad_band_3 = MOD02_retrieve_radiance(mod02_file, 3)
    rad_band_26 = MOD02_retrieve_radiance(mod02_file, 26)
    rad_band_2 = MOD02_retrieve_radiance(mod02_file, 2)
    rad_band_5 = MOD02_retrieve_radiance(mod02_file, 5)
    rad_band_6 = MOD02_retrieve_radiance(mod02_file, 6)
    rad_band_7 = MOD02_retrieve_radiance(mod02_file, 7)
    rad_band_8 = MOD02_retrieve_radiance(mod02_file, 8)
    rad_band_20 = MOD02_retrieve_radiance(mod02_file, 20)    
    rad_band_27 = MOD02_retrieve_radiance(mod02_file, 27)
    rad_band_28 = MOD02_retrieve_radiance(mod02_file, 28)
    rad_band_29 = MOD02_retrieve_radiance(mod02_file, 29)
    rad_band_31 = MOD02_retrieve_radiance(mod02_file, 31)
    rad_band_32 = MOD02_retrieve_radiance(mod02_file, 32)
    rad_band_33 = MOD02_retrieve_radiance(mod02_file, 33)
    rad_band_35 = MOD02_retrieve_radiance(mod02_file, 35)
    
    rad_17_bands = np.dstack((rad_band_1, rad_band_2, rad_band_3, rad_band_4, rad_band_5, rad_band_6, \
                             rad_band_7, rad_band_8, rad_band_20, rad_band_26, rad_band_27, rad_band_28, \
                             rad_band_29, rad_band_31, rad_band_32, rad_band_33, rad_band_35))
    return rad_17_bands
    

def MOD35_retrieve_all(mod35_file):
    """
    Retrieve cloud mask and viewing geometry data from a MOD35 file, using MOD35_cloud_mask and MOD35_viewing_geometry
    functions.
    INPUT: MOD35 file path
    OUTPUT: An numpy.array containing:
            [0] Cloud Mask Flag                    0=not determined   1=determined
            [1] Unobstructed FOV Quality Flag      0=cloudy  1=prob. cloudy  2=prob. clear  3=clear
            [2] Day/Night Flag                     0=Night            1=Day
            [3] Sun glint Flag                     0=Yes              1=No
            [4] Snow/Ice Background Flag           0=Yes              1=No
            [5] Land/Water Flag                    0=Water   1=Coastal       2=Desert       3=Land
            [6] Snow cover from ancillary map      0=Yes              1=No
            [7] Solar zenith angles (in degree)
            [8] Solar azimuth angles (in degree)
            [9] Sensor zenith angles (in degree)
            [10] Sensor azimuth angles (in degree)
    """
    mod35_cm = MOD35_retrieve_cloud_mask(mod35_file)
    mod35_vg = MOD35_retrieve_viewing_geometry(mod35_file)
    if mod35_cm.shape[:2] != mod35_vg.shape[:2]:
        print "Dataset shape is inconsistent, please check MOD35 file: {}".format(mod35_file)
        mod35_all = []
    else:
        mod35_all = np.concatenate((mod35_cm, mod35_vg), axis=2)
    
    return mod35_all


def MOD35_retrieve_cloud_mask(mod35_file):
    """
    use first 8 bits information
    INPUT: MOD35 file path
    OUTPUT: An numpy.array containing:
            [0] Cloud Mask Flag                    0=not determined   1=determined
            [1] Unobstructed FOV Quality Flag      0=cloudy  1=prob. cloudy  2=prob. clear  3=clear
            [2] Day/Night Flag                     0=Night            1=Day
            [3] Sun glint Flag                     0=Yes              1=No
            [4] Snow/Ice Background Flag           0=Yes              1=No
            [5] Land/Water Flag                    0=Water   1=Coastal       2=Desert       3=Land
            [6] Snow cover from ancillary map      0=Yes              1=No
    """
        
    def subfunc_retrieve_cloud_mask(icloud_mask):
        # retrieve 48 bit MODIS Cloud Mask SDS product
        # icloud_mask should have the shape of (6,), and dtype of 'uint8'.
        mask = ''
        for ibyte in icloud_mask:
            tmp = bin(ibyte)[2:].zfill(8)
            mask = mask + tmp[::-1]
        return mask

    
    mfile = SD(mod35_file)
    mdata = np.array(mfile.select('Cloud_Mask')[:], dtype='uint8')
    mdata = np.rollaxis(mdata, 0, 3) # roll first dimension to last

    flag_cloud_mask = [[] for i in range(mdata.shape[0])]
    for i in range(mdata.shape[0]):
        for j in range(mdata.shape[1]):
            data_48_bits = subfunc_retrieve_cloud_mask(mdata[i, j])
            bit_0 = data_48_bits[0]
            bit_1_2 = data_48_bits[1:3]
            bit_3 = data_48_bits[3]
            bit_4 = data_48_bits[4]
            bit_5 = data_48_bits[5]
            bit_6_7 = data_48_bits[6:8]
            bit_10 = data_48_bits[10]

            out = []
            out.append(bit_0)
            out.append(int(bit_1_2, 2))
            out.append(bit_3)
            out.append(bit_4)
            out.append(bit_5)
            out.append(int(bit_6_7, 2))
            out.append(bit_10)
            
            flag_cloud_mask[i].append(out)
            
    flag_cloud_mask = np.array(flag_cloud_mask, dtype='int8')
    return flag_cloud_mask
    

def MOD35_retrieve_viewing_geometry(mod35_file):
    # MOD35 viewing geometry data (VGD)
    def subfunc_retrieve_viewing_geometry(ifield):
        mdata = mfile.select(ifield)[:]/100.
        
        #   Since VGD's shape is (408/406, 270), I expand it to (408/406, 271) first and then repeat
        # by a factor of 5, making it to (2040/2030, 1355). Finally select (2040/2030, 1354) to 
        # match the MOD35 scene data.
        vgd = []
        for i in mdata:
            tmp = np.append(i, i[-1])
            vgd.append(tmp)
        vgd = np.array(vgd)
        
        vgd = np.repeat(vgd, 5, axis=0)
        vgd = np.repeat(vgd, 5, axis=1)
        
        vgd_fnl = vgd[:, :-1]
        return vgd_fnl
    
    mfile = SD(mod35_file)
    
    mod_array = []
    field_names = ['Solar_Zenith', 'Solar_Azimuth', 'Sensor_Zenith', 'Sensor_Azimuth']
    for ifield in field_names :
        mod_array.append( subfunc_retrieve_viewing_geometry(ifield) )
    
    mod_array = np.rollaxis(np.array(mod_array), 0, 3)
    return mod_array
