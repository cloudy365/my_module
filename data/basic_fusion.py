
# -*- coding: utf-8 -*-

#from .. import h5py, np

__all__ = [
		"bf_info",
		"get_rad_latlon",
		"get_rgb",
		]


from .. import h5py, np


def bf_info(file_path, instrument):
    """
    Input: basic fusion file path, instrument
    Output: data structure
    """
    h5f = h5py.File(file_path, 'r')
    a = h5f[instrument]
    
    for _, i in a.iteritems():
        lev1 = i.name
        print ""
        print lev1

        for _, j in h5f[lev1].iteritems():
            lev2 = j.name
            lev2_main = h5f[lev2]

            if isinstance(lev2_main, h5py.Dataset):
                print "  ", lev2, j.maxshape
            else:
                print "  ", lev2

                for _, k in lev2_main.iteritems():
                    lev3 = k.name
                    lev3_main = h5f[lev3]
                    if isinstance(lev3_main, h5py.Dataset):
                        print "    ", lev3, k.maxshape
                    else:        
                        print "    ", lev3

                        for _, l in lev3_main.iteritems():
                            lev4 = l.name
                            lev4_main = h5f[lev4]
                            if isinstance(lev4_main, h5py.Dataset):
                                print "      ", lev4, l.maxshape
                            else:
                                print "      ", lev4


def get_rad_latlon(bf_path, instrument, granblock, band=None, camres=None):
    """
    Note: Inputs vary for different instruments, working on single granule/block only.
    ASTER requires granule (str), band (int);
    MODIS requires granule (str), band (could be float for 13.5 and 14.5), res could choose from 
                   1) None (default), 
                   2) 'Aggr500' (for band 1/2),
                   3) 'Aggr1km' (for band 1--7);
    MISR requires block (int), band (e.g., 'Red'), camera (e.g., 'AN')
    MOPITT requires nothing and will output its all bands.
    """

    h5f = h5py.File(bf_path, 'r')
    
    try:
        if instrument == 'ASTER':
            bands_VNIR = ['ImageData1', 'ImageData2', 'ImageData3N']
            bands_TIR =  ['ImageData10', 'ImageData11', 'ImageData12', 'ImageData13', 'ImageData14']
            if band in [1, 2, 3]:
                radiance = h5f['ASTER/{}/VNIR/{}'.format(granblock, bands_VNIR[band-1])][:]
                lat = h5f['ASTER/{}/VNIR/Geolocation/Latitude'.format(granblock)][:]
                lon = h5f['ASTER/{}/VNIR/Geolocation/Longitude'.format(granblock)][:]
            elif band in range(10, 15):
                radiance = h5f['ASTER/{}/TIR/{}'.format(granblock, bands_TIR[band-10])][:]
                lat = h5f['ASTER/{}/TIR/Geolocation/Latitude'.format(granblock)][:]
                lon = h5f['ASTER/{}/TIR/Geolocation/Longitude'.format(granblock)][:]
        

        elif instrument == 'MISR':
            radiance = h5f['MISR/{}/Data_Fields/{}_Radiance'.format(camres[:2], band)][granblock]
            if radiance.shape[0] == 512:
                lat = h5f['MISR/HRGeolocation/GeoLatitude'][granblock]
                lon = h5f['MISR/HRGeolocation/GeoLongitude'][granblock]
            elif radiance.shape[0] == 128:
                lat = h5f['MISR/Geolocation/GeoLatitude'][granblock]
                lon = h5f['MISR/Geolocation/GeoLongitude'][granblock]
            
                                                
        elif instrument == 'MODIS':
            bands_250 = np.array([1, 2])
            bands_500 = np.array([3, 4, 5, 6, 7])
            bands_1km_RefSB = np.array([8, 9, 10, 11, 12, 13, 13.5, 14, 14.5, 15, 16, 17, 18, 19, 26])
            bands_1km_Emissive = np.array([20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])
            if camres == None:
                if band in bands_250:
                    idx = np.where(bands_250==band)[0][0]
                    radiance = h5f['MODIS/{}/_250m/Data_Fields/EV_250_RefSB'.format(granblock)][idx]
                    lat = h5f['MODIS/{}/_250m/Geolocation/Latitude'.format(granblock)][:]
                    lon = h5f['MODIS/{}/_250m/Geolocation/Longitude'.format(granblock)][:]  
                elif band in bands_500:
                    idx = np.where(bands_500==band)[0][0]
                    radiance = h5f['MODIS/{}/_500m/Data_Fields/EV_500_RefSB'.format(granblock)][idx]
                    lat = h5f['MODIS/{}/_500m/Geolocation/Latitude'.format(granblock)][:]
                    lon = h5f['MODIS/{}/_500m/Geolocation/Longitude'.format(granblock)][:]  
                elif band in bands_1km_RefSB:
                    idx = np.where(bands_1km_RefSB==band)[0][0]
                    radiance = h5f['MODIS/{}/_1KM/Data_Fields/EV_1KM_RefSB'.format(granblock)][idx]
                    lat = h5f['MODIS/{}/_1KM/Geolocation/Latitude'.format(granblock)][:]
                    lon = h5f['MODIS/{}/_1KM/Geolocation/Longitude'.format(granblock)][:]
                elif band in bands_1km_Emissive:
                    idx = np.where(bands_1km_Emissive==band)[0][0]
                    radiance = h5f['MODIS/{}/_1KM/Data_Fields/EV_1KM_Emissive'.format(granblock)][idx]
                    lat = h5f['MODIS/{}/_1KM/Geolocation/Latitude'.format(granblock)][:]
                    lon = h5f['MODIS/{}/_1KM/Geolocation/Longitude'.format(granblock)][:]

            elif camres == 'Aggr500':
                if band not in bands_250:
                    raise Exception("Invalid band for Aggr500, which should be 1 or 2.", band)
                else:
                    idx = np.where(bands_250==band)[0][0]
                    radiance = h5f['MODIS/{}/_500m/Data_Fields/EV_250_Aggr500_RefSB'.format(granblock)][idx]
                    lat = h5f['MODIS/{}/_500m/Geolocation/Latitude'.format(granblock)][:]
                    lon = h5f['MODIS/{}/_500m/Geolocation/Longitude'.format(granblock)][:]  
            
            elif camres == 'Aggr1km':
                if band not in np.append(bands_250, bands_500):
                    raise Exception("Invalid band for Aggr1km, which should be 1 -- 7.", band)
                elif band in bands_250:
                    idx = np.where(bands_250==band)[0][0]
                    radiance = h5f['MODIS/{}/_1KM/Data_Fields/EV_250_Aggr1km_RefSB'.format(granblock)][idx]
                    lat = h5f['MODIS/{}/_1KM/Geolocation/Latitude'.format(granblock)][:]
                    lon = h5f['MODIS/{}/_1KM/Geolocation/Longitude'.format(granblock)][:]  
                else:
                    idx = np.where(bands_500==band)[0][0]
                    radiance = h5f['MODIS/{}/_1KM/Data_Fields/EV_500_Aggr1km_RefSB'.format(granblock)][idx]
                    lat = h5f['MODIS/{}/_1KM/Geolocation/Latitude'.format(granblock)][:]
                    lon = h5f['MODIS/{}/_1KM/Geolocation/Longitude'.format(granblock)][:]


        elif instrument == 'MOPITT':
            radiance = h5f['/MOPITT/{}/Data_Fields/MOPITTRadiances'.format(granblock)][:, :, :, :, 0]
            lat = h5f['MOPITT/{}/Geolocation/Latitude'.format(granblock)][:]
            lon = h5f['MOPITT/{}/Geolocation/Longitude'.format(granblock)][:]
        
        np.place(radiance, radiance<0, 0)

    except Exception as err:
        print ">> You have selected instrument: {}, granule/block: {}, band: {}, and camera: {}, but failed.".format(
        instrument, granblock, band, camres)
        print ">> Please choose another combination."
        print ">> Err message: {}".format(err)
        return [], [], []
    
    return radiance, lat, lon


def get_rgb(bf_path, instrument, granblock, camera=None, customize_rgb_bands=[]):
    """
    Note: Inputs vary for different instruments.
    ASTER requires granule (str) and represents R, G, B using 3N, 2, 1 bands (15 m);
    MODIS requires granule (str) and represents R, G, B using 1, 4, 3 bands (500 m);
    MISR requires block (int) and camera (CAPTICAL), represents R, G, B using red, green, and blue bands (1.1 km).
         If camera is "AN", the output resolution is 275 m. If camera is "AN_1km", the output resolution is 1.1 km.
    
    Note that the customize_rgb_bands only works for MODIS.
    """

    h5f = h5py.File(bf_path, 'r')
    
    if instrument == 'MODIS':
        if len(customize_rgb_bands) == 0:
            print ">> Retrieving NORMAL MODIS RGB, granule: {}.".format(granblock)
            granule = granblock
            r = h5f['/MODIS/{}/_500m/Data_Fields/EV_250_Aggr500_RefSB'.format(granule)][0]
            g = h5f['/MODIS/{}/_500m/Data_Fields/EV_500_RefSB'.format(granule)][1]
            b = h5f['/MODIS/{}/_500m/Data_Fields/EV_500_RefSB'.format(granule)][0]
        else:
            print ">> Retrieving USER-DEFINED MODIS RGB, granule: {}.".format(granblock)
            print ">> Red: band {}, Green: band {}, Blue: band {}".format(customize_rgb_bands[0], 
                                                                          customize_rgb_bands[1],
                                                                          customize_rgb_bands[2])
            granule = granblock
            
            tmp = []
            if any(np.array(customize_rgb_bands)>7):
                # Aggr1km
                for iband in customize_rgb_bands:
                    if iband in range(1, 8):
                        rad, _, _ = get_rad_latlon(bf_path, instrument, granule, iband, camres='Aggr1km')
                    else:
                        rad, _, _ = get_rad_latlon(bf_path, instrument, granule, iband)
                    tmp.append(rad)
            else:
                # Aggr500
                for iband in customize_rgb_bands:
                    if iband in range(1, 3):
                        rad, _, _ = get_rad_latlon(bf_path, instrument, granule, iband, camres='Aggr500')
                    else:
                        rad, _, _ = get_rad_latlon(bf_path, instrument, granule, iband)
                    tmp.append(rad)
            r = tmp[0]
            g = tmp[1]
            b = tmp[2]

    
    elif instrument == 'MISR':
        print ">> Retrieving MISR RGB, camera: {}, block: {}.".format(camera, granblock)
        
        if len(granblock) > 1:
            r = h5f['/MISR/{}/Data_Fields/Red_Radiance'.format(camera[:2])][granblock[0]:granblock[1]]
            g = h5f['/MISR/{}/Data_Fields/Green_Radiance'.format(camera[:2])][granblock[0]:granblock[1]]
            b = h5f['/MISR/{}/Data_Fields/Blue_Radiance'.format(camera[:2])][granblock[0]:granblock[1]]
        else:
            r = h5f['/MISR/{}/Data_Fields/Red_Radiance'.format(camera[:2])][granblock]
            g = h5f['/MISR/{}/Data_Fields/Green_Radiance'.format(camera[:2])][granblock]
            b = h5f['/MISR/{}/Data_Fields/Blue_Radiance'.format(camera[:2])][granblock]
        
        if camera == 'AN': 
            r = r[0]
            g = g[0]
            b = b[0]
            print r.shape
        # deresolution of 275 m -> 1.1 km
        elif camera == 'AN_1km':
            r_new = []
            g_new = []
            b_new = []
            for iblk in range(len(granblock)):
                tmp1 = np.array([[r[iblk, j*4:(j+1)*4, i*4:(i+1)*4].mean(axis=1).mean(axis=0) 
                                 for i in range(512)] for j in range(128)])
                tmp2 = np.array([[g[iblk, j*4:(j+1)*4, i*4:(i+1)*4].mean(axis=1).mean(axis=0) 
                                 for i in range(512)] for j in range(128)])
                tmp3 = np.array([[b[iblk, j*4:(j+1)*4, i*4:(i+1)*4].mean(axis=1).mean(axis=0) 
                                 for i in range(512)] for j in range(128)])
                r_new.append(tmp1)
                g_new.append(tmp2)
                b_new.append(tmp3)
                
            r = np.array(r_new)
            g = np.array(g_new)
            b = np.array(b_new)  
        else:
            r_new = []
            for iblk in range(len(granblock)):
                tmp = np.array([[r[iblk, j*4:(j+1)*4, i*4:(i+1)*4].mean(axis=1).mean(axis=0) 
                                 for i in range(512)] for j in range(128)])
                r_new.append(tmp)
            r = np.array(r_new)
        
        # Stack blocks together
        # if len(granblock) > 1:
        r = np.vstack(r)
        g = np.vstack(g)
        b = np.vstack(b)
    

    elif instrument == 'ASTER':
        print ">> Retrieving ASTER RGB, granule: {}.".format(granblock)
        r = h5f['/ASTER/{}/VNIR/ImageData3N'.format(granblock)][:]
        g = h5f['/ASTER/{}/VNIR/ImageData2'.format(granblock)][:]
        b = h5f['/ASTER/{}/VNIR/ImageData1'.format(granblock)][:]
    
    
    np.place(r, (r<0)|(g<0)|(b<0), 0)
    np.place(g, (r<0)|(g<0)|(b<0), 0)
    np.place(b, (r<0)|(g<0)|(b<0), 0)
    
    rgb = []
    rgb.append(r)
    rgb.append(g)
    rgb.append(b)
    rgb = np.rollaxis(np.array(rgb), 0, 3)

    return rgb


def get_bounding_latlon(bf_path, instrument, granblock):
    with h5py.File(bf_path, 'r') as h5f:
        
        if instrument == 'ASTER':
            lat = h5f['/ASTER/{}/VNIR/Geolocation/Latitude'.format(granblock)][:]
            lon = h5f['/ASTER/{}/VNIR/Geolocation/Longitude'.format(granblock)][:]
            
        elif instrument == 'MISR':
            lat = h5f['/MISR/Geolocation/GeoLatitude'][granblock]
            lon = h5f['/MISR/Geolocation/GeoLongitude'][granblock]
            lat = np.vstack(lat)
            lon = np.vstack(lon)
            
        elif instrument == 'MODIS':
            lat = h5f['/MODIS/{}/_1KM/Geolocation/Latitude'.format(granblock)][:]
            lon = h5f['/MODIS/{}/_1KM/Geolocation/Longitude'.format(granblock)][:]
    
    lats = [ lat[0, 0], lat[-1, 0], lat[-1, -1], lat[0, -1] ]
    lons = [ lon[0, 0], lon[-1, 0], lon[-1, -1], lon[0, -1] ] 
    
    return lats, lons