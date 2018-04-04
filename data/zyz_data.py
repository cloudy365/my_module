# -*- coding: utf-8 -*-
"""
module name: zyz_ml
description: Contains the common used data codes used in my research,
             Functions arranged by data types.
"""

from zyz_core import *



#########
##  写数据函数  ##
#########
def save_data_hdf5(filename, data_path, data):
    with h5py.File(filename, 'a') as h5f:
	h5f.create_dataset(data_path, data=data, compression='gzip', compression_opt=5)
    return
	

def save_data_ncdf(fout, data_in, lat_in=[], lon_in=[], tm=[]):
    """
    write the gridded dataset into netcdf format.
    """
    # 1. Created file
    nc_file = Dataset(fout, 'w')

    # 2. Created dimensions
    if len(lat_in) == 0:
    	lat_in=arange(70, 90)+0.5
    dim_lat = nc_file.createDimension('lat', len(lat_in))
    if len(lon_in) == 0:
    	lon_in=arange(360)+0.5
    dim_lon = nc_file.createDimension('lon', len(lon_in))
    if len(tm) > 0:
        dim_time = nc_file.createDimension('time', len(tm))
    print "-- Created dimensions"
    
    # 3. Define a variable to hold the data
    lat = nc_file.createVariable('lat', 'f4', ('lat',))
    lat.units = 'degree'
    lat.standard_name = 'latitude'
    lon = nc_file.createVariable('lon', 'f4', ('lon',))
    lon.units = 'degree'
    lon.standard_name = 'longitude'
    if len(tm) > 0:
        time = nc_file.createVariable('time', 'f4', ('time',))
        time.units = ' '
        time.standard_name = 'time'
    
    if len(tm) > 0:
        data = nc_file.createVariable('var', 'f4', ('time', 'lat', 'lon'))
    else:
        data = nc_file.createVariable('var', 'f4', ('lat', 'lon'))
    data.standard_name = 'variable'
    print "-- Created variables with attributes"
    
    # 4. Write the data
    lat[:] = lat_in
    lon[:] = lon_in # arange(360)+0.5 CERES; arange(-180, 180)+0.5 MISR
    if len(tm) > 0:
        time[:] = tm
        data[:,:,:] = data_in
    else:
        data[:, :] = data_in
    
    # 5. Close the file
    nc_file.close()
    print '*** SUCCESS writing example file %s!'%fout
    return 
    
    
    
#########
##  显示数据函数  ##
#########
def view_data_polar_stereo(lons, lats, var, plot_type='pcolormesh', clevs=None, title=None, clabel=None):
    print lons.shape, lats.shape, var.shape
    # set up the drawing board
    # plt.figure(figsize=(9, 6))
    m = Basemap(projection='npstere',boundinglat=45,lon_0=0,resolution='l',round=True)
    m.drawcoastlines(linewidth=0.5)
    # m.fillcontinents(color='aqua', lake_color='aqua', alpha=0.5)
    m.drawparallels(np.arange(-80.,81.,20.),dashes=[4,1],linewidth=0.5)
    m.drawmeridians(np.arange(-180.,181.,30.),dashes=[4,1],linewidth=0.5)
    
    # show data
    if plot_type == 'pcolormesh':
        cs = m.pcolormesh(lons,lats,var,latlon=True,shading='gouraud')
    elif plot_type == 'contourf':
        print "Remember to specify {clevs} ..."
        clevs = clevs
        cs = m.contourf(lons,lats,var,clevs,latlon=True,alpha=0.7)
    elif plot_type == 'scatter':
        cs = m.scatter(lons,lats,color='r',latlon=True,marker='o',s=60)
    
    # add colorbar and label
    cbar = m.colorbar(cs,location='bottom',pad="5%")
    if clabel != None:
        cbar.set_label("{}".format(clabel))
    if title != None:
        plt.title("{}".format(title))
    return


def view_data_1d_hist(x, vmax=None, vmin=None, bins=20):
    """
    Draw 1-D histogram (%).
    """
    if vmax == None:
        max_value = np.max(x)
    else:
        max_value = vmax
    
    if vmin == None:
        min_value = np.min(x)
    else:
        min_value = vmin
        
    x = [min_value if i < min_value else i for i in x]
    x = np.array([max_value if i > max_value else i for i in x])
    res = plt.hist(x, weights=np.zeros_like(x)+100./x.size, bins=bins, alpha=0.5, color='b', range=[min_value, max_value])
    x_axis = np.linspace(vmin, vmax, bins)
    plt.plot(x_axis, 2*norm.pdf(x_axis,np.mean(x),np.std(x)), 'r-', lw=3, alpha=0.6)
    return res


def draw_2d_hist(x, y, x_scale=[0,1.4], y_scale=[0, 1.4]):
    """
    Draw 2D-histogram.
    """
    # Estimate the 2D histogram
    nbins = 100
    
    #tot = float(len(x))
    H, xedges, yedges = histogram2d(x, y, bins=nbins, range=[x_scale, y_scale])
    
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
    print H.shape
    # Mask zeros
    Hmasked = ma.masked_where(H==0,H) # Mask pixels with a value of zero
    
    # Plot 2D histogram using pcolor
    pcolormesh(xedges,yedges,Hmasked)
    cbar = colorbar()
    cbar.ax.set_ylabel('Counts')
    show()


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = concatenate((linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in xrange(N+1) ]
    # Return colormap object.

    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)



#############
##  COMMON ##
#############
def arc_ocean_mask(instrument):
    """
    Input: "CERES" or "MISR" or "MERRA"
    Get Arctic Ocean mask based on ftp://daac.ornl.gov/data/islscp_ii/ancillary/combined_ancillary_xdeg/data/
    Return the arctic ocean mask (20 * 360) array. Mask out the land grids.
    """
    raw_data = np.loadtxt(ocean_mask_1d)
    
    if instrument == 'CERES':
        # select Arctic part
        # because CERES ranges from 70.5 -> 89.5 N (20 points)
        arctic_0 = raw_data[1:21, :]
        # transform to CERES format
        # transform latitude, as CERES lat 70 -> 90 N
        arctic_1 = arctic_0[::-1]
        # transform longitude, as CERES lon 0.5 -> 359.5 E
        arctic_2 = np.array([np.append(arctic_1[i, 180:], arctic_1[i, :180]) for i in range(20)])
        m = np.ma.masked_equal(arctic_2, 1)
        
    elif instrument == 'MISR':
        # select Arctic part
        # because CERES ranges from 70.5 -> 89.5 N (20 points)
        arctic_0 = raw_data[1:21, :]
        # transform to MISR format
        # transform latitude, as MISR lat 70 -> 90 N
        arctic_1 = arctic_0[::-1]
        m = np.ma.masked_equal(arctic_1, 1)
        
    elif instrument == 'MERRA': # same as ERA-Interim
        arctic_0 = raw_data[:21, :]
        # keep it
        m = np.ma.masked_equal(arctic_0, 1)
        
    elif instrument == 'CFSR':
        raw_data = np.loadtxt(cean_mask_hd)
        # select Arctic part
        arctic_0 = raw_data[:41, :]     
        # transform longitude to 0 -> 359.5 E
        arctic_1 = np.array([np.append(arctic_0[i, 360:], arctic_0[i, :360]) for i in range(41)])
        m = np.ma.masked_equal(arctic_1, 1)
    
    elif instrument == 'JRA55':
        # convert 0.25 deg ocean mask to 1.25 deg
        raw_data = np.loadtxt(ocean_mask_qd)
        # select Arctic part
        arctic_0 = raw_data[:86, :]
        
        tmp_mask = [[] for i in range(17)]    
        for ilat in range(17):
            for ilon in range(288):
                tmp = arctic_0[ilat*5:ilat*5+5, ilon*5:ilon*5+5]
                if any(tmp.ravel() == 1):
                    tmp_mask[ilat].append(1)
                else:
                    tmp_mask[ilat].append(0)   
        arctic_0 = np.array(tmp_mask)
        # transform longitude to 0 -> 359.5 E
        arctic_1 = np.array([np.append(arctic_0[i, 144:], arctic_0[i, :144]) for i in range(17)])
        m = np.ma.masked_equal(arctic_1, 1)
        
    return m


def area_weighted_mean(data, lat):
    """
    Function used to calculate area-weighted Arctic mean value. 
    Write this function because different datasets may have different shapes. 
    This function then serves as an universal function.
    Data should be a 2-D array while lat should be a 1-D array, 
    and they should share the same first dimension.
    """
    lat_mean_data = np.nanmean(data, axis=1)
    lat_new = []
    for ilat, idata in zip(lat, lat_mean_data):
        if np.isnan(idata) == False:
            lat_new.append(ilat)
        else:
            lat_new.append(np.nan)
    area_mean = np.nansum(lat_mean_data*np.cos(np.deg2rad(lat))) / np.nansum(np.cos(np.deg2rad(lat)))
    
    return area_mean



#############
##  CERES  ##
#############
# functions used to process CERES Level-3 data
def CERES_LV3_grid(dset, fld_name, year, month, version=4.0):
    
    if dset == 'ebaf_toa':
        if version == 4.0:
            ncfile = ceres_dir + "/EBAF/CERES_EBAF-TOA_Ed4.0_Subset_200003-201707.nc"
        elif version == 2.8:
            ncfile = ceres_dir + "/EBAF/CERES_EBAF-TOA_Ed2.8_Subset_200003-201702.nc"
    elif dset == 'ebaf_sfc':
        if version == 4.0:
            ncfile = ceres_dir + "/EBAF/CERES_EBAF-Surface_Ed4.0_Subset_200003-201701.nc"
        elif version == 2.8:
            ncfile = ceres_dir + "/EBAF/CERES_EBAF-Surface_Ed2.8_Subset_200003-201702.nc"
    elif dset == 'ssf1deg':
        if year == 2017:
            ncfile = ceres_dir + "/SSF1deg/monthly/CERES_SSF1deg-Month_Terra-MODIS_Ed4A_Subset_201703-201708.nc"
        else:
            ncfile = ceres_dir + "/SSF1deg/monthly/CERES_SSF1deg-Month_Terra-MODIS_Ed4A_Subset_{0}03-{0}12.nc".format(year)
    else:
        print "Wrong dataset ..."
        return
    
    ncdf = Dataset(ncfile)
    fld = ncdf.variables[fld_name][:]
    
    if dset == 'ssf1deg':
        idx = month-3
    else:
        idx = (year-2000)*12 + (month-3)
        
    if (idx < 0) or (idx >= len(fld)): 
        print "Your selected time <{}.{}> is out of the time boundary, please choose a time within {}..."\
                    .format(year, month, ncfile.split('_')[-1][:-3])
        out_data = np.nan
    else:
        out_data = fld[idx]
    
    return out_data


def CERES_LV3_Arctic_mean(dset, fld_name, year, month, ocean_mask, version=4.0):
    m = arc_ocean_mask('CERES')
    
    lat = np.arange(70, 90) + 0.5
    fld = CERES_LV3_grid(dset, fld_name, year, month, version)[-20:, :]
    
    if ocean_mask:
        fld_ocean = np.ma.masked_array(fld, mask=m.mask)
    else:
        fld_ocean = fld

    arc_mean = area_weighted_mean(fld_ocean, lat)

    return arc_mean




############
##  MISR  ##
############
# functions used to process MISR CGAL data
def MISR_LV3_grid(dset, fld_name, year, month):
    """
    -------------
    | MISR-Arctic area grid data|
    -------------
    """

    if dset == 'CGAL':
        ncfile = misr_dir + "/CGAL/{0}/MISR_AM1_CGAL_1_DEG_F04_0024_{0}_{1}".format(year, str(month).zfill(2))    
    elif dset == 'CGAL':
        ncfile = misr_dir + "/CGCL/{0}/MISR_AM1_CGCL_0_5_DEG_F04_0024_{0}_{1}".format(year, str(month).zfill(2))  
    else:
        print "Wrong dataset ..."
        return
    
    ncdf = Dataset(ncfile)
    data = ncdf.variables[fld_name][:]
    # print "Retrieve \"{}\" from MISR \"{}\" dataset successfully, data shape in {}.".format(fld_name, dset, data.shape)
    return np.array(data)


def MISR_LV3_Arctic_mean_TOA_alb_RSR(alb_type, alb_band, year, month, ocean_mask, output='toa_rsr'):
    """
    alb_type could be 'Restrictive_albedo_average', 'Expansive_albedo_average' or 'Local_albedo_average';
    alb_band could be 0: blue, 1: green, 2: red, 3: NIR, or 4: broadband.
    """
    m = arc_ocean_mask('MISR')
    
    lat = np.arange(70, 90) + 0.5
    if alb_type in ['Restrictive_albedo_average', 'Expansive_albedo_average', 'Local_albedo_average']:
        alb = MISR_LV3_grid("CGAL", alb_type, year, month)[alb_band][0][-20:]
        insol_name = "{}_albedo_solar_insolation".format(alb_type.split('_')[0])
        sol = MISR_LV3_grid("CGAL", insol_name, year, month)[alb_band][0][-20:]
        
        if ocean_mask:
            # use ocean mask
            alb = np.ma.masked_array(alb, mask=m.mask)
            alb = np.ma.masked_equal(alb, -9999.0)
            sol = np.ma.masked_array(sol, mask=m.mask)
            sol = np.ma.masked_equal(sol, -9999.0)
        else:
            alb = np.ma.masked_equal(alb, -9999.0) 
            sol = np.ma.masked_equal(sol, -9999.0) 
            
        rsr = alb * sol
        arc_rsr = area_weighted_mean(rsr, lat)
        
        if output == 'toa_rsr':
            arc_mean = arc_rsr
        elif output == 'toa_alb':
            arc_mean = arc_rsr / area_weighted_mean(sol, lat)
            
    else:
        arc_mean = np.nan
    
    return arc_mean



############
##  MODIS  ##
############
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
        # 3retrieve 48 bit MODIS Cloud Mask SDS product
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
#     for i in tqdm(range(2030), miniters=50):
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
            # print bit_0, bit_1_2, bit_3, bit_4, bit_5, bit_6_7

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


############
##  NSIDC ##
############
# functions used to process NSIDC Sea Ice Index (V3.0) data
def NSIDC_SeaIceIndex_month(imon):
    '''
    Referred to G02135 dataset (V3.0).
    39-year monthly values between 1979 and 2016.
    '''
    file_path = "/u/sciteam/smzyz/data/Satellite/NSIDC/G02135/N_{}_area.txt".format(str(imon).zfill(2))
    f = open(file_path, 'r')
    data = f.readlines()
    
    sie = [float(iline.split(',')[4]) for iline in data]
    sie = np.ma.masked_values(sie, -9999.)
    
    return sie


def NSIDC_read_psn25lats():
    """
    
    """
    filename = '/u/sciteam/smzyz/data/Satellite/NSIDC/psn25lats_v3.dat'
    f = open(filename, 'rb')
    b_data = f.read()
    
    data = []
    for i in range(0, len(b_data)/4):
        data.append(unpack('I', b_data[i*4:i*4+4])[0])
    data = np.reshape(data, (448, 304))
    return np.array(data)/100000.


def NSIDC_read_psn25lons():
    filename = '/u/sciteam/smzyz/data/Satellite/NSIDC/psn25lons_v3.dat'
    f = open(filename, 'rb')
    b_data = f.read()
    
    data = []
    for i in range(0, len(b_data)/4):
        data.append(unpack('i', b_data[i*4:i*4+4])[0])
    data = np.reshape(data, (448, 304))
    return np.array(data)/100000.


def NSIDC_read_sic(filename):
    '''
    sea-ice concentration, valid value 0~250.
    Return sea-ice concentration (0-1.0).
    '''
    f = open(filename, 'rb')
    b_data = f.read()
    if len(b_data) != 136492:
        print 'dataset is incomplete... please have a check before further processing...'
    
    data = []
    for i in range(300, len(b_data)):
        data.append(unpack('B', b_data[i])[0])
    data = np.reshape(data, (448, 304))

    # mask 251-255
    data = np.ma.masked_greater_equal(data, 251)
    return data/250.


def NSIDC_read_mo(filename):
    '''
    Return melt onset date.
    '''
    f = open(filename, 'rb')
    b_data = f.read()
    if len(b_data) != 136192:
        print 'dataset shape do not match (448, 304), choosing the first 136192 values to unpack ...'
        
    data = []
    for i in range(0, len(b_data)):
        data.append(unpack('B', b_data[i])[0])
    data = np.reshape(data[:136192], (448, 304))
    return data


