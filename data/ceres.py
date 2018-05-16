
# -*- coding: utf-8 -*-


from .. import np, Dataset
from .. import ceres_dir


__all__ = ["CERES_LV3_grid","CERES_LV3_Arctic_mean"]


def CERES_LV3_grid(dset, fld_name, year, month, version=4.0):
    """
    Inputs: dset chose from 'ebaf_toa', 'ebaf_sfc', 'ssf1deg';
    		fld_name;
    		year;
    		month;
    		version (default is 4.0)
    Output: chosen data field, please note the dimension could be different.
    """

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
    """
    Inputs: dset chose from 'ebaf_toa', 'ebaf_sfc', 'ssf1deg';
    		fld_name;
    		year;
    		month;
    		ocean_mask (True or False);
    		version (default is 4.0)
    Output: chosen data field, please note the dimension could be different.
    """

    m = arc_ocean_mask('CERES')
    
    lat = np.arange(70, 90) + 0.5
    fld = CERES_LV3_grid(dset, fld_name, year, month, version)[-20:, :]
    
    if ocean_mask:
        fld_ocean = np.ma.masked_array(fld, mask=m.mask)
    else:
        fld_ocean = fld

    arc_mean = area_weighted_mean(fld_ocean, lat)

    return arc_mean