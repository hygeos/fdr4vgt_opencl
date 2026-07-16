from probav_vito import Level1_probav
from fdr4vgt.spotvgt_vito import Level1_spotvgt
import xarray as xa
from datetime import date, datetime
from netCDF4 import Dataset, set_chunk_cache
import numpy as np
from glob import glob
import dask as dk
from core import interpolate
from funcs import memory_tracker


# Définition des attributs CF standards
#cf_attrs = {
#    'lat': {
#        'standard_name': 'latitude',
#        'long_name': 'Latitude',
#        'units': 'degrees_north',
#    },
#    'lon': {
#        'standard_name': 'longitude',
#        'long_name': 'Longitude',
#        'units': 'degrees_east',
#    },
#    'SZA': {
#        'standard_name': 'solar_zenith_angle',
#        'long_name': 'Solar zenith angle',
#        'units': 'degrees',
#        'valid_min': 0.0,
#        'valid_max': 90.0
#    },
#    'SAA': {
#        'standard_name': 'solar_azimuth_angle',
#        'long_name': 'Solar azimuth angle',
#        'units': 'degrees',
#        'valid_min': 0.0,
#        'valid_max': 360.0
#    },
#    'VZA': {
#        'standard_name': 'sensor_zenith_angle',
#        'long_name': 'Sensor zenith angle',
#        'units': 'degrees',
#        'valid_min': 0.0,
#        'valid_max': 90.0
#    },
#    'VAA': {
#        'standard_name': 'sensor_azimuth_angle',
#        'long_name': 'Sensor azimuth angle',
#        'units': 'degrees',
#        'valid_min': 0.0,
#        'valid_max': 360.0
#    },
#    'VZA_IR': {
#        'standard_name': 'sensor_zenith_angle',
#        'long_name': 'SWIR sensor zenith angle',
#        'units': 'degrees',
#        'valid_min': 0.0,
#        'valid_max': 90.0
#    },
#    'VAA_IR': {
#        'standard_name': 'sensor_azimuth_angle',
#        'long_name': 'SWIR sensor azimuth angle',
#        'units': 'degrees',
#        'valid_min': 0.0,
#        'valid_max': 360.0
#    },
#    'rTOC': {
#        'standard_name': 'top_of_canopy_reflectance',
#        'long_name': 'Top of canopy reflectance',
#        'units': '1',
#        'valid_min': 0.0,
#        'valid_max': 1.0
#    },
#}

def load_brdf(dirname, date, lat, lon, nbands, chunks):
    '''
    Load BRDF coefficients for a given date
    Inputs:
        dirname : directory containing BRDF files
        date    : date as a numpy.datetime64 object
        chunks  : chunk size for dask
    Outputs:
        xarray.Dataset containing BRDF coefficients
    '''

    filename = '{}/*{}*.nc'.format(dirname, np.datetime_as_string(date, unit='D').replace('-',''))
    #filename = '/mnt/ceph/proj/FDR4VGT/input/brdf/c3s_brdf_20050810000000_X32Y08_AVHRR_NOAA17_V1.0.1.nc'
    filename = glob(filename)

    if len(filename) != 1:
        k1p = xa.DataArray(np.zeros((nbands, lat.shape[0], lat.shape[1]), dtype='float32'), dims=('bands', 'y','x'))
        return k1p, k1p

    filename = filename[0]
    ds = xa.open_dataset(filename) #, chunks={'y':chunks, 'x':chunks})

    lat_axis = ds['LAT'][:,0]
    lon_axis = ds['LON'][0,:]
    ds = ds.rename_dims({'Y': 'lat_axis', 'X': 'lon_axis'})
    ds = ds.assign_coords({
        'lat_axis': lat_axis.values,
        'lon_axis': lon_axis.values
    })
    k012 = interpolate.interp(ds['K012'], lat_axis=interpolate.Linear(lat), lon_axis=interpolate.Linear(lon)).astype(np.float32)
    k0 = k012[:, :, :,0]
    k1 = k012[:, :, :,1]
    k2 = k012[:, :, :,2]
    k1p = k1/k0
    k2p = k2/k0
    filtre = np.isfinite(k1p)
    k1p = k1p.where(filtre, other=0.0)
    k1p = k1p.transpose('NBAND','y','x')
    filtre = np.isfinite(k2p)
    k2p = k2p.where(filtre, other=0.0)
    k2p = k2p.transpose('NBAND','y','x')

    return k1p, k2p

def Level1(input, sensor, smac_dir, version,chunks=None):
    '''
    Read a Level1 product as an xarray.Dataset
    '''

    func = {'PROBA-V': Level1_probav, 'SPOTVGT1': Level1_spotvgt, 'SPOTVGT2': Level1_spotvgt}
    
    ds = func[sensor](input, smac_dir, version, sensor, chunks)

    return ds   

def create_nc(filename, gl_size, bands, attrs, version):
    '''
    Start netCDF output file creation
    Inputs:
        filename : string of absolute path
        gl_size  : a tuple containing height and width
        attrs    : a list oa attributs
        version  : string containing version

    Outputs:
        a NETCDF4 Dataset
    '''
    print("save netCDF : {}".format(filename))
    # Shrink the per-variable HDF5 chunk cache (default 64 MiB/var). With ~84
    # variables that default caps aggregate chunk RAM at ~5.25 GB and, combined
    # with the library's large auto-chunks (~1683x1557), keeps the whole file
    # resident. The variables are written in tile-aligned 512x512 chunks (see
    # save_nc_batch chunksizes), so a small cache holding a few 1 MiB chunks is
    # ample; this is set globally before the variables are created so each new
    # dataset inherits it. Cuts writer steady-state RSS from ~5 GB to ~0.2 GB
    # with no change to the stored data.
    set_chunk_cache(4 * 1024 * 1024, 1009, 0.75)
    out = Dataset(filename, 'w', format='NETCDF4')

    for att, value in attrs:
        out.setncattr(att, value)

    out.date_created = str(datetime.now())

    width = gl_size[1]
    height = gl_size[0]

    out.createDimension('height', height)
    out.createDimension('width', width)
    out.createDimension('bands', len(bands))

#    # Ajouter les attributs globaux
#    out.setncatts({
#        'Conventions': 'CF-1.8',
#        'title': 'PROBA-V Level 1 data',
#        'institution': 'VITO',
#        'source': 'PROBA-V satellite observations',
#        'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
#        'references': 'https://proba-v.vgt.vito.be/',
#        'bands': ','.join(bands)
#    })

    out.close()

    return out

def attrs_corr(attr):
    if 'time_find_index' in attr: del attr['time_find_index']
    if 'time_interp' in attr: del attr['time_interp']
#    for key, values in attr.items():
#        print('key :',key, type(values))
#        if isinstance(values, datetime):
#            attr[key] = values.strftime("%Y-%m-%dT%H:%M:%S")
#            print("attr :" ,key, attr[key])

    return attr

#@memory_tracker
#def save_nc_batch(out, ds_in, ds_out, iband, jband, band_size, error=False): #, debug=False):  
def save_nc_batch(out, ds_in, ds_out, iband, jband, band_size, error=False): 
    '''
    Save a batch of data to a NetCDF file.
    out : an open netCDF4 Dataset, OR a filename string. When a filename is
          given the file is opened in append mode and closed here (legacy path);
          when an open Dataset is given it is reused and left open by the caller
          (keeping the handle open across tiles avoids 100x HDF5 open/close).
    ds_in : xarray.Dataset containing input datk
    ds_out : xarray.Dataset containing output data
    '''

    close_after = False
    if isinstance(out, str):
        out = Dataset(out, 'a', format='NETCDF4')
        close_after = True

    # netCDF4/HDF5 has no half-precision type; widen any float16 variables (e.g.
    # ERROR and the float16 MERRA fields kept in RAM to save memory) to float32
    # for writing. The per-tile datasets are small, so this cast is cheap.
    for _ds in (ds_in, ds_out):
        for _v in list(_ds.data_vars):
            if _ds[_v].dtype == np.float16:
                _ds[_v] = _ds[_v].astype(np.float32)

    x_start = jband * band_size
    x_end = min(x_start + band_size, out.dimensions['width'].size)
    y_start = iband * band_size
    y_end = min(y_start + band_size, out.dimensions['height'].size)

#    list_vars_out = ['rTOC', 'UrTOC', 'flag']
    list_vars_out = {'Rtoc_': 'rTOC', 'Rtoc_uncertainty_': 'UrTOC', 'uncertainty_from_RTM_terrain_':'urtoc_terrain', 'Quality_flag': 'flag'}
    #list_vars_in = {'SM_MAP_': 'SM_MAP', 'clm':'clm'}
    list_vars_in = {'clm':'clm', 'TOA_':'TOA','SM_MAP':'SM_MAP'}
    if error:
        list_vars_out = {**list_vars_out, **{'Jacobian_Rtoc_vs_Rtoa_': 'Jrtoa', 'Jacobian_Rtoc_vs_UO3_': 'Juo3', 'Jacobian_Rtoc_vs_UH2O_': 'Juh2o',
                                            'Jacobian_Rtoc_vs_Ps_': 'Jpre', 'Jacobian_Rtoc_vs_AOD_': 'Jtau550',
                                            'uncertainty_from_ozone_': 'unc_o3', 'uncertainty_from_h2o_': 'unc_h2o', 'uncertainty_from_pressure_': 'unc_ps', 'uncertainty_from_aod_': 'unc_aot',
                                            'uncertainty_from_RTM_BRDF_':'UrTOC_rtm_brdf', 'uncertainty_from_RTM_fit_':'UrTOC_rtm_fit', 'uncertainty_from_aerosol_':'UrTOC_ens'} 
                                              }
        list_vars_in = {**list_vars_in, **{'uncertainty_from_TOA_':'ERROR'}}
    list_vars_out = {**list_vars_out, **{'aerosol_model_index':'iaero'}}
    list_vars_in = {**list_vars_in, **{'AOD_550_used': 'TOTEXTTAU', 'ozone_column_used': 'TO3', 'water_vapor_column_used': 'TQV', 'surface_pressure_used': 'SLP'}}

    # global attributes
    out.setncatts(ds_out.attrs)
    existing_vars = set(out.variables.keys())

    for var in ['lat','lon','SZA', 'SAA', 'VZA', 'VAA', 'VZA_IR', 'VAA_IR']:
        if var in ds_in.variables.keys():
#            if var not in out.variables.keys():
            if var not in existing_vars:
                dtype = ds_in[var].dtype
                out.createVariable(var, dtype, ('height','width'), zlib=True, complevel=1, chunksizes=(band_size, band_size))
                existing_attrs = ds_in[var].attrs if hasattr(ds_in[var], 'attrs') else {}
                out[var].setncatts({**existing_attrs})
#            out.variables[var][y_start:y_end,:] = ds_in[var].values
            out.variables[var][y_start:y_end,x_start:x_end] = ds_in[var].values

    for var_out, var_in in list_vars_out.items():
        if len(ds_out[var_in].dims) == 3:
            for ib in range(ds_out[var_in].shape[0]):
                var_out_band = '{}B{}'.format(var_out, ib+1)
#                if var_out_band not in out.variables.keys():
                if var_out_band not in existing_vars:
                    dtype = ds_out[var_in].dtype
                    out.createVariable(var_out_band, dtype, ('height','width'), zlib=True, complevel=1, chunksizes=(band_size, band_size))
                    existing_attrs = ds_out[var_in].attrs if hasattr(ds_out[var_in], 'attrs') else {}
                    out[var_out_band].setncatts({**existing_attrs})
#                out.variables[var_out_band][y_start:y_end, :] = ds_out[var_in][ib, :, :].values
                out.variables[var_out_band][y_start:y_end, x_start:x_end] = ds_out[var_in][ib, :, :].values
        elif len(ds_out[var_in].dims) == 2:
            if var_out not in existing_vars:
                dtype = ds_out[var_in].dtype
                out.createVariable(var_out, dtype, ('height','width'), zlib=True, complevel=1, chunksizes=(band_size, band_size))
                existing_attrs = ds_out[var_in].attrs if hasattr(ds_out[var_in], 'attrs') else {}
                out[var_out].setncatts({**existing_attrs})
#            out.variables[var_out][y_start:y_end, :] = ds_out[var_in].values
            out.variables[var_out][y_start:y_end, x_start:x_end] = ds_out[var_in].values


    for var_out, var_in in list_vars_in.items():
        if len(ds_in[var_in].dims) == 3:
            for ib in range(ds_in[var_in].shape[0]):
                var_out_band = '{}B{}'.format(var_out, ib+1)
                if var_out_band not in existing_vars:
                    dtype = ds_in[var_in].dtype
                    out.createVariable(var_out_band, dtype, ('height','width'), zlib=True, complevel=1, chunksizes=(band_size, band_size))
                    existing_attrs = ds_in[var_in].attrs if hasattr(ds_in[var_in], 'attrs') else {}
                    out[var_out_band].setncatts({**existing_attrs})
#                out.variables[var_out_band][y_start:y_end, :] = ds_in[var_in][ib, :, :].values
                out.variables[var_out_band][y_start:y_end, x_start:x_end] = ds_in[var_in][ib, :, :].values
        elif len(ds_in[var_in].dims) == 2:
            if var_out not in existing_vars:
                dtype = ds_in[var_in].dtype
                out.createVariable(var_out, dtype, ('height','width'), zlib=True, complevel=1, chunksizes=(band_size, band_size))
                existing_attrs = ds_in[var_in].attrs if hasattr(ds_in[var_in], 'attrs') else {}
                existing_attrs = attrs_corr(existing_attrs)
                out[var_out].setncatts({**existing_attrs})
#            out.variables[var_out][y_start:y_end, :] = ds_in[var_in].values
            out.variables[var_out][y_start:y_end, x_start:x_end] = ds_in[var_in].values
    if close_after:
        out.close()

def save_nc(ds, filename):
    '''
    Save an xarray.Dataset to a NetCDF file following CF conventions.
    '''
    out = xa.Dataset()
    
    # Définition des attributs CF standards
#    cf_attrs = {
#        'lat': {
#            'standard_name': 'latitude',
#            'long_name': 'Latitude',
#            'units': 'degrees_north',
#        },
#        'lon': {
#            'standard_name': 'longitude',
#            'long_name': 'Longitude',
#            'units': 'degrees_east',
#        },
#        'SZA': {
#            'standard_name': 'solar_zenith_angle',
#            'long_name': 'Solar zenith angle',
#            'units': 'degrees',
#            'valid_min': 0.0,
#            'valid_max': 90.0
#        },
#        'SAA': {
#            'standard_name': 'solar_azimuth_angle',
#            'long_name': 'Solar azimuth angle',
#            'units': 'degrees',
#            'valid_min': 0.0,
#            'valid_max': 360.0
#        },
#        'VZA': {
#            'standard_name': 'sensor_zenith_angle',
#            'long_name': 'Sensor zenith angle',
#            'units': 'degrees',
#            'valid_min': 0.0,
#            'valid_max': 90.0
#        },
#        'VAA': {
#            'standard_name': 'sensor_azimuth_angle',
#            'long_name': 'Sensor azimuth angle',
#            'units': 'degrees',
#            'valid_min': 0.0,
#            'valid_max': 360.0
#        },
#        'VZA_IR': {
#            'standard_name': 'sensor_zenith_angle',
#            'long_name': 'SWIR sensor zenith angle',
#            'units': 'degrees',
#            'valid_min': 0.0,
#            'valid_max': 90.0
#        },
#        'VAA_IR': {
#            'standard_name': 'sensor_azimuth_angle',
#            'long_name': 'SWIR sensor azimuth angle',
#            'units': 'degrees',
#            'valid_min': 0.0,
#            'valid_max': 360.0
#        },
#        'TOC': {
#            'standard_name': 'top_of_canopy_reflectance',
#            'long_name': 'Top of canopy reflectance',
#            'units': '1',
#            'valid_min': 0.0,
#            'valid_max': 1.0
#        },
#    }

    # Ajouter les attributs globaux
#    out.attrs = {
#        'Conventions': 'CF-1.8',
#        'title': 'PROBA-V Level 1 data',
#        'institution': 'VITO',
#        'source': 'PROBA-V satellite observations',
#        'history': f'Created {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
#        'references': 'https://proba-v.vgt.vito.be/',
#    }

    # save geometry with CF attributes
    for var in ['lat', 'lon', 'SZA', 'SAA', 'VZA', 'VAA', 'VZA_IR', 'VAA_IR']:
        if var in ds.variables.keys():
            out[var] = ds[var]
            # Fusionner les attributs existants avec les attributs CF
            existing_attrs = ds[var].attrs if hasattr(ds[var], 'attrs') else {}
            out[var].attrs = {**cf_attrs[var], **existing_attrs}

    for iband, band in enumerate(ds.attrs['bands']):
        if 'rsurf_best' in ds.variables.keys():
            var = 'rsurf_best_{}'.format(band)
            out[var] = ds['rsurf_best'][iband]
            existing_attrs = ds['rsurf_best'].attrs if hasattr(ds['rsurf_best'], 'attrs') else {}
            out[var].attrs = {**cf_attrs['TOC'], **existing_attrs}
        if 'rsurf_mean' in ds.variables.keys():
            var = 'rsurf_mean_{}'.format(band)
            out[var] = ds['rsurf_mean'][iband]

#        if 'TOC' in ds.variables.keys():
#            var =  'TOC_{}'.format(band)
#            out[var] = ds['TOC'][iband]
#            existing_attrs = ds['TOC'].attrs if hasattr(ds['TOC'], 'attrs') else {}
#            out[var].attrs = {**cf_attrs['TOC'], **existing_attrs}

        if 'jacobians' in ds.variables.keys():
            for ijac, jac in enumerate(ds.attrs['jac_name']):
                out['{}_{}'.format(jac, band)] = ds['jacobians'][ijac, iband]

    out.to_netcdf(filename)