from probav_vito import Level1_probav
import xarray as xa
from datetime import datetime
from netCDF4 import Dataset

# Définition des attributs CF standards
cf_attrs = {
    'lat': {
        'standard_name': 'latitude',
        'long_name': 'Latitude',
        'units': 'degrees_north',
    },
    'lon': {
        'standard_name': 'longitude',
        'long_name': 'Longitude',
        'units': 'degrees_east',
    },
    'SZA': {
        'standard_name': 'solar_zenith_angle',
        'long_name': 'Solar zenith angle',
        'units': 'degrees',
        'valid_min': 0.0,
        'valid_max': 90.0
    },
    'SAA': {
        'standard_name': 'solar_azimuth_angle',
        'long_name': 'Solar azimuth angle',
        'units': 'degrees',
        'valid_min': 0.0,
        'valid_max': 360.0
    },
    'VZA': {
        'standard_name': 'sensor_zenith_angle',
        'long_name': 'Sensor zenith angle',
        'units': 'degrees',
        'valid_min': 0.0,
        'valid_max': 90.0
    },
    'VAA': {
        'standard_name': 'sensor_azimuth_angle',
        'long_name': 'Sensor azimuth angle',
        'units': 'degrees',
        'valid_min': 0.0,
        'valid_max': 360.0
    },
    'VZA_IR': {
        'standard_name': 'sensor_zenith_angle',
        'long_name': 'SWIR sensor zenith angle',
        'units': 'degrees',
        'valid_min': 0.0,
        'valid_max': 90.0
    },
    'VAA_IR': {
        'standard_name': 'sensor_azimuth_angle',
        'long_name': 'SWIR sensor azimuth angle',
        'units': 'degrees',
        'valid_min': 0.0,
        'valid_max': 360.0
    },
    'rTOC': {
        'standard_name': 'top_of_canopy_reflectance',
        'long_name': 'Top of canopy reflectance',
        'units': '1',
        'valid_min': 0.0,
        'valid_max': 1.0
    },
}

def Level1(input, sensor, chunks=None):
    '''
    Read a Level1 product as an xarray.Dataset
    '''

    func = {'probav': Level1_probav}
    
    ds = func[sensor](input, chunks)

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
    out = Dataset(filename, 'w', format='NETCDF4')

    for att, value in attrs:
        out.setncattr(att, value)

    out.date_created = str(datetime.now())

    width = gl_size[1]
    height = gl_size[0]

    out.createDimension('height', height)
    out.createDimension('width', width)
    out.createDimension('bands', len(bands))

    # Ajouter les attributs globaux
    out.setncatts({
        'Conventions': 'CF-1.8',
        'title': 'PROBA-V Level 1 data',
        'institution': 'VITO',
        'source': 'PROBA-V satellite observations',
        'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'references': 'https://proba-v.vgt.vito.be/',
        'bands': ','.join(bands)
    })

    return out

def save_nc_batch(out, ds_in, ds_out, iband, band_size, error=False, debug=False):  
    '''
    Save a batch of data to a NetCDF file.
    out : pointer to the netCDF4 Dataset
    ds_in : xarray.Dataset containing input data
    ds_out : xarray.Dataset containing output data
    ''' 
    y_start = iband * band_size
    y_end = min(y_start + band_size, out.dimensions['height'].size)

    list_vars_out = ['rTOC', 'UrTOC', 'ac_process_flag','ac_flag']
    list_vars_in = {'SM_MAP': 'SM_MAP', 'clm':'clm'}
    if error:
        list_vars_out += ['err_slope', 'Drsurf', 'Jrtoa', 'Juh2o', 'Juo3', 'Jpre']
    if debug:
        list_vars_in = {**list_vars_in, **{'rTOA': 'TOA','elev':'elev','SLP':'SLP','TOTEXTTAU':'TOTEXTTAU', 'TQV':'TQV', 'TO3':'TO3', 'T10M':'T10M'}}

    for var in ['lat','lon','SZA', 'SAA', 'VZA', 'VAA', 'VZA_IR', 'VAA_IR']:
        if var in ds_in.variables.keys():
            if var not in out.variables.keys():
                out.createVariable(var, 'f4', ('height','width'), zlib=True, complevel=4)
                existing_attrs = ds_in[var].attrs if hasattr(ds_in[var], 'attrs') else {}
                out[var].setncatts({**cf_attrs[var], **existing_attrs})
            out.variables[var][y_start:y_end,:] = ds_in[var].values

#    for iband, band in enumerate(ds_in.attrs['bands']):
    for var in list_vars_out:
        print(var, ds_out[var].shape)
        if var in ds_out.variables.keys():
            if var not in out.variables.keys():
                if len(ds_out[var].dims) == 3:
                    out.createVariable(var, 'f4', ('bands', 'height','width'), zlib=True, complevel=4)
                elif len(ds_out[var].dims) == 2:
                    out.createVariable(var, 'f4', ('height','width'), zlib=True, complevel=4)
                if var in cf_attrs.keys():
                    existing_attrs = ds_out[var].attrs if hasattr(ds_out[var], 'attrs') else {}
                    out[var].setncatts({**cf_attrs[var], **existing_attrs})
            if len(ds_out[var].dims) == 3:
                out.variables[var][:,y_start:y_end, :] = ds_out[var].values
            elif len(ds_out[var].dims) == 2:
                out.variables[var][y_start:y_end, :] = ds_out[var].values

    for var_out, var_in in list_vars_in.items():
        if var_in in ds_in.variables.keys():
            if var_out not in out.variables.keys():
                if len(ds_in[var_in].dims) == 3:
                    out.createVariable(var_out, 'f4', ('bands','height','width'), zlib=True, complevel=4)
                elif len(ds_in[var_in].dims) == 2:
                    out.createVariable(var_out, 'f4', ('height','width'), zlib=True, complevel=4)
                if var_out in cf_attrs.keys():
                    existing_attrs = ds_in[var_in].attrs if hasattr(ds_in[var_in], 'attrs') else {}
                    out[var_out].setncatts(existing_attrs)
            if len(ds_in[var_in].dims) == 3:
                out.variables[var_out][:,y_start:y_end, :] = ds_in[var_in].values
            elif len(ds_in[var_in].dims) == 2:
                out.variables[var_out][y_start:y_end, :] = ds_in[var_in].values

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
    out.attrs = {
        'Conventions': 'CF-1.8',
        'title': 'PROBA-V Level 1 data',
        'institution': 'VITO',
        'source': 'PROBA-V satellite observations',
        'history': f'Created {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'references': 'https://proba-v.vgt.vito.be/',
    }

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