import xarray as xr
from glob import glob
import numpy as np
from os.path import basename
from dask.array import meshgrid
from dask.array import stack as dask_stack
from pathlib import Path

def date_to_float(d, epoch=np.datetime64('1980-01-01T00:00:00.000000000')):
    '''
        transform the date into a duration in minutes since epoch
        by default from '1980-01-01T00:00:00.000000000'
    '''
    
    return (d - epoch).astype(np.float64)/1.0e9/60.

def Level1_spotvgt(dirname,
                smac_dir,
                version,
                sensor,
                chunks=None,
                ):
    '''
    Read an SpotVGT Level1 product as an xarray.Dataset
    '''

    ds = read_spotvgt(dirname,
                    smac_dir,
                    version,
                    sensor,
                    chunks
                    )
    return ds

def read_spotvgt_variable(filename, chunks, dtype='float32'):
    '''
    Read a variable from a SpotVGT Level1 product.
    '''
    src = xr.open_dataset(filename, chunks=chunks)
    varname = list(src.variables.keys())
    assert(len(varname)==1)
    varname = varname[0]

    data = src[varname]
    if dtype == 'uint16' and data.dtype == 'uint16':
        return data

    if 'SCALE' in src[varname].attrs.keys():
        scale = np.float32(src[varname].SCALE)
        offset = np.float32(src[varname].OFFSET)
        data = data.where(data!=float(data.NO_DATA), np.nan) #.astype(np.float32)
        data = data/scale + offset
    if 'CODING' in src[varname].attrs.keys():
        slope = np.float32(src[varname].CODING[0])
        offset = np.float32(src[varname].CODING[1])
        data = data.where(data!=float(data.NOVALUE), np.nan) #.astype(np.float32)
        data = data/slope + offset

    return data

def read_spotvgt_geometry(dirname, 
                         chunks,
    #                     kind='vis', 
                         ):
    '''
    Read the geometry of a SpotVGT Level1 product.
    dirname : str, path to the directory containing the SpotVGT data.
    kind : str, type of data to read ('vis' or 'swir')
    chunks : int, size of the chunks to use for reading the data.
    '''

    ds = xr.Dataset()

#    if kind == 'vis':
##        band = 'RED'
#        band = ''
#    elif kind == 'swir':
#       band = 'SWIR'
#    else:
#        raise ValueError('kind must be either "vis" or "swir"')

    for angle in ['SZA','SAA','VZA', 'VAA']:
        filename = glob(dirname+'/*_{}*.hdf*'.format(angle))
        assert (len(filename)==1 or len(filename)==2)
#        filename = filename[0]
        for f in filename:
            if 'SWIR' in f:
                name = '{}_{}'.format(angle, f.split('_')[-2])
            else:
                name = angle
            ds[name] = read_spotvgt_variable(f, chunks).astype('float32') 

    return ds

def set_latlon(mapping, sizes, start=0):
    p = np.arange(sizes[1], dtype='float')
    l = np.arange(start, start+sizes[0], dtype='float')

    top_left_lon = float(mapping[3])
    top_left_lat = float(mapping[4])
    step_x = float(mapping[5])
    step_y = float(mapping[6])

    lon = top_left_lon + p*step_x
    lat = top_left_lat - l*step_y

    return lat, lon

def read_spotvgt(dirname,
                smac_dir,
                version,
                sensor,
                chunks
                 ):
    '''
    Read an SpotVGT Level1 product as an xarray
    '''


    # read geometries
    print("reading geometries...")
    angles = read_spotvgt_geometry(dirname, chunks) #, kind='vis')
#    angles_swir = read_ProbaV_geometry(dirname, chunks, kind='swir')

    # read cloud, lat and lon
#    filename = glob(dirname+'/*SM_SHD*.hdf')
    filename = glob(dirname+'/*CLOUD_PROBABILITY*.hdf*')
    assert(len(filename)==1)
    filename = filename[0]
#    cam = {'1':'LEFT', '2':'CENTER', '3':'RIGHT'}
    date = basename(filename).split('_')[2]
    date = '{}-{}-{}'.format(date[:4], date[4:6], date[6:8])
    hour = basename(filename).split('_')[3]
    hour = '{}:{}:{}'.format(hour[:2], hour[2:4], hour[4:6])
    dt = np.datetime64('{}T{}'.format(date, hour))
#    camera = cam[basename(filename).split('_')[4]]
    src = xr.open_dataset(filename, chunks=chunks)
#    cloud = src['SM_SHD']
#    cloud = (src['CLOUD_PROBABILITY']/src['CLOUD_PROBABILITY'].attrs['SCALE'] + src['CLOUD_PROBABILITY'].attrs['OFFSET']) > 0.1
    filename = glob(dirname+'/*SM*.hdf*')
    assert(len(filename)==1)
    filename = filename[0]
    sm = read_spotvgt_variable(filename, chunks, 'uint16')
    sm = sm.astype('uint16')
    cloud = (sm.data&1)==1

    map = src['CLOUD_PROBABILITY'].attrs['MAPPING']
#    map = cloud.attrs['MAPPING']
    lat_axis, lon_axis = set_latlon(map, cloud.shape)
    lon, lat = meshgrid(lon_axis, lat_axis)
    filtre = np.isnan(angles['SZA'].data)
    lon[filtre] = np.nan
    lat[filtre] = np.nan
    lon = lon.rechunk((chunks, chunks)).astype('float32')
    lat = lat.rechunk((chunks, chunks)).astype('float32')

    ds = xr.Dataset({'SZA':(['y','x'], angles['SZA'].data), 'SAA':(['y','x'], angles['SAA'].data), 'VZA': (['y','x'], angles['VZA'].data), 'VAA': (['y','x'], angles['VAA'].data), 'VAA_IR': (['y','x'], angles['VAA_SWIR'].data), 'VZA_IR': (['y','x'], angles['VZA_SWIR'].data), 
                         'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 
                           'clm':(['y','x'], cloud.astype('uint8'))}, 
                          coords={'x':(['x'], np.array(lon_axis, dtype='float32')), 'y':(['y'], np.array(lat_axis, dtype='float32'))})

    # read toa and bitmask
    bandnames = ['BLUE', 'RED', 'NIR', 'SWIR']
    filtre = (sm.data&8)==8
    prdnames = ['TOA','UNC_RANDOM','UNC_STRUCTURED','UNC_SYSTEMATIC']
    # Keep TOA / UNC lazy (dask) instead of materialising ~1.5 GB of numpy up front;
    # they are read per tile during processing, which keeps the resident base small.
    for p in prdnames:
        band_arrays = []
        for c in bandnames:
            filename = glob('{}/*{}_{}*.hdf*'.format(dirname, c, p))
            assert(len(filename)==1)
            filename = filename[0]
            toa = read_spotvgt_variable(filename, chunks)
            band_arrays.append(toa.data.astype('float32'))
        ds[p] = (['bands','y','x'], dask_stack(band_arrays, axis=0))

#    smnames = ['BLUE_SM_MAP', 'RED_SM_MAP', 'NIR_SM_MAP', 'SWIR_SM_MAP']
#    for i, c in enumerate(smnames):                                                                                                                                                                                  
#        filename = glob('{}/*{}*.hdf*'.format(dirname, c))
#        assert(len(filename)==1)
#        filename = filename[0]
#        sm = read_ProbaV_variable(filename, chunks) 
#        ds['SM_MAP'][i] = sm.data.astype('uint8')
    ds['SM_MAP'] = (['y','x'], sm.data.astype('uint16'))
    # dem
    filename = glob(dirname+'/*_DEM*.hdf*')
    if len(filename)==1:
        filename = filename[0]
        dem = read_spotvgt_variable(filename, chunks)
        dem = dem.where(filtre, np.nan)
        ds['elev'] = (['y','x'], dem.data.astype('float32'))
    
    # Try to load DELTADEM for elevation uncertainty
    filename = glob(dirname+'/*_DELTADEM*.hdf*')
    if len(filename)==1:
        filename = filename[0]
        dem = read_spotvgt_variable(filename, chunks)
        dem = dem.where(filtre, np.nan)  # Apply same mask as for elev
        ds['Delev'] = (['y','x'], dem.data.astype('float32'))
    else:
        # Fallback: use zeros if DELTADEM not available
        ds['Delev'] = (['y','x'], np.zeros_like(ds['elev'].values, dtype='float32')) 


    smac_coeffs_file = Path(smac_dir)/'VGT{}_smac_coeffs_v{version}.npy'.format(sensor[-1],version=version)
#    filename = glob(dirname+'/*_DELTADEM*.hdf*')
#    if len(filename)==1:
#        filename = filename[0]
#        dem = read_ProbaV_variable(filename, chunks)
#        dem = dem.where(~filtre, np.nan)
#        ds['Delev'] = (['y','x'], dem.data.astype('float32'))

    attributs = {"bands": bandnames, 'sensor':'spotvgt1', 'wavelengths': [.463, .655, .865, 1.600], 'smac_coeffs_file': smac_coeffs_file} #, 'mean-time': dt, 'mean-time-dec': date_to_float(dt)}
    ds = ds.assign_attrs(attributs)
    ds = ds.chunk({'y':chunks, 'x':chunks, 'bands':-1})
    ds['mean-time'] = dt
    ds['mean-time-dec'] = date_to_float(dt)

    return ds

