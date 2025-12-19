import xarray as xr
from glob import glob
import numpy as np
from os.path import basename
from dask.array import meshgrid

def date_to_float(d, epoch=np.datetime64('1980-01-01T00:00:00.000000000')):
    '''
        transform the date into a duration in minutes since epoch
        by default from '1980-01-01T00:00:00.000000000'
    '''
    
    return (d - epoch).astype(np.float64)/1.0e9/60.

def Level1_probav(dirname,
                chunks=None,
                ):
    '''
    Read an ProbaV Level1 product as an xarray.Dataset
    '''

    ds = read_ProbaV(dirname,
                     chunks
                    )
    return ds

def read_ProbaV_variable(filename, chunks):
    '''
    Read a variable from a ProbaV Level1 product.
    '''
    src = xr.open_dataset(filename, chunks=chunks)
    varname = list(src.variables.keys())
    assert(len(varname)==1)
    varname = varname[0]

    data = src[varname]
    if 'CODING' in src[varname].attrs.keys():                                                                                                                                                                           
        slope = np.float32(src[varname].CODING[0])
        offset = np.float32(src[varname].CODING[1])
        data = data.where(data!=float(data.NOVALUE), np.nan) #.astype(np.float32)
        data = data/slope + offset

    return data

def read_ProbaV_geometry(dirname, 
                         chunks,
                         kind='vis', 
                         ):
    '''
    Read the geometry of a ProbaV Level1 product.
    dirname : str, path to the directory containing the ProbaV data.
    kind : str, type of data to read ('vis' or 'swir')
    chunks : int, size of the chunks to use for reading the data.
    '''

    ds = xr.Dataset()

    if kind == 'vis':
        band = 'RED'
    elif kind == 'swir':
       band = 'SWIR'
    else:
        raise ValueError('kind must be either "vis" or "swir"')

    for angle in ['SZA','SAA','VZA', 'VAA']:
        filename = glob(dirname+'/*{}_{}.hdf'.format(band, angle))
        assert (len(filename)==1)
        filename = filename[0]
        ds[angle] = read_ProbaV_variable(filename, chunks).astype('float32') 

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

def read_ProbaV(dirname,
                 chunks,
                 ):
    '''
    Read an ProbaV Level1 product as an xarray
    '''


    # read geometries
    angles_vis = read_ProbaV_geometry(dirname, chunks, kind='vis')
    angles_swir = read_ProbaV_geometry(dirname, chunks, kind='swir')

    # read cloud, lat and lon
    filename = glob(dirname+'/*SM_SHD*.hdf')
    assert(len(filename)==1)
    filename = filename[0]
    cam = {'1':'LEFT', '2':'CENTER', '3':'RIGHT'}
    date = basename(filename).split('_')[2]
    date = '{}-{}-{}'.format(date[:4], date[4:6], date[6:8])
    hour = basename(filename).split('_')[3]
    hour = '{}:{}:{}'.format(hour[:2], hour[2:4], hour[4:6])
    dt = np.datetime64('{}T{}'.format(date, hour))
    camera = cam[basename(filename).split('_')[4]]
    src = xr.open_dataset(filename, chunks=chunks)
    cloud = src['SM_SHD']
    map = cloud.attrs['MAPPING']
    lat_axis, lon_axis = set_latlon(map, cloud.shape)
    lon, lat = meshgrid(lon_axis, lat_axis)
    lon = lon.rechunk((chunks, chunks)).astype('float32')
    lat = lat.rechunk((chunks, chunks)).astype('float32')

    ds = xr.Dataset({'SZA':(['y','x'], angles_vis['SZA'].data), 'SAA':(['y','x'], angles_vis['SAA'].data), 'VZA': (['y','x'], angles_vis['VZA'].data), 'VAA': (['y','x'], angles_vis['VAA'].data), 'VAA_IR': (['y','x'], angles_swir['VAA'].data), 'VZA_IR': (['y','x'], angles_swir['VZA'].data), 
                         'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 
                           'clm':(['y','x'], cloud.data.astype('float32'))}, 
                          coords={'x':(['x'], np.array(lon_axis, dtype='float32')), 'y':(['y'], np.array(lat_axis, dtype='float32'))})

    # read toa and bitmask
    bandnames = ['BLUE', 'RED', 'NIR', 'SWIR']
    prdnames = ['TOA','UNC_RANDOM','UNC_STRUCTURED','UNC_SYSTEMATIC']
    for p in prdnames:
        ds[p] = (['bands','y','x'], np.zeros((len(bandnames), ds.dims['y'], ds.dims['x']), dtype='float32'))
    ds['SM_MAP'] = (['bands','y','x'], np.zeros((len(bandnames), ds.dims['y'], ds.dims['x']), dtype='uint8'))
    for i, c in enumerate(bandnames):
        for p in prdnames:
            filename = glob('{}/*{}_{}*.hdf'.format(dirname, c, p))
            assert(len(filename)==1)
            filename = filename[0]
            toa = read_ProbaV_variable(filename, chunks) 
           # name = '{}_{}'.format(b, p.lower())
            ds[p][i] = toa.data.astype('float32')

    smnames = ['BLUE_SM_MAP', 'RED_SM_MAP', 'NIR_SM_MAP', 'SWIR_SM_MAP']
    for i, c in enumerate(smnames):                                                                                                                                                                                  
        filename = glob('{}/*{}*.hdf'.format(dirname, c))
        assert(len(filename)==1)
        filename = filename[0]
        sm = read_ProbaV_variable(filename, chunks) 
        ds['SM_MAP'][i] = sm.data.astype('uint8')

    ds['mean-time'] = dt
    ds['mean-time-dec'] = date_to_float(dt)
    attributs = {'CAMERA':camera, "bands": bandnames, 'sensor':'Proba-V'} #, 'mean-time': dt, 'mean-time-dec': date_to_float(dt)}
    ds = ds.assign_attrs(attributs)
    ds = ds.chunk({'y':chunks, 'x':chunks, 'bands':-1})

    return ds

