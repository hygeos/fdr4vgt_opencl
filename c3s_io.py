from netCDF4 import Dataset
import h5py
import xarray as xa
import numpy as np
#from smaccl import get_smac_coeffs, type_coeff_reduced, type_coeff
#from c3s_lib import SRF, date_to_float
from datetime import datetime
#import satpy as sp
#from satpy.readers.eps_l1b import read_records
import sys
#sys.path.insert(0,'./eoread')
#from eoread.msi import Level1_MSI
#from eoread.landsat8_oli import Level1_L8_OLI
from os.path import basename
from glob import glob

def read_probav(filename, chunkidx, chunksize):
    if (chunksize < 0):
        yslice = slice(None, None)
    else:
        ymin = chunkidx*chunksize
        ymax = ymin + chunksize
        yslice = slice(ymin, ymax)

    f = h5py.File(filename, 'r')
    ds = f[list(f.keys())[0]]
    map = ds.attrs['MAPPING']
#    for at in ds.attrs.keys():
#        print(at, ds.attrs[at])
    data = ds[yslice,:]
    if 'CODING' in ds.attrs.keys():
        slope = float(ds.attrs['CODING'][0])
        offset = float(ds.attrs['CODING'][1])
        data = data/slope + offset
    if 'NOVALUE' in ds.attrs.keys() and (data.dtype=='float32' or data.dtype=='float64'):
        novalues = (ds[yslice,:]==int(ds.attrs['NOVALUE'][0]))
        data[novalues] = np.NaN
    f.close()
    return data, map

def set_latlon(mapping, sizes, start):
    p = np.arange(sizes[1], dtype='float')
    l = np.arange(start, start+sizes[0], dtype='float')

    top_left_lon = float(mapping[3])
    top_left_lat = float(mapping[4])
    step_x = float(mapping[5])
    step_y = float(mapping[6])

    lon = top_left_lon + p*step_x
    lat = top_left_lat - l*step_y

    return lat, lon

def get_info_probav(dirname):
    filename = glob('{}/*BLUE_SM_MAP*.hdf'.format(dirname))[0]
    print(filename)
    sm, _ = read_probav(filename, 0, -1)

    return xa.Dataset(), sm.shape
    
def load_probav(dirname, chunkidx, chunksize, dirsmac, smac_version):
    # read sm map
#    filename = glob('{}/*SM_MAP*.hdf'.format(dirname))[0]
#    sm, mapping = read_probav(filename, chunkidx, chunksize)

    if (chunksize < 0):
        yslice = slice(None, None)
    else:
        ymin = chunkidx*chunksize
        ymax = ymin + chunksize
        yslice = slice(ymin, ymax)

    # read sm shd
    filename = glob('{}/*SM_SHD*.hdf'.format(dirname))[0]
    cloud, mapping = read_probav(filename, chunkidx, chunksize)
    lat_axis, lon_axis = set_latlon(mapping, cloud.shape, chunkidx*chunksize)
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    SIZE1 = len(lat_axis)
    SIZE2 = len(lon_axis)
#    cloud = (cloud==0)
    date = basename(filename).split('_')[2]
    date = '{}-{}-{}'.format(date[:4], date[4:6], date[6:8])
    hour = basename(filename).split('_')[3]
    hour = '{}:{}:{}'.format(hour[:2], hour[2:4], hour[4:6])
    dt = np.datetime64('{}T{}'.format(date, hour))
    camera = basename(filename).split('_')[4]

    # read geometries
    filename = glob('{}/*RED_SZA*.hdf'.format(dirname))[0]
    sza, _ = read_probav(filename, chunkidx, chunksize)
    filename = glob('{}/*RED_SAA*.hdf'.format(dirname))[0]
    saa, _ = read_probav(filename, chunkidx, chunksize)
    filename = glob('{}/*RED_VZA*.hdf'.format(dirname))[0]
    vza, _ = read_probav(filename, chunkidx, chunksize)
    filename = glob('{}/*RED_VAA*.hdf'.format(dirname))[0]
    vaa, _ = read_probav(filename, chunkidx, chunksize)
    filename = glob('{}/*SWIR_VZA*.hdf'.format(dirname))[0]
    vza_swir, _ = read_probav(filename, chunkidx, chunksize)
    filename = glob('{}/*SWIR_VAA*.hdf'.format(dirname))[0]
    vaa_swir, _ = read_probav(filename, chunkidx, chunksize)

    xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza), 'VAA': (['y','x'], vaa), 'VAA_SWIR': (['y','x'], vaa_swir), 'VZA_SWIR': (['y','x'], vza_swir), 
                         'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 
                           'clm':(['y','x'], cloud.astype('float32'))}, 
                          coords={'x':(['x'], np.array(lon_axis)), 'y':(['y'], np.array(lat_axis))})

    # read toa
    bandnames = {'band1':'BLUE_TOA', 'band2':'RED_TOA', 'band3':'NIR_TOA', 'band4':'SWIR_TOA'}
    for b, c in bandnames.items():
        filename = glob('{}/*{}*.hdf'.format(dirname, c))[0]
        toa, _ = read_probav(filename, chunkidx, chunksize)
        xdataset[b] = (['y','x'], toa)
    # read sm
    smnames = {'band1_sm':'BLUE_SM_MAP', 'band2_sm':'RED_SM_MAP', 'band3_sm':'NIR_SM_MAP', 'band4_sm':'SWIR_SM_MAP'}
    for b, c in smnames.items():
        filename = glob('{}/*{}*.hdf'.format(dirname, c))[0]
        sm, _ = read_probav(filename, chunkidx, chunksize)
        xdataset[b] = (['y','x'], sm)

    xdataset['mean-time'] = dt
    xdataset['mean-time-dec'] = date_to_float(xdataset['mean-time'].data)

    # define smac filenames
    cam = {'1':'LEFT', '2':'CENTER', '3':'RIGHT'}
    smacfile = '{}/PROBA-V_{}_smac_coeffs_v{}.npy'.format(dirsmac, cam[camera], smac_version)
    smacdata = np.load(smacfile)
    coeffs = np.zeros((len(bandnames.keys()), smacdata.shape[1]), dtype=type_coeff_reduced, order='C')
    for idx in range(len(bandnames)):
        for i2, d in enumerate(smacdata[idx]):
            coeffs[idx,i2] = d.tolist()[1:]

    return xdataset, SIZE1, SIZE2, list(bandnames.keys()), coeffs

def get_info_testcase_vito(fname):
    data = Dataset(fname)
    lat_axis = data['lat'][:]
    lon_axis = data['lon'][:]
    SIZE2 = len(lon_axis)
    SIZE1 = len(lat_axis)
    gl_size = (SIZE1, SIZE2)
    del data

    return xa.Dataset(), gl_size

def load_testcase_vito(fname, chunkidx, chunksize, dirsmac, smac_version, sensor):

    if (chunksize < 0):
        ymin=0
        ymax=-1
        yslice = slice(None,None)
    else:
        ymin = chunkidx*chunksize
        ymax  = ymin + chunksize
        yslice = slice(ymin,ymax)

    data = Dataset(fname)

    lat_axis = data['lat'][yslice]
    lon_axis = data['lon'][:]
    SIZE2 = len(lon_axis)
    SIZE1 = len(lat_axis)
    gl_size = (SIZE1, SIZE2)

    sza =      data['sza'][yslice,:]
    vza_vnir = data['vza_vnir'][yslice,:]
    vza_swir = data['vza_swir'][yslice,:]
    saa =      data['saa'][yslice,:]
    vaa_vnir = data['vaa_vnir'][yslice,:]
    vaa_swir = data['vaa_swir'][yslice,:]

    lon, lat = np.meshgrid(lon_axis, lat_axis)

    if sensor == 'AVHRR':
        cloud = np.logical_not((np.reshape(data['sm'][:], gl_size).astype('int')&15 == 8)).astype('int')
        sm = np.reshape(data['sm'], gl_size)
        hour = basename(fname).split('_')[-5]
        platform = basename(fname).split('_')[-2]
        smacfile = '{}/{}_{:02d}_smac_coeffs_v{}.npy'.format(dirsmac, platform[:4], int(platform[4:]), smac_version)
        if int(platform[4:]) < 15:
            bandnames = ['band1','band2']
        else:
            bandnames = ['band1','band2','band3']
    elif sensor == 'PROBAV':
        bandnames = ['band1','band2','band3','band4']
#        if 'sm' in data.variables:
#            varname = 'sm'
#            hour = basename(fname).split('_')[-5]
#        else:
        varname = 'SM'
#            hour = basename(fname).split('_')[-4]
        hour = basename(fname).split('_')[-5]
        cloud = np.logical_not((data[varname][yslice, :].astype('int')&15 == 8)).astype('int')
        sm = data[varname][yslice, :]
        smacfile = '{}/PROBA-V_smac_coeffs_v{}.npy'.format(dirsmac, smac_version)
        if 'camera' in data.ncattrs():
            camera = data.getncattr('camera')
            cam = {'1':'LEFT', '2':'CENTER', '3':'RIGHT'}
            smacfile = '{}/PROBA-V_{}_smac_coeffs_v{}.npy'.format(dirsmac, cam[camera], smac_version)

    elif sensor == 'VGT':
        bandnames = ['band1','band2','band3','band4']
        cloud = np.logical_not((np.reshape(data['sm'][:], gl_size).astype('int')&15 == 8)).astype('int')
        sm = np.reshape(data['sm'], gl_size)
        hour = str(datetime.strptime(data.time_coverage_start, '%Y/%m/%d %H:%M:%S') + (datetime.strptime(data.time_coverage_end, '%Y/%m/%d %H:%M:%S') - datetime.strptime(data.time_coverage_start, '%Y/%m/%d %H:%M:%S'))/2).split()[-1].replace(':','')
        smacfile = '{}/VGT{}_smac_coeffs_v{}.npy'.format(dirsmac, data.sensor.split('-')[-1], smac_version)

    date = basename(fname).split('_')[2]
    date = "{}-{}-{}".format(date[:4], date[4:6], date[6:8])
    hour = "{}:{}:{}".format(hour[:2], hour[2:4], hour[4:6])
    dt = np.datetime64('{}T{}'.format(date,hour))

    # test fichier exemple vito
    slope = 2000. 
    slope_azimuth = 2./3.
    slope_zenith = 2.
    sza =      np.array(sza)/slope_zenith
    vza_vnir = np.array(vza_vnir)/slope_zenith
    vza_swir = np.array(vza_swir)/slope_zenith
    saa =      np.array(saa)/slope_azimuth
    vaa_vnir = np.array(vaa_vnir)/slope_azimuth
    vaa_swir = np.array(vaa_swir)/slope_azimuth
    # fin test

    xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza_vnir), 'VAA': (['y','x'], vaa_vnir), 'VAA_SWIR': (['y','x'], vaa_swir), 'VZA_SWIR': (['y','x'], vza_swir), 
                           'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 
                           'clm':(['y','x'], cloud.astype('float32')), 'sm':(['y','x'], sm)}, 
                           coords={'x':(['x'], np.array(lon_axis)), 'y':(['y'], np.array(lat_axis))})


    for b in bandnames:
        xdataset[b] = (['y','x'], np.reshape(data[b][yslice, :]/slope, gl_size))
        berr = '{}_err'.format(b)
        err = data[berr][yslice, :]
        err[(err==-1)] = 0
        xdataset[berr] = (['y','x'], np.reshape(err*data[b][yslice, :]*.01, gl_size))
#        xdataset[berr] = (['y','x'], np.reshape(data[berr][yslice, :]*data[b][yslice, :]*.01, gl_size))

    xdataset['mean-time'] = dt
    xdataset['mean-time-dec'] = date_to_float(xdataset['mean-time'].data)

    smacdata = np.load(smacfile)
    coeffs = np.zeros((len(bandnames), smacdata.shape[1]), dtype=type_coeff_reduced, order='C')
    for idx in range(len(bandnames)):
        for i2, d in enumerate(smacdata[idx]):
            coeffs[idx,i2] = d.tolist()[1:]

    return xdataset, SIZE1, SIZE2, bandnames, coeffs, gl_size

def load_avhrr(filename, smacfile, chunkidx, chunksize, bands):
    if (chunksize < 0):
        ymin=0
        ymax=-1
        yslice = slice(None, None)
    else:
        ymin = chunkidx*chunksize
        ymax = ymin + chunksize
        yslce = slice(ymin,ymax)

    sc = sp.Scene([filename], reader='avhrr_l1b_eps')
    dataset_names = ['satellite_azimuth_angle', 'satellite_zenith_angle', 'solar_azimuth_angle', 'solar_zenith_angle', 'latitude', 'longitude']
    sc.load(dataset_names)
    date_mean = sc.start_time + (sc.end_time - sc.start_time)/2

    lat = sc['latitude'].values[yslice,:].astype('float32')
    lon = sc['longitude'].values[yslice,:].astype('float32')
    bitmask = read_records(filename)
    cloud = bitmask[0][('mdr',2)]
    cloud = cloud['CLOUD_INFORMATION'][yslice,:]
    vza = sc['satellite_zenith_angle'].values[yslice,:]
    vaa = sc['satellite_azimuth_angle'].values[yslice,:]
    sza = sc['solar_zenith_angle'].values[yslice,:]
    saa = sc['solar_azimuth_angle'].values[yslice,:]

    xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza), 'VAA': (['y','x'], vaa), 
                           'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 'clm':(['y','x'], cloud)})

    sc.load(bands)

    for band in bands:
        xdataset[band] = (['y','x'], sc[band].values[yslice,:]/100.)

    coeff_smac = get_smac_coeffs(smacfile['avhrr'], np.arange(4))

    gl_size = sc['latitude'].shape

    dt = np.datetime64(date_mean)
    xdataset['mean-time'] = dt
    xdataset['mean-time-dec'] = date_to_float(dt)

    return xdataset, coeff_smac, gl_size

def load_olci_slstr(fname, smacfile, chunkidx, chunksize, platform='SENTINEL3_1', bands_olci=None, bands_slstr=None):

    _,_,_,wvl_central_olci,_,_,_  = SRF(platform+'_OLCI')
    pyt_,_,_,wvl_central_slstr,_,_,_ = SRF(platform+'_SLSTR')
    wav = {'olci': wvl_central_olci, 'slstr': wvl_central_slstr}  

    if (chunksize < 0):
        ymin=0
        ymax=-1
        yslice = slice(None, None)
    else:
        ymin = chunkidx*chunksize
        ymax = ymin + chunksize
        yslice= slice(ymin,ymax)
    pfile = Dataset(fname)
    date = pfile.getncattr('start_date')

    lat = pfile['latitude'][yslice,:].astype('float32')
    lon = pfile['longitude'][yslice,:].astype('float32')
    cloud = pfile['cloud_an'][yslice,:]
    quality_fg = pfile['quality_flags'][yslice,:]
    pxl_classif_fg = pfile['pixel_classif_flags'][yslice,:]

    vza = pfile['OZA'][yslice,:]
    vaa = pfile['OAA'][yslice,:]
    sza = pfile['SZA'][yslice,:]
    saa = pfile['SAA'][yslice,:]

    mus = np.cos(sza*np.pi/180.)
    if bands_olci is None:
        olci_idx = [2,3,4,5,6,7,8,9,10,11,12,16,17,18,21]
    else:
        olci_idx = bands_olci

    if 'lat_intern' in pfile.variables.keys():
        lat_axis = pfile['lat_intern']
        lon_axis = pfile['lon_intern']
    else:
        lat_axis = pfile['lat']
        lon_axis = pfile['lon']

    xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza), 'VAA': (['y','x'], vaa), 
                           'cloud_an': (['y','x'], cloud), 'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 
                           'quality_flags':(['y','x'], quality_fg.astype('int32')), 'pixel_classif_flags':(['y','x'], pxl_classif_fg), 
                           'clm':(['y','x'], cloud.astype('float32'))}, 
                           coords={'x':(['x'], lon_axis[:]), 'y':(['y'], lat_axis[yslice])})

    tab_band_internal = []
    central_wvl = []
    sensor= []
    # bands olci
    for idx in olci_idx:
        rad_band = 'Oa{:02d}_radiance'.format(idx)
        tab_band_internal.append(rad_band)
        central_wvl.append(wav['olci'][idx-1])
        sensor.append('olci')
        ltoa = pfile[rad_band][yslice,:]
        f0_band = 'solar_flux_band_{}'.format(idx)
        f0 = pfile[f0_band][yslice, :]
        rtoa = (np.pi*ltoa)/(mus*f0)
        xdataset[rad_band] = (['y','x'], rtoa)

    coeff_olci = get_smac_coeffs(smacfile['olci'], np.array(olci_idx)-1)

    # band slstr
    if bands_slstr is None:
#        slstr_idx = [1,2,3,4,5,6]
        slstr_idx = [1,2,3,5,6]
    else:
        slstr_idx = bands_slstr

    for idx in slstr_idx:
        s_band = 'S{}_radiance_an'.format(idx)
        if idx!=4:
            tab_band_internal.append(s_band)
            central_wvl.append(wav['slstr'][idx-1])
            sensor.append('slstr')
        ltoa = pfile[s_band][yslice,:]
        f0 = pfile[s_band].getncattr('solar_irradiance')[0]
        rtoa = (np.pi*ltoa)/(mus*f0)
        xdataset[s_band] = (['y','x'], rtoa)

    SIZE1, SIZE2 = xdataset[tab_band_internal[0]].shape
    if 4 in slstr_idx : slstr_idx.remove(4)
    coeff_slstr = get_smac_coeffs(smacfile['slstr'], np.array(slstr_idx)-1)

    coeff_smac = np.concatenate([coeff_olci, coeff_slstr])

    day =   date[:2]
    year =  date[7:11]
    month = date[3:6]
    hour =  date[12:14]
    minu =  date[15:17]
    sec =   date[18:20]
    strmonths = np.array(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
    month = np.where(month==strmonths)[0][0]
    dt = np.datetime64(year + '-' + '{:02d}'.format(month+1) + '-' + day + 'T' + hour + ':' + minu + ':' + sec)
    xdataset['mean-time'] = dt
    xdataset['mean-time-dec'] = date_to_float(xdataset['mean-time'].data)

    filtre = (np.isnan(xdataset['clm'].values))
    xdataset['clm'].values[filtre] = 0

    gl_size = pfile['latitude'][:].shape

    pfile.close()

    return xdataset, SIZE1, SIZE2, tab_band_internal, central_wvl, sensor, coeff_smac, gl_size


def load_msi(fname, smacfile, chunkidx, chunksize, 
        platform='S2A', bands_msi=None, remove_blank=True, split=True, resolution='10'):

    _,_,_,wvl_central_msi,_,_,_  = SRF(platform+'_MSI')
    wav = {'msi': wvl_central_msi} 

    pfile = Level1_MSI(fname, split=split, resolution=resolution)
    date  = pfile.attrs['datetime']
    bnames=[]
    for ds in pfile:
        if 'Rtoa' in ds:
            bnames.append(ds)
    if remove_blank:
        good = np.where(pfile[bnames[0]] != 0.)
        gslicex = slice(good[1][0], good[1][-1])
        gslicey = slice(good[0][0], good[0][-1])
        pfile   = pfile.sel(columns=gslicex, rows=gslicey)

    if (chunksize < 0):
        yslice = slice(None, None)
    else:
        ymin  = chunkidx*chunksize
        ymax  = ymin + chunksize
        yslice= slice(ymin,ymax)

    lat = pfile['latitude'][yslice,:].astype('float32')
    lon = pfile['longitude'][yslice,:].astype('float32')
    cloud = np.zeros_like(lat)

    vza = pfile['vza'][yslice,:]
    vaa = pfile['vaa'][yslice,:]
    sza = pfile['sza'][yslice,:]
    saa = pfile['saa'][yslice,:]
    if bands_msi is None:
        msi_idx = list(np.arange(13)+1)
    else:
        msi_idx = bands_msi

    xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza), 'VAA': (['y','x'], vaa), 
                           'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 'clm':(['y','x'], cloud)})

    tab_band_internal = []
    central_wvl = []
    sensor = []

    for idx in msi_idx:
        rad_band = bnames[idx-1] 
        tab_band_internal.append(rad_band)
        central_wvl.append(wav['msi'][idx-1])
        sensor.append('msi')
        rtoa = pfile[rad_band][yslice,:]
        xdataset[rad_band] = (['y','x'], rtoa)

    coeff_smac = get_smac_coeffs(smacfile['msi'], np.array(msi_idx)-1)

    dt = np.datetime64(date)
    xdataset['mean-time'] = dt
    xdataset['mean-time-dec'] = date_to_float(dt)

    gl_size = pfile['latitude'].shape

    pfile.close()
    SIZE1, SIZE2 = xdataset[tab_band_internal[0]].shape

    return xdataset, SIZE1, SIZE2, tab_band_internal, central_wvl, sensor, coeff_smac, gl_size


def load_oli(fname, smacfile, chunkidx, chunksize, 
        platform='LANDSAT8', bands_oli=None, remove_blank=True, split=True):

    _,_,_,wvl_central_oli,_,_,_  = SRF(platform+'_OLI')
    wav = {'oli': wvl_central_oli} 

    pfile = Level1_L8_OLI(fname, split=split, l8_angles='./l8_angles/l8_angles')
    date  = pfile.attrs['datetime']
    gl_size = (int(pfile['totalheight'].data), int(pfile['totalwidth'].data))
    bnames=[]
    for ds in pfile:
        if 'Rtoa' in ds:
            bnames.append(ds)
    if remove_blank:
        good = np.where(pfile[bnames[0]] > 0)
        if len(good[0]) == 0:
            return None, None, None, None, None, None, None, gl_size 
        gslicex = slice(good[1][0], good[1][-1])
        gslicey = slice(good[0][0], good[0][-1])
        pfile   = pfile.sel(columns=gslicex, rows=gslicey)

    if (chunksize < 0):
        yslice=slice(None)
    else:
        ymin  = chunkidx*chunksize
        ymax  = ymin + chunksize
        yslice= slice(ymin,ymax)

    lat = pfile['latitude'][yslice,:].astype('float32')
    lon = pfile['longitude'][yslice,:].astype('float32')
    cloud = np.zeros_like(lat)

    vza = pfile['vza'][yslice,:]
    vaa = pfile['vaa'][yslice,:]
    sza = pfile['sza'][yslice,:]
    saa = pfile['saa'][yslice,:]
    if bands_oli is None:
        oli_idx = list(np.arange(7)+1)
    else:
        oli_idx = bands_oli

    xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza), 'VAA': (['y','x'], vaa), 
                           'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 'clm':(['y','x'], cloud)})

    tab_band_internal = []
    central_wvl = []
    sensor = []

    for idx in oli_idx:
        rad_band = bnames[idx-1] 
        tab_band_internal.append(rad_band)
        central_wvl.append(wav['oli'][idx-1])
        sensor.append('oli')
        rtoa = pfile[rad_band][yslice,:]
        xdataset[rad_band] = (['y','x'], rtoa)

    coeff_smac = get_smac_coeffs(smacfile['oli'], np.array(oli_idx)-1)

    dt = np.datetime64(date)
    xdataset['mean-time'] = dt
    xdataset['mean-time-dec'] = date_to_float(dt)

    gl_size = pfile['latitude'].shape

    pfile.close()
    SIZE1, SIZE2 = xdataset[tab_band_internal[0]].shape

    return xdataset, SIZE1, SIZE2, tab_band_internal, central_wvl, sensor, coeff_smac, gl_size

def load_viirs(filename: str, smacfile, chunkidx, chunksize, bands):
    """
    This function is based on load_avhrr
    
    - filename: path str pointing to a VITO's VIIRS L1 data file
    """
    
    if (chunksize < 0):
        yslice = slice(None, None)
    else:
        ymin = chunkidx * chunksize
        ymax = ymin + chunksize
        yslice= slice(ymin,ymax)
    
    # Open data file
    # remove unecessary time dimension / coordinates
    ds = xa.open_mfdataset(filename).squeeze('time').drop_vars('time')
    
    # process reflectance for each band (modify inplace)
    for band in bands: 
        ds[band] = ds[band] / np.cos(np.deg2rad(ds.solar_zenith))
    
    # process 2D lat and lon TODO: verify geometry
    glon2D, glat2D = np.meshgrid(ds['lon'].values, ds['lat'].values)
    glat2D = glat2D.astype('float32')
    glon2D = glon2D.astype('float32')
    
    # get region for lat and lon
#    lat2D = glat2D[yslice, :]
#    lon2D = glon2D[yslice, :]
    lat2D = glat2D[yslice, :]
    lon2D = glon2D[yslice, :]
    
    #--------------------------------------------------------
    # /!\ WARNING /!\ TODO review this strategy:
    #--------------------------------------------------------
    cloud = ds['Integer_Cloud_Mask']
    cloud = cloud != 3          # clouds = everything not 'confidently clear'
    cloud = cloud[yslice,:]  # get region
    # 0 = cloudy, 
    # 1 = probably cloudy, 
    # 2 = probably clear, 
    # 3 = confident clear, -1 = no result)
    #--------------------------------------------------------
    
    vza = ds['sensor_zenith' ].values[yslice, :]
    vaa = ds['sensor_azimuth'].values[yslice, :]
    sza = ds['solar_zenith'  ].values[yslice, :]
    saa = ds['solar_azimuth' ].values[yslice, :]
    
    xdataset = xa.Dataset({
        'SZA': (['y','x'], sza), 
        'SAA': (['y','x'], saa), 
        'VZA': (['y','x'], vza), 
        'VAA': (['y','x'], vaa), 
        'lat': (['y','x'], lat2D.data), 
        'lon': (['y','x'], lon2D[:,:]), 
        'clm': (['y','x'], cloud.data)
        })
    
    for band in bands:
        # /!\ Unsure about the division (would suggest percentage data, but current data E [0, 1])
#        xdataset[band] = (['y','x'], ds[band].values[yslice,:]/100.)
        xdataset[band] = (['y','x'], ds[band].values[yslice,:])
    
    # /!\ really unsure about the second argument TODO: verify
    coeff_smac = get_smac_coeffs(smacfile['viirs'], np.arange(11))
    
    gl_size = lat2D.shape # global shape ?
    
    # compute mean datetime 
    # str -> datetime obj
    dt_s = datetime.strptime(ds.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%S.%fZ')
    dt_e = datetime.strptime(ds.attrs['time_coverage_end'], '%Y-%m-%dT%H:%M:%S.%fZ')
    # compute mean of start and end
    mean_time = np.datetime64(dt_s + (dt_e - dt_s) / 2)
    xdataset['mean-time'] = mean_time
    xdataset['mean-time-dec'] = date_to_float(mean_time)
    
    return xdataset, coeff_smac, gl_size

def create_nc(filename, gl_size, attrs, version):
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
    out.production_centre = 'vito'
    out.version = version

    width = gl_size[1]
    height = gl_size[0]

    out.createDimension('height', height)
    out.createDimension('width', width)

    return out


def save_nc(out, data, rsurf, Drsurf, version, dataset_names, chunkidx, chunksize, Tg, Drtoa=None, Dtaup=None, ancillary=None, sensor=None, save_error=True): 
    '''
    Save outputs
    Inputs:
        out : a NetCDF4 Dataset    
        data: the xarray containing Level 1 data 
        rsurf:  the numpy array containing the TOC reflectance
        Drsurf:  the numpy array containing the uncertainty on TOC reflectance
        version  : string containing version
        dataset_names : the list of string containing the names of the bands in level 1 file
        chunkidx: the number of the chunk to be saved
        chunksize: size of the chunk
   '''
    if (chunksize < 0):
        yslice=slice(None)
    else:
        ymin  = chunkidx*chunksize
        ymax  = ymin + chunksize
        yslice= slice(ymin,ymax)

    # test if some chunks have already been saved in the output file
    create  = not ('Lat' in out.variables)

    ##############################
    # test ecriture toa 
    if not(Drtoa is None):
        for iband, band in enumerate(dataset_names):
#        if create : sds = out.createVariable('TOA_{}'.format(band), 'f', ('height','width'), complevel=9)
#        else: sds = out['TOA_{}'.format(band)]
#        try:
#            sds[yslice, :] = data[band].data
#        except:
#            print("erreur {}".format(band))
            if create: sds = out.createVariable('Jacobien_TOA_{}'.format(band), 'f', ('height','width'), complevel=9)
            else: sds = out['Jacobien_TOA_{}'.format(band)]
            try:
                sds[yslice, :] = Drtoa[iband].data
            except:
                print("erreur jacobien TOA {}".format(band))
            if create: sds = out.createVariable('Jacobien_AOT_{}'.format(band), 'f', ('height','width'), complevel=9)
            else: sds = out['Jacobien_AOT_{}'.format(band)]
            try:
                sds[yslice, :] = Dtaup[iband].data
            except:
                print("erreur jacobien AOT {}".format(band))
####################### test ecriture sm source
#    for band in ['band1_sm_old','band2_sm_old','band3_sm_old','band4_sm_old']:
#        if create: sds = out.createVariable(band, 'u4', ('height','width'))
#        else: sds = out[band]
#        try: 
#            sds[yslice,:] = data[band].data
#        except:
#            print("erreur {}".format(band))
######################## fin test

    slope_toc = 20000
    slope_angle = 100
    for idx in range(rsurf.shape[0]):
        band = dataset_names[idx].replace('_radiance','').replace('Rtoa_','')
#        if create : sds = out.createVariable('TOC_{}'.format(band), 'f', ('height','width'), complevel=9)
        if create : sds = out.createVariable('TOC_{}'.format(band), 'i', ('height','width'), complevel=9, fill_value=-999)
        else: sds = out['TOC_{}'.format(band)]
        tmp = rsurf[idx]*slope_toc
        filtre = np.isnan(rsurf[idx])
        tmp[filtre] = -999
        sds[yslice, :] = tmp
        sds.Long_name = 'Top of Canopy Reflectance'
        sds.Unit = 'None'
        sds.slope = 1/slope_toc
        if save_error:
            band = 'TOC_{} error'.format(band)
#            if create: sds = out.createVariable(band, 'f', ('height','width'), complevel=9)
            if create: sds = out.createVariable(band, 'i', ('height','width'), complevel=9, fill_value=-999)
            else: sds = out[band]
            tmp = Drsurf[idx]*slope_toc
            filtre = np.isnan(Drsurf[idx])
            tmp[filtre] = -999
            sds[yslice,:] = tmp
            sds.Long_name = 'Uncertainty Top of Canopy Reflectance'
            sds.Unit = 'None'
            sds.slope = 1/slope_toc
    if create: sds = out.createVariable('Lat', 'i', ('height','width'), complevel=9, fill_value=-999)
    else: sds = out['Lat']
    tmp = data['lat'].data*slope_angle
    filtre = np.isnan(data['lat'].data)
    tmp[filtre] = -999
    sds[yslice,:] = tmp
    sds.Unit = 'Degree'
    sds.slope = 1/slope_angle
    if create: sds = out.createVariable('Lon', 'i', ('height', 'width'), complevel=9, fill_value=-999)
    else: sds = out['Lon']
    tmp = data['lon'].data*slope_angle
    filtre = np.isnan(data['lon'].data)
    tmp[filtre] = -999
    sds[yslice,:] = tmp
    sds.Unit = 'Degree'
    sds.slope = 1/slope_angle
    
    if create: sds = out.createVariable('SZA', 'i', ('height', 'width'), complevel=9, fill_value=-999)
    else: sds = out['SZA']
    tmp = data['SZA'].data*slope_angle
    filtre = np.isnan(data['SZA'].data)
    tmp[filtre] = -999
    sds[yslice,:] = tmp
    sds.Unit = 'Degree'
    sds.slope = 1/slope_angle
    if create : sds = out.createVariable('SAA', 'i', ('height', 'width'), complevel=9, fill_value=-999)
    else: sds = out['SAA']
    tmp = data['SAA'].data*slope_angle
    filtre = np.isnan(data['SAA'].data)
    tmp[filtre] = -999
    sds[yslice,:] = tmp
    sds.Unit = 'Degree'
    sds.slope = 1/slope_angle
    if sensor=='PROBAV':
        if create: sds = out.createVariable('VZA_VNIR', 'i', ('height', 'width'), complevel=9, fill_value=-999)
        else: sds = out['VZA_VNIR']
        tmp = data['VZA'].data*slope_angle
        filtre = np.isnan(data['VZA'].data)
        tmp[filtre] = -999
        sds[yslice,:] = tmp
        sds.Unit = 'Degree'
        sds.slope = 1/slope_angle
        if create: sds = out.createVariable('VAA_VNIR', 'i', ('height', 'width'), complevel=9, fill_value=-999)
        else: sds = out['VAA_VNIR']
        tmp = data['VAA'].data*slope_angle
        filtre = np.isnan(data['VAA'].data)
        tmp[filtre] = -999
        sds[yslice,:] = tmp
        sds.Unit = 'Degree'
        sds.slope = 1/slope_angle

        if create: sds = out.createVariable('VZA_SWIR', 'i', ('height', 'width'), complevel=9, fill_value=-999)
        else: sds = out['VZA_SWIR']
        tmp = data['VZA_SWIR'].data*slope_angle
        filtre = np.isnan(data['VZA_SWIR'].data)
        tmp[filtre] = -999
        sds[yslice,:] = tmp
        sds.Unit = 'Degree'
        sds.slope = 1/slope_angle
        if create: sds = out.createVariable('VAA_SWIR', 'i', ('height', 'width'), complevel=9, fill_value=-999)
        else: sds = out['VAA_SWIR']
        tmp = data['VAA_SWIR'].data*slope_angle
        filtre = np.isnan(data['VAA_SWIR'].data)
        tmp[filtre] = -999
        sds[yslice,:] = tmp
        sds.Unit = 'Degree'
        sds.slope = 1/slope_angle
    elif sensor=='AVHRR':
        if create: sds = out.createVariable('cloud', 'i', ('height','width'), complevel=9)
        else: sds = out['cloud']
        sds[yslice,:] = data['clm']
        if create: sds = out.createVariable('VZA', 'i', ('height', 'width'), complevel=9, fill_value=-999)
        else: sds = out['VZA']
        tmp = data['VZA'].data*slope_angle
        filtre = np.isnan(data['VZA'].data)
        tmp[filtre] = -999
        sds[yslice,:] = tmp
        sds.Unit = 'Degree'
        sds.slope = 1/slope_angle
        if create: sds = out.createVariable('VAA', 'i', ('height', 'width'), complevel=9, fill_value=-999)
        else: sds = out['VAA']
        tmp = data['VAA'].data*slope_angle
        filtre = np.isnan(data['VAA'].data)
        tmp[filtre] = -999
        sds[yslice,:] = tmp
        sds.Unit = 'Degree'
        sds.slope = 1/slope_angle
    else:
        #### TEST DEBUG ####
        for iband, band in enumerate(dataset_names):
            if create: sds = out.createVariable(band, 'f', ('height', 'width'), complevel=9, fill_value=-999)
            else: sds = out[band]
            sds[yslice,:] = data[band].data
            if create: sds = out.createVariable('Tg_{}'.format(band), 'f', ('height', 'width'), complevel=9, fill_value=-999)
            else: sds = out[band]
            sds[yslice,:] = Tg[iband].data

        if create: sds = out.createVariable('VZA', 'i', ('height', 'width'), complevel=9, fill_value=-999)
        else: sds = out['VZA']
        tmp = data['VZA'].data*slope_angle
        filtre = np.isnan(data['VZA'].data)
        tmp[filtre] = -999
        sds[yslice,:] = tmp
        sds.Unit = 'Degree'
        sds.slope = 1/slope_angle
        if create: sds = out.createVariable('VAA', 'i', ('height', 'width'), complevel=9, fill_value=-999)
        else: sds = out['VAA']
        tmp = data['VAA'].data*slope_angle
        filtre = np.isnan(data['VAA'].data)
        tmp[filtre] = -999
        sds[yslice,:] = tmp
        sds.Unit = 'Degree'
        sds.slope = 1/slope_angle

    if ancillary is not None:
        if create: sds = out.createVariable('uo3', 'f', ('height', 'width'), complevel=9)
        else: sds = out['uo3']
        sds[yslice,:] = ancillary[0]
        if create: sds = out.createVariable('uh2o', 'f', ('height', 'width'), complevel=9)
        else: sds = out['uh2o']
        sds[yslice,:] = ancillary[1]
        if create: sds = out.createVariable('aot550', 'f', ('height', 'width'), complevel=9)
        else: sds = out['aot550']
        sds[yslice,:] = ancillary[2]
        if create: sds = out.createVariable('iaer', 'u4', ('height', 'width'), complevel=9)
        else: sds = out['iaer']
        sds[yslice,:] = ancillary[3]
        if create: sds = out.createVariable('pressure', 'f', ('height', 'width'), complevel=9)
        else: sds = out['pressure']
        sds[yslice,:] = ancillary[4]
        if create: sds = out.createVariable('alt', 'f', ('height', 'width'), complevel=9)
        else: sds = out['alt']
        sds[yslice,:] = ancillary[5]

    # test mask
#    if create: sds = out.createVariable('sm', 'u4', ('height', 'width'), complevel=9)
#    else: sds = out['sm']
#    sds[yslice,:] = data['sm']
    # end test
    for d in data.variables:
        if 'sm' in d:
#        if 'sm' in d and not('old' in d):
            if create: sds = out.createVariable(d, 'u4', ('height', 'width'), complevel=9)
            else: sds = out[d]    
            sds[yslice,:] = data[d].data

    create2  = 'cloud_an' in data.variables
    if create2 : 
        if create: sds = out.createVariable('cloud_an', 'f', ('height', 'width'), complevel=9)
        else: sds = out['cloud_an']
        sds[yslice,:] = data['cloud_an']
        if create: sds = out.createVariable('quality_flags', 'u4', ('height', 'width'), complevel=9)
        else: sds = out['quality_flags']
        sds[yslice,:] = data['quality_flags']#.data.astype('uint32')
        if create: sds = out.createVariable('pixel_classif_flags', 'u2', ('height', 'width'), complevel=9)
        else: sds = out['pixel_classif_flags']
        sds[yslice,:] = data['pixel_classif_flags']#.data.astype('uint16')
        if create: sds = out.createVariable('AC_process_flag', 'u1', ('height','width'), complevel=9)
        else: sds = out['AC_process_flag']
        sds[yslice,:] = data['ac_process_flag']#.data.astype('uint8')

    if create: sds = out.createVariable('ac_flag', 'u4', ('height','width'), complevel=9)
    else: sds = out['ac_flag']
    sds[yslice,:] = data['ac_flag']

#class S3_slstr_olci:
#    def __init__(self, fname, smaccoef, chunksize):
#        self.filename = fname
#        self.smaccoef = smaccoef
#        self.chunksize = chunksize
#        self.olci_idx = [2,3,4,5,6,7,8,9,10,11,12,16,17,18,21]
#        self.open()
#
#    def open(self):
#        self.pfile = Dataset(self.filename)
#
#    def read(self, ichunk):
#        print("lecture chunk ",ichunk)
#        if (self.chunksize < 0):
#            ymin=0
#            ymax=-1
#        else:
#            ymin = ichunk*self.chunksize
#            ymax = ymin + self.chunksize
#
#        pfile = Dataset(self.filename)
#        print(pfile)
#        date = pfile.getncattr('start_date')
#        lat = pfile['latitude'][yslice,:].astype('float32')
#        lon = pfile['longitude'][yslice,:].astype('float32')
#        cloud = pfile['cloud_an'][yslice,:]
#        quality_fg = pfile['quality_flags'][yslice,:]
#        #pxl_classif_fg = pfile['pixel_classif_flags'][yslice,:]
#
#        print(lat)
#        vza = pfile['OZA'][yslice,:]
#        vaa = pfile['OAA'][yslice,:]
#        sza = pfile['SZA'][yslice,:]
#        saa = pfile['SAA'][yslice,:]
#
#        mus = np.cos(sza*np.pi/180.)
#
#        if 'lat_intern' in pfile.variables.keys():
#            lat_axis = pfile['lat_intern']
#            lon_axis = pfile['lon_intern']
#        else:
#            lat_axis = pfile['lat']
#            lon_axis = pfile['lon']
#
#        xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza), 'VAA': (['y','x'], vaa), 
#                           'cloud_an': (['y','x'], cloud), 'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 
#                           'quality_flags':(['y','x'], quality_fg.astype('int32')), 
#                           #'pixel_classif_flags':(['y','x'], pxl_classif_fg), 
#                           'clm':(['y','x'], cloud.astype('float32'))}, 
#                           coords={'x':(['x'], lon_axis[:]), 'y':(['y'], lat_axis[yslice])})
#
#        tab_band_internal = []
#        central_wvl = []
#        sensor= []
#
#
#        platform = 'S3B'
#        _,_,_,wvl_central_olci,_,_,_  = SRF(platform+'_OLCI')
#        print(wvl_central_olci)
#        print(np.mean(pfile['lambda0_band_2'][:,:]))
#        print(np.mean(pfile['lambda0_band_3'][:,:]))
        # bands olci
#        for idx in self.olci_idx:
#            rad_band = 'Oa{:02d}_radiance'.format(idx)
#            tab_band_internal.append(rad_band)
#            central_wvl.append(wav['olci'][idx-1])
#            sensor.append('olci')
#            ltoa = pfile[rad_band][yslice,:]
#            f0_band = 'solar_flux_band_{}'.format(idx)
#            f0 = pfile[f0_band][yslice, :]
#            rtoa = (np.pi*ltoa)/(mus*f0)
#            xdataset[rad_band] = (['y','x'], rtoa)

#def open(fname, smaccoef, plateform):
#    if 'S3' in plateform:
#        obj = S3_slstr_olci(fname, smaccoef)
#
#    return obj
        
if __name__=='__main__':
#    dirname = '/rfs/proj/REPRO_PROBAV/HDF5/'
#    chunkidx = 1
    chunksize = 100
#    smac_version ='3.0'
    dirsmac = '/rfs/proj/C3S/SMAC_COEFFS/'
#    data, s1, s2, bandnames, coeffs = load_probav(dirname, chunkidx, chunksize, dirsmac, smac_version)
#    print(data)
#    print(s1, s2)
#    print(bandnames)
#    trash, gl_size = get_info_probav(dirname)
#    print(gl_size)
#    filename = '/rfs/user/bruno/C3S/input/cgl_SEN3-SYN-BS-RP_201807030950_AE-Carpentras_S3A_v0.0.rc2.nc'
#    level1 = S3_slstr_olci(filename, dirsmac, chunksize)
#    level1.read(0)
    filename = '/rfs/proj/CCI+/vito_inputs/input_samples/Metop-A-AVHRR/AVHR_xxx_1B_M02_20080920171902Z_20080920172202Z_N_O_20080920180521Z'
    smacfile = {'avhrr':'/rfs/proj/CCI+/SMAC_COEFFS/METOP_1_AVHRR_smac_coeffs_v3.0.npy'}
    xs, coeffs,sizes = load_avhrr(filename, smacfile, 0, 100,['1'])
    print(xs)
    print(coeffs)
    print(xs['mean-time'])
    print(xs['clm'].dtype)
