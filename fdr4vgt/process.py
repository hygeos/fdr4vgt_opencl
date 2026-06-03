#from probav_vito import Level1_probav
from in_out import Level1, save_nc, create_nc, save_nc_batch, load_brdf
import configparser
import xarray as xa
from core import interpolate
import numpy as np
from pathlib import Path
#import harp
#import core
#from harp.providers.NASA import MERRA2
#from tempfile import TemporaryDirectory
import dask.array as da
#from lib.jsmac_lib_dev import read_smac_coefficients, smac_neo
#import jax.numpy as jnp
from time import time
#import datetime
import gc
import sys
import dask
from funcs import calculate_monthly_aerosol, config, get_slope_err, open_monthly_aerosol, read_smac_coefficients, build_flag, setup_logger, memory_tracker, get_iaer
from CF_from_json import apply_cf_attributes_from_json, validate_cf_compliance
import onnxruntime as ort

from smaccl.ISmaccl import ISmaccl
from smaccl.smaccl import type_coeff, get_smac_coeffs

# Ajouter au début du fichier, après les imports
#setup_logger()

def Ps(z,p0,T, g=9.801, R=287.058, lam=-0.006):
    T1 = np.log(R*T) - np.log(-R*lam*z+R*T)

    return p0*np.exp(-g/(R*lam)*T1)

def closest_model(X, Xb):
    '''
    return the closest model number compared to reference basis
    it is a distance minimization in a 5-dimensional space
    '''

    return np.sum((X[:, np.newaxis, :]-Xb[:, :, np.newaxis])**2, axis=0).argmin(axis=0)

def closest_models(X, Xb):
    '''
    return the 10 closest model numbers compared to reference basis
    it is a distance minimization in a 5-dimensional space
    '''

    return np.sum((X[:, np.newaxis, :]-Xb[:, :, np.newaxis])**2, axis=0).argsort(axis=0)[:11, :]

#def get_iaero(frac_aer_model, lat, frac, config):
#
#    iaero = None
#    xb = []
#    xm = []
#    # OPTIONAEROFIXE
#    if 'aero_nmod' in config.keys():
#        nb_pixel = len(lat)
#        iaero = np.zeros(nb_pixel)
#        iaero[:] = config['aero_nmod']
#    else:
#        sizes = frac[0].shape
#        for k,key in enumerate(frac_aer_model.keys()):
##            frac = merra[match[key]+'_FRAC'].data[good].astype(np.float32, order='C')
#            xb.append(frac_aer_model[key])
##            xm.append(frac[k][good])
#            xm.append(frac[k].ravel())
#        xb = np.stack(xb, axis=0)
#        xm = np.stack(xm, axis=0)
#        iaero = closest_model(xm, xb)
#        iaero = iaero.reshape(sizes).astype(np.int32)
#
#    return iaero

#@memory_tracker
def readConfig(configfile):
    cf = configparser.ConfigParser()
    cf.read(configfile)
    config = {}
    for k,v  in cf.items('Coefficients'):
       config[k] = float(v) 
    for k,v  in cf.items('Sizes'):
       config[k] = int(v) 
    for k,v in cf.items('Paths'):
        config[k] = v
    for k,v in cf.items('Sensor'):
        config[k] = v
    for k,v in cf.items('Output'):
        config[k] = eval(v)

    return config

#@memory_tracker
def read_dem(dem, lat , lon, chunks=None):
    dem = xa.open_dataset(dem) #, chunks=chunks)
    elev = dem['elev']
    Delev = dem['Delev']
    elev_interp = interpolate.interp(elev, lat=interpolate.Linear(lat), lon=interpolate.Linear(lon)).astype(np.float32)
    Delev_interp = interpolate.interp(Delev, lat=interpolate.Linear(lat), lon=interpolate.Linear(lon)).astype(np.float32)
#    dem = xa.Dataset({'elev': elev_interp.T, 'Delev': Delev_interp.T, 'lat': lat, 'lon': lon})
    dem = xa.Dataset({'elev': elev_interp, 'Delev': Delev_interp, 'lat': lat, 'lon': lon})
    dem = dem.chunk({'y':chunks, 'x':chunks}) 

    return dem

#@memory_tracker
def read_merra(merra_aer, merra_p2, lat_sat, lon_sat, time, chunks):
    aer = xa.open_dataset(merra_aer)
    p2 = xa.open_dataset(merra_p2)
    assert((time >= np.min(aer['time'])) and (time <= np.max(aer['time'])))
    assert((time >= np.min(p2['time'])) and (time <= np.max(p2['time'])))

    tau  =    interpolate.interp(aer['TOTEXTTAU'], lat=interpolate.Linear(lat_sat), lon=interpolate.Linear(lon_sat), time=interpolate.Linear(time)).astype(np.float32)
    uh2o    = interpolate.interp(p2['TQV'], lat=interpolate.Linear(lat_sat), lon=interpolate.Linear(lon_sat), time=interpolate.Linear(time)).astype(np.float32)
    uo3     = interpolate.interp(p2['TO3'], lat=interpolate.Linear(lat_sat), lon=interpolate.Linear(lon_sat), time=interpolate.Linear(time)).astype(np.float32)
    p0      = interpolate.interp(p2['SLP'], lat=interpolate.Linear(lat_sat), lon=interpolate.Linear(lon_sat), time=interpolate.Linear(time)).astype(np.float32)
    t10m    = interpolate.interp(p2['T10M'], lat=interpolate.Linear(lat_sat), lon=interpolate.Linear(lon_sat), time=interpolate.Linear(time)).astype(np.float32)
    bc_frac = interpolate.interp(aer['BCEXTTAU'] , lat=interpolate.Linear(lat_sat), lon=interpolate.Linear(lon_sat), time=interpolate.Linear(time)).astype(np.float32)
    du_frac = interpolate.interp(aer['DUEXTTAU'] , lat=interpolate.Linear(lat_sat), lon=interpolate.Linear(lon_sat), time=interpolate.Linear(time)).astype(np.float32)
    oc_frac = interpolate.interp(aer['OCEXTTAU'] , lat=interpolate.Linear(lat_sat), lon=interpolate.Linear(lon_sat), time=interpolate.Linear(time)).astype(np.float32)
    ss_frac = interpolate.interp(aer['SSEXTTAU'] , lat=interpolate.Linear(lat_sat), lon=interpolate.Linear(lon_sat), time=interpolate.Linear(time)).astype(np.float32)
    su_frac = interpolate.interp(aer['SUEXTTAU'] , lat=interpolate.Linear(lat_sat), lon=interpolate.Linear(lon_sat), time=interpolate.Linear(time)).astype(np.float32)

    del aer
    del p2

    merra = xa.Dataset({'TOTEXTTAU':tau, 'TQV':uh2o*1e-1, 'TO3':uo3*1e-3, 'SLP':p0*1e-2, 'T10M':t10m, 'BC_FRAC':bc_frac/tau, 'DU_FRAC':du_frac/tau, 'SS_FRAC':ss_frac/tau, 'SU_FRAC':su_frac/tau, "OC_FRAC":oc_frac/tau, 
                       'lat': lat_sat, 'lon': lon_sat}
                       )
    merra = merra.chunk({'y':chunks, 'x':chunks})
    
    return merra

def read_smaccoeffs_probav(smaccoef_dir, camera, version, bandnames):
    smacfile = '{}/PROBA-V_{}_smac_coeffs_v{}.npy'.format(smaccoef_dir, camera, version)
    smacdata = np.load(smacfile)
    type_coeff = [(key,dtype[0]) for key, dtype in smacdata.dtype.fields.items()]
    type_coeff = type_coeff[1:]
#    coeffs = np.zeros((len(bandnames.keys()), smacdata.shape[1]), dtype=type_coeff, order='C')
    coeffs = np.zeros((len(bandnames.keys()), smacdata.shape[1], len(type_coeff)), dtype=np.float32, order='C')
    for idx in range(len(bandnames)):
        for i2, d in enumerate(smacdata[idx]):
            coeffs[idx,i2] = d.tolist()[1:]

    return coeffs

def pre_aer_models(faer):
    '''
        Read MERRA2 aerosols components fraction of the aerosol models
    '''
    match = {'sulf':'SU', 'dust':'DU', 'oc':'OC', 'ssalt':'SS', 'bc':'BC'}
    f = open(faer, 'r')
    frac_aer_model = {}
    for key in match.keys():
        f.readline ()
        line = f.readline ()
        frac_aer_model[key] =  np.array(line.split()).astype(float)
    f.close()
    
    return frac_aer_model

def calc_error(data):
    bands = data.bands
    size1, size2, = data['TOA'].shape[1:3]
    err =  np.zeros((len(bands), size1, size2), dtype=np.float32)
    for i in range(len(bands)):
#        tmp = b.split('_')[0]
        err[i] = np.sqrt(data ['UNC_RANDOM'][i]**2 + data['UNC_STRUCTURED'][i]**2 + data['UNC_SYSTEMATIC'][i]**2)
#    data = data.drop_vars(['UNC_RANDOM', 'UNC_STRUCTURED', 'UNC_SYSTEMATIC'])   

    data["ERROR"] = (['bands','y','x'], err)
    return data

def array_to_jax(a):
    return jnp.array(a.astype(np.float32).flatten())

def array_to_jax_batched(a, batch_size=1000):
    """Convert array to JAX array in batches to reduce memory usage"""
    if len(a.shape) == 2:
        a_flat = a.astype(np.float32).flatten()
    else:
        a_flat = []
        for i in range(0, a.shape[0]):
            a_flat.append(a[i].astype(np.float32).flatten())
        a_flat = jnp.array(a_flat)


    for i in range(0, len(a_flat), batch_size):
        if len(a.shape) == 2:
            batch = jnp.array(a_flat[i:i+batch_size])
        else:
            batch = jnp.array((a_flat[:,i:i+batch_size]))

    return batch

#@memory_tracker
def compute_urtoc(Jtoa, Utoa, Jh2o, Uh2o,Jo3, Uo3, Jps, Ups, Jt550, Ut550, Urtoc_ens, Urtoc_rtm_slope, Urtoc_rtm_brdf, Urtoc_rtm_fit, u2_0):

    unc_toa = Jtoa*Utoa
    sum = unc_toa ** 2

    unc_h2o = Jh2o*Uh2o
    sum += unc_h2o ** 2

    unc_o3 = Jo3*Uo3
    sum += unc_o3 ** 2

    unc_ps = Jps*Ups
    sum += unc_ps ** 2

    unc_aot = Jt550*Ut550
    sum += unc_aot ** 2

    dummy = Urtoc_ens
    sum += dummy ** 2

#    dummy = Urtoc_rtm_slope
#    sum += dummy ** 2
    sum += Urtoc_rtm_brdf ** 2
    sum += Urtoc_rtm_fit ** 2

    sum += u2_0 ** 2

    return np.sqrt(sum), (np.abs(unc_h2o), np.abs(unc_o3), np.abs(unc_ps), np.abs(unc_aot))

#@memory_tracker
def compute_rtm_fit(onnx_file, pression, iaermodel, raa, raa_ir, rtoc, sza, tauaer, vza, vza_ir, wvl):
    session = ort.InferenceSession(onnx_file)
    results = np.zeros(rtoc.shape, dtype=np.float32) + np.nan
    for i, w in enumerate(wvl):
        filtre = ~np.isnan(pression.values)
        pre = pression.values[filtre].ravel()
        iaer = iaermodel.values[filtre].ravel()
        if i < 2:
            phi = raa[filtre].ravel()
            thetav = vza[filtre].ravel()
        else:
            phi = raa_ir[filtre].ravel()
            thetav = vza_ir[filtre].ravel()
        rsurf = rtoc[i][filtre].ravel()
        thetas = sza[filtre].ravel()
        aod = tauaer[filtre].ravel()
        wave = np.full(pre.shape, w).ravel()
        x = np.array([pre, iaer, phi, rsurf, thetas, aod, thetav, wave]).T.astype(np.float32)
        outputs = session.run(None, {'input': x})
        results[i][filtre] = np.array(outputs[0]).ravel()
    return np.abs(results)


#@memory_tracker
def run(S, data_batch, iaero):
    return S.run(data_batch, iaero)

#@memory_tracker
#def process_batched(data, frac_aer_model, ca_, ca_ind, config_, list_aerosol, batch_size=512):
def process_batched(data, frac_aer_model, ca_, ca_ind, config_, batch_size=512):
    dtype = np.dtype([(k, 'f4') for k in ca_ind.keys()])
    ca2_ = np.ones((ca_.shape[0], ca_.shape[1]), dtype=dtype, order='C')
    for idx, name in enumerate(ca_ind.keys()):
        ca2_[name] = ca_[:,:,idx]

    s1, s2 = data['SZA'].shape
    S = ISmaccl(config_, frac_aer_model, ca2_, platform='CPU', XBLOCK=batch_size, XGRID=batch_size, breakpoint=False)

    out = create_nc(config_['output'], (s1,s2), data.attrs['bands'], [], 1.0)
    latitude = data.lat
    longitude = data.lon
    date_time = data["mean-time"]

#    iaer_month = calculate_monthly_aerosol(date_time, latitude, longitude) #,  list_aerosol)
#    data['iaer_month'] = (['naero','y','x'], iaer_month.astype(np.uint32))

    gc.collect()

    for iband, i in enumerate(range(0, s1, batch_size)):
        # bandeau + 1 ligne au-dessus et en-dessous pour éviter les effets de bord
        y_min = max(0,i-1)
        y_max = min(s1, i + batch_size +1)
#        data_batch = data.isel(y=slice(y_min, y_max))
        for jband, j in enumerate(range(0, s2, batch_size)):
            x_min = max(0,j-1)
            x_max = min(s2, j + batch_size +1)
            data_batch = data.isel(x=slice(x_min, x_max),y=slice(y_min, y_max))
            latitude =  data_batch.lat
            longitude = data_batch.lon
            date_time = data_batch["mean-time"]
    #        iaer_best = get_iaero(frac_aer_model, latitude.values, [data_batch['SU_FRAC'].values, data_batch['DU_FRAC'].values, data_batch['OC_FRAC'].values, data_batch['SS_FRAC'].values, data_batch['BC_FRAC'].values], config_)
    #        iaer_original = iaer_best.copy()
            iaer_best = get_iaer(data_batch)
            iaer_month = calculate_monthly_aerosol(date_time, latitude, longitude) #,  list_aerosol)
            iaero  = np.zeros((config_['nmodels']+1, iaer_best.shape[0], iaer_best.shape[1]), dtype=np.int32)
            iaero[0] = iaer_best
    #        iaero[1:] = data_batch['iaer_month']
            iaero[1:] = iaer_month
    #        ds_out = S.run(data_batch, iaero)
            pression = data_batch['SLP'] * config_['k_p0']
    #        slope_err = get_slope_err(data_batch, data_batch['TOTEXTTAU'].values, pression.values, ca_, ca_ind, iaer_month[0])
            slope_err = get_slope_err(data_batch, data_batch['TOTEXTTAU'].values, pression.values, ca_, ca_ind, iaero[1], chunksize=batch_size)
            slope_err = np.maximum(slope_err, -2)
            slope_err = np.minimum(slope_err, 2)

    #        if config_['debug']:
#            ds_out['iaero'] = (('y','x'), iaer_best)

            # suppression des lignes supplémentaires
            if y_min != 0: y_min_2 = 1 
            else: y_min_2 = 0
            if y_max != s1 : y_max_2 = -1 
            else: y_max_2 = s1
            if x_min != 0: x_min = 1 
            else: x_min = 0
            if x_max != s2 : x_max = -1
            else: x_max = s2

#            data_batch = data_batch.isel(y=slice(y_min, y_max))
#            ds_out = ds_out.isel(y=slice(y_min, y_max))
#            slope_err = slope_err[:, y_min:y_max, :]
            data_batch = data_batch.isel(x=slice(x_min, x_max), y=slice(y_min_2, y_max_2))
            iaero = iaero[:, y_min_2:y_max_2, x_min:x_max]
            ds_out = run(S, data_batch, iaero)
#            ds_out = ds_out.isel(x=slice(x_min, x_max), y=slice(y_min_2, y_max_2))
            slope_err = slope_err[:, y_min_2:y_max_2, x_min:x_max]
            ds_out['iaero'] = (('y','x'), iaer_best[y_min_2:y_max_2, x_min:x_max])

            # gradiant AOD
            ygrad, xgrad = da.gradient(data_batch['TOTEXTTAU'].data)
            aod_grad = np.sqrt(ygrad**2 + xgrad**2)
            ds_out = ds_out.assign({
                'aod_grad': (('y','x'), aod_grad)
            })
            del ygrad
            del xgrad

            urtoc_rtm_slope = ds_out['rTOC'].data*(slope_err - 1)
            ds_out['slope_err'] = (('bands','y', 'x'), slope_err)
            ds_out['urtoc_terrain'] = (('bands','y', 'x'), urtoc_rtm_slope)
            flag = build_flag(data_batch, ds_out, config)
            ds_out['flag'] = flag

            urtoc_rtm_brdf = ds_out['rTOC_0'].values - ds_out['rTOC'].values
            ds_out['UrTOC_rtm_brdf'] = (('bands', 'y', 'x'), urtoc_rtm_brdf)
            raa = data_batch['SAA'].values - data_batch['VAA'].values
            raa = raa % 360
            f = (raa > 180)
            raa[f] = 360-raa[f]
            raa_ir = data_batch['SAA'].values - data_batch['VAA_IR'].values
            raa_ir = raa_ir % 360
            f = (raa_ir > 180)
            raa_ir[f] = 360-raa_ir[f]
            urtoc_rtm_fit = compute_rtm_fit(config_['rmt_fit_file'], data_batch['SLP']*config_['k_p0'], ds_out['iaero'], raa, raa_ir, ds_out['rTOC'].values, data_batch['SZA'].values, data_batch['TOTEXTTAU'].values, data_batch['VZA'].values, data_batch['VZA_IR'].values, data_batch.wavelengths)
            ds_out['UrTOC_rtm_fit'] = (('bands', 'y', 'x'), urtoc_rtm_fit)
        
            urtoc, unc = compute_urtoc(ds_out['Jrtoa'].data, ds_out['Drtoa'].data, ds_out['Juh2o'].data, ds_out['Duh2o'].data, ds_out['Juo3'].data, ds_out['Duo3'].data, ds_out['Jpre'].data, ds_out['Dpre'].data, ds_out['Jtau550'].data, ds_out['Dtaup'].data, ds_out['UrTOC_ens'].data, urtoc_rtm_slope, urtoc_rtm_brdf, urtoc_rtm_fit, config_['u2_0'])
            ds_out['UrTOC'] = (('bands','y','x'), urtoc)
            ds_out['unc_h2o'] = (('bands','y','x'), unc[0])
            ds_out['unc_o3']  = (('bands','y','x'), unc[1])
            ds_out['unc_ps']  = (('bands','y','x'), unc[2])
            ds_out['unc_aot'] = (('bands','y','x'), unc[3])

            ds_out = apply_cf_attributes_from_json(ds_out, config_['cf_json_path'])
            data_batch = apply_cf_attributes_from_json(data_batch, config_['cf_json_path'])
#            save_nc_batch(out, data_batch, ds_out, iband, jband, batch_size, config_['jacobian']) #, config_['debug'])
            save_nc_batch(config_['output'], data_batch, ds_out, iband, jband, batch_size, config_['jacobian']) #, config_['debug'])

            gc.collect()


    return None, None

#@memory_tracker
def process_block(data, frac_aer_model, ca_, ca_ind, config, batch_size=128):
    """
    Utilise dask.array.map_blocks pour paralléliser le calcul TOC et jacobians.
    """
    S = ISmaccl(config, frac_aer_model, ca_, platform='CPU', breakpoint=False)

    ds_out = S.run_block(data, batch_size=512)
    return ds_out
    NB = len(data.bands)
    SIZE1 = data['SZA'].shape[0]
    SIZE2 = data['SZA'].shape[1]
    Naero = 11
    zero4d = xa.DataArray(np.zeros((NB, SIZE1, SIZE2, Naero), dtype=np.float32), dims=('bands','Y','X','aermodel')).chunk({'bands':-1, 'Y':batch_size, 'X':batch_size, 'aermodel':-1})
#    zero4d = xa.DataArray(da.zeros((NB, SIZE1, SIZE2, Naero), dtype=np.float32, order='C'), dims=('bands','Y','X','aermodel')).chunk({'bands':-1, 'Y':batch_size, 'X':batch_size, 'aermodel':-1})
    print(zero4d)
    zero3d = xa.DataArray(da.zeros((NB, SIZE1, SIZE2), dtype=np.float32), dims=('bands','Y','X')).chunk({'bands':-1, 'Y':batch_size, 'X':batch_size})
    print(zero3d)
    
    list_rsurf = ['rsurf_{}'.format(i) for i in range(Naero)]
    ds_in = xa.Dataset(
        {
            x: data[x] for x in ['lat','lon','TOA','SZA','VZA','SAA','VAA','VZA_IR','VAA_IR','TQV','TO3','TOTEXTTAU','SLP','elev','Delev','T10M','SU_FRAC','SS_FRAC','DU_FRAC','OC_FRAC','BC_FRAC','ERROR']
        }
    )
    ds_in = ds_in.assign_attrs(data.attrs)
    ds_out = xa.map_blocks(
        S.run,
        ds_in, 
        template=xa.Dataset(
            {
                'rsurf': zero4d,
#                'Juh2o': zero3d,
#                'Juo3':  zero3d,
#                'Jrtoa': zero3d,
#                'Jpre':  zero3d,
#                'Drsurf':zero3d,
            }
        )
    )
    return ds_out

#@memory_tracker
def save(data, fileout):
    save_nc(data, fileout)

#@memory_tracker
def process(configfile):

    dask.config.set(scheduler='synchronous') 
#    chunks = 256
    config_ = readConfig(configfile)
    config.set('amip_path', config_['amip_path'])
    coef_path = Path(config_['smaccoef_dir']) / f"{config_['sensor']}_smac_coeffs_v3.0.npy"
    config.set('aerosol_model_fraction', config_['faer'])
    config.set('dem_path', config_['dem'])
    config.set('kept_data_path', config_['kept_data_path'])
    config.set('aotmax', config_['aotmax'])
    config.set('tocmin', config_['tocmin'])
    config.set('tocmax', config_['tocmax'])
    config.set('szamax', config_['szamax'])
    config.set('aod_max_grad', config_['aod_max_grad'])
    dirname = config_['input']
    merra_aer = config_['merraaero']
    merra_p2 = config_['merraptwo']
    aer_file = config_['faer']
    chunks = config_['chunks_size']
    sensor = config_['sensor']
    smac_dir = config_['smaccoef_dir']
    version = config_['smaccoef_version']

    # read ProbaV data
    data = Level1(dirname, sensor, smac_dir, version, chunks=chunks)
    data = calc_error(data)
    filtre = np.isnan(data['SZA'])

    dir_brdf = config_['brdf_dir']
    date = data['mean-time']
    nbands = len(data.bands)
    k1p, k2p = load_brdf(dir_brdf, date, data['lat'], data['lon'], nbands, chunks)
#    k1p.compute()
    data['k1p'] = k1p
    data['k2p'] = k2p
    del k1p
    del k2p

    # read DEM
#    demfile = config_['dem']
#    dem = read_dem(demfile, data['lat'], data['lon'], chunks=chunks)
##    data['den'] = dem['elev'].where(~filtre, other=np.nan)
#    data['Delev'] = dem['Delev'].where(~filtre, other=np.nan)
#    del dem
#    gc.collect()

    # read MERRA2
    merra = read_merra(merra_aer, merra_p2, data['lat'], data['lon'], data['mean-time'], chunks)
    for p in ['TOTEXTTAU','TQV','TO3','SLP','T10M','BC_FRAC','DU_FRAC','SS_FRAC','SU_FRAC','OC_FRAC']:
        tmp = merra[p].where(~filtre, other=np.nan)
        data[p] = tmp

    del merra

    # read smac coeffs
#    bandnames = {'band1':'BLUE_TOA', 'band2':'RED_TOA', 'band3':'NIR_TOA', 'band4':'SWIR_TOA'}
    frac_aer_model = pre_aer_models(aer_file)

#    smac_coeffs_file = Path('/archive/proj/C3S')/'{sensor}_{camera}_smac_coeffs_v{version}.npy'.format(sensor=config_['sensor'], camera=data.CAMERA, version=config_['smaccoef_version'])
#    smac_coeffs_file = Path(smaccoef_dir)/'{sensor}_{camera}_smac_coeffs_v{version}.npy'.format(sensor=config_['sensor'], camera=data.CAMERA, version=config_['smaccoef_version'])
    smac_coeffs_file = data.attrs['smac_coeffs_file']
    ca_, ca_ind = read_smac_coefficients(smac_coeffs_file)

    data = data.assign_attrs({'jac_name' : ['Juh2o','Juo3','Jrtoa','Jpre','Drsurf']})
    data = data.chunk({'y':chunks, 'x':chunks, 'bands':-1})

    print("#### date ####", date.values, type(date))
    toc,  jac = process_batched(data, frac_aer_model, ca_, ca_ind, config_, batch_size=chunks)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
#        print("Usage: python process.py <dirname> <demfile> <merra_aer> <merra_p2> <smac_dir> <aer_file> <fileout> <configfile>")
        print("Usage: python process.py <configfile>")
        exit(1)
    print()
#    dirname = sys.argv[1]
#    demfile = sys.argv[2]
#    merra_aer = sys.argv[3]
#    merra_p2 = sys.argv[4]
#    smac_dir = sys.argv[5]
#    aer_file = sys.argv[6]
#    fileout = sys.argv[7]
    configfile = sys.argv[1]
#    process(dirname, demfile, merra_aer, merra_p2, smac_dir, aer_file, fileout, configfile)
    process(configfile)