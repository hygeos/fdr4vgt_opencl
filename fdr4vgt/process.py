#from probav_vito import Level1_probav
from in_out import Level1, create_nc, save_nc_batch, load_brdf
from netCDF4 import Dataset
import configparser
import xarray as xa
from core import interpolate
from core.tools import xrcrop
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
import os
import shutil
import tempfile
import atexit
import warnings
warnings.filterwarnings("ignore")
import dask
from funcs import calculate_monthly_aerosol, config, get_slope_err, preload_monthly_aerosol, read_smac_coefficients, build_flag, get_iaer, shift_lon_to_360, regular_interp, compute_delta_elevation_from_elev, load_koppen_geiger_zone_multipliers, sample_koppen_geiger_multiplier
from CF_from_json import apply_cf_attributes_from_json
import onnxruntime as ort
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import psutil

try:
    import rasterio
except Exception:
    rasterio = None

from smaccl.ISmaccl import ISmaccl

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
    if cf.has_section('Options'):
        for k, v in cf.items('Options'):
            low = v.strip().lower()
            if low in {'true', 'false'}:
                config[k] = (low == 'true')
            else:
                try:
                    config[k] = int(v)
                except ValueError:
                    try:
                        config[k] = float(v)
                    except ValueError:
                        config[k] = v

    return config

def open_merra_global(merra_aer, merra_p2, lon_sat_full, time):
    '''
    Open the (small, global) MERRA2 aerosol + surface files once, keep them in
    memory, and apply the longitude-shift decision based on the FULL satellite
    grid so it stays consistent across every tile. Returns (aer, p2, shift_sat).
    '''
    aer = xa.open_dataset(merra_aer)[['TOTEXTTAU','BCEXTTAU','DUEXTTAU','OCEXTTAU','SSEXTTAU','SUEXTTAU']].compute()
    p2 = xa.open_dataset(merra_p2)[['TQV','TO3','SLP','T10M']].compute()
    assert((time >= np.min(aer['time'])) and (time <= np.max(aer['time'])))
    assert((time >= np.min(p2['time'])) and (time <= np.max(p2['time'])))
    aer = shift_lon_to_360(aer, lon_sat_full)
    p2 = shift_lon_to_360(p2, lon_sat_full)

    lon_vals = np.asarray(lon_sat_full)
    lon_max = np.nanmax(lon_vals)
    lon_min = np.nanmin(lon_vals)
    shift_sat = bool(lon_max > 0 and lon_min < 0 and (lon_max - lon_min) > 300)
    return aer, p2, shift_sat


def interp_merra_tile(aer, p2, lat_sat, lon_sat, time, shift_sat):
    '''
    Interpolate the MERRA2 fields to a single tile's lat/lon grid. This is
    numerically identical to interpolating the whole grid and slicing, but keeps
    peak memory bounded to the tile size (the full-grid version materialised
    ~2.8 GB of numpy arrays). ``shift_sat`` is the global decision from
    open_merra_global so the longitude convention is consistent across tiles.
    '''
    if shift_sat:
        # Preserve the (y, x) dims/coords of the satellite lon grid; only shift values.
        lon_sat = lon_sat.copy(deep=True)
        vals = lon_sat.values
        vals[vals < 0] += 360
    # else: use lon_sat as-is (DataArray with y,x dims, matching lat_sat)

    lat_v = np.asarray(lat_sat.values)
    lon_v = np.asarray(lon_sat.values)
    tv = np.datetime64(time.values) if hasattr(time, 'values') else np.datetime64(time)

    tau     = regular_interp(aer['TOTEXTTAU'], lat_v, lon_v, tv)
    uh2o    = regular_interp(p2['TQV'],  lat_v, lon_v, tv)
    uo3     = regular_interp(p2['TO3'],  lat_v, lon_v, tv)
    p0      = regular_interp(p2['SLP'],  lat_v, lon_v, tv)
    t10m    = regular_interp(p2['T10M'], lat_v, lon_v, tv)
    bc_frac = regular_interp(aer['BCEXTTAU'], lat_v, lon_v, tv)
    du_frac = regular_interp(aer['DUEXTTAU'], lat_v, lon_v, tv)
    oc_frac = regular_interp(aer['OCEXTTAU'], lat_v, lon_v, tv)
    ss_frac = regular_interp(aer['SSEXTTAU'], lat_v, lon_v, tv)
    su_frac = regular_interp(aer['SUEXTTAU'], lat_v, lon_v, tv)

    merra = xa.Dataset({'TOTEXTTAU':(('y','x'),tau), 'TQV':(('y','x'),uh2o*1e-1),
                        'TO3':(('y','x'),uo3*1e-3), 'SLP':(('y','x'),p0*1e-2), 'T10M':(('y','x'),t10m),
                        'BC_FRAC':(('y','x'),bc_frac/tau), 'DU_FRAC':(('y','x'),du_frac/tau),
                        'SS_FRAC':(('y','x'),ss_frac/tau), 'SU_FRAC':(('y','x'),su_frac/tau),
                        "OC_FRAC":(('y','x'),oc_frac/tau)})
    return merra


def interp_merra_native_aod_grad(aer, lat_sat, lon_sat, time, shift_sat,
                                 include_temporal=False,
                                 temporal_scale_steps=1.0):
    """
    Compute AOD gradient magnitude on the native regular MERRA2 grid and
    interpolate it to the target satellite grid.

    This avoids taking gradients on already-interpolated satellite-grid AOD,
    which can smooth or distort sharp MERRA2 transitions.
    """
    if shift_sat:
        lon_sat = lon_sat.copy(deep=True)
        vals = lon_sat.values
        vals[vals < 0] += 360

    tau = aer['TOTEXTTAU'].transpose('time', 'lat', 'lon')
    lat = np.asarray(tau['lat'].values, dtype=np.float64)
    lon = np.asarray(tau['lon'].values, dtype=np.float64)
    tcoord = np.asarray(tau['time'].values)
    vals = np.asarray(tau.values, dtype=np.float64)

    # Keep native-index derivative scaling so thresholds remain comparable to
    # the historical satellite-grid gradient metric.
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        vals = vals[:, ::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        vals = vals[:, :, ::-1]

    dlat, dlon = np.gradient(vals, axis=(1, 2))
    grad2 = dlat * dlat + dlon * dlon
    if include_temporal and vals.shape[0] > 1:
        dt = np.gradient(vals, axis=0)
        grad2 = grad2 + (float(temporal_scale_steps) * dt) ** 2
    grad_native = np.sqrt(grad2).astype(np.float32)

    grad_da = xa.DataArray(
        grad_native,
        dims=('time', 'lat', 'lon'),
        coords={'time': tcoord, 'lat': lat, 'lon': lon},
    )
    lat_v = np.asarray(lat_sat.values)
    lon_v = np.asarray(lon_sat.values)
    tv = np.datetime64(time.values) if hasattr(time, 'values') else np.datetime64(time)
    return regular_interp(grad_da, lat_v, lon_v, tv)

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
    # ERROR is an uncertainty term (small, ~0-0.1) so float16 is precise enough
    # and halves the full-grid array (~377 MB -> ~188 MB). It is widened back to
    # float32 on NetCDF write (netCDF4 has no half type).
    err =  np.zeros((len(bands), size1, size2), dtype=np.float16)
    for i in range(len(bands)):
#        tmp = b.split('_')[0]
        err[i] = np.sqrt(data ['UNC_RANDOM'][i]**2 + data['UNC_STRUCTURED'][i]**2 + data['UNC_SYSTEMATIC'][i]**2)
#    data = data.drop_vars(['UNC_RANDOM', 'UNC_STRUCTURED', 'UNC_SYSTEMATIC'])   

    data["ERROR"] = (['bands','y','x'], err)
    return data

def compute_urtoc(Jtoa, Utoa, Jh2o, Uh2o,Jo3, Uo3, Jps, Ups, Jt550, Ut550, Urtoc_ens, Urtoc_rtm_brdf, Urtoc_rtm_fit, u2_0):

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

    sum += Urtoc_rtm_brdf ** 2
    sum += Urtoc_rtm_fit ** 2

    sum += u2_0 ** 2

    return np.sqrt(sum), (np.abs(unc_h2o), np.abs(unc_o3), np.abs(unc_ps), np.abs(unc_aot))

#@memory_tracker
def compute_rtm_fit(session, pression, iaermodel, raa, raa_ir, rtoc, sza, tauaer, vza, vza_ir, wvl):
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

# Read-only state shared with worker processes via fork (populated before the pool
# is created, so children inherit it copy-on-write without pickling).
_SHARED = {}

# Per-phase wall-clock accumulators for the tile worker (accurate for nworkers=1;
# native OpenCL/ONNX calls release the GIL so cProfile mis-attributes their time).
_PHASE_T = defaultdict(float)

# Variables actually needed by save_nc_batch from the input dataset (keeps the
# amount of data pickled back to the main process small).
_SAVE_IN_VARS = ['lat','lon','SZA','SAA','VZA','VAA','VZA_IR','VAA_IR','clm','TOA',
                 'SM_MAP','ERROR','TOTEXTTAU','TO3','TQV','SLP']


def _worker_compute_tile(task):
    """
    Compute one tile end-to-end inside a worker process and return the loaded
    (numpy-backed) input subset + output dataset. The NetCDF write is done by the
    main process to avoid concurrent writers. Heavy OpenCL / ONNX objects are
    created lazily once per worker and cached in the module-global _SHARED.
    """
    iband, jband, i, j = task
    g = _SHARED
    data = g['data']
    batch_size = g['batch_size']
    config_ = g['config_']
    ca_ = g['ca_']
    ca_ind = g['ca_ind']
    monthly_datasets = g['monthly_datasets']
    dem_ds = g['dem_ds']
    merra_aer_g, merra_p2_g, merra_shift = g['merra_globals']
    anc_store = g.get('anc_store')
    kg_zone_multipliers = g.get('kg_zone_multipliers')
    s1 = g['s1']
    s2 = g['s2']
    # When enabled ([Options] debug=True), also save the terrain slope (degrees),
    # elevation and delta-elevation used by the terrain uncertainty term, to help
    # diagnose cases where uncertainty_from_terrain looks too large.
    debug = bool(config_.get('debug', False))

    S = g.get('S')
    if S is None:
        # Cap the OpenCL CPU threads per worker so N workers don't oversubscribe
        # the cores (must be set before the first OpenCL context is created).
        ncpu = os.cpu_count() or 1
        per = max(1, ncpu // max(1, g.get('nworkers', 1)))
        os.environ.setdefault('POCL_MAX_PMT_COUNT', str(per))
        os.environ.setdefault('POCL_MAX_CU_COUNT', str(per))
        S = ISmaccl(config_, g['frac_aer_model'], g['ca2_'], platform='CPU', XBLOCK=batch_size, XGRID=batch_size, breakpoint=False)
        g['S'] = S
    rtm_session = g.get('rtm_session')
    if rtm_session is None:
        rtm_session = ort.InferenceSession(config_['rmt_fit_file'])
        g['rtm_session'] = rtm_session

    y_min = max(0, i - 1)
    y_max = min(s1, i + batch_size + 1)
    x_min = max(0, j - 1)
    x_max = min(s2, j + batch_size + 1)
    data_batch = data.isel(x=slice(x_min, x_max), y=slice(y_min, y_max))

    # Derive delta-elevation uncertainty from elevation and local maximum slope
    # using a geolocation error of N pixels (default 0.5 px).
    if 'elev' in data_batch.variables:
        delev = compute_delta_elevation_from_elev(
            data_batch['elev'].values,
            config_.get('sensor_spatial_resolution_m', 1000),
            config_.get('terrain_geolocation_error_pixels', 0.5),
            config_.get('terrain_slope_window', 3),
        )
        data_batch['Delev'] = (('y', 'x'), delev)

    latitude = data_batch.lat
    longitude = data_batch.lon
    date_time = data_batch["mean-time"]
    _tp = time()

    # MERRA2 fields (+ best/monthly aerosol indices) come from the local Zarr
    # cache built once in precompute_ancillary: slice this tile's halo'd window
    # from local disk instead of re-interpolating and re-reading the network.
    # Fallback to per-tile interpolation when the cache is disabled.
    aod_grad_native = None
    if anc_store is not None:
        anc = g.get('anc_ds')
        if anc is None:
            anc = xa.open_zarr(anc_store, consolidated=False)
            g['anc_ds'] = anc
        anc_tile = anc.isel(y=slice(y_min, y_max), x=slice(x_min, x_max)).load()
        for p in ['TOTEXTTAU','TQV','TO3','SLP','T10M','BC_FRAC','DU_FRAC','SS_FRAC','SU_FRAC','OC_FRAC']:
            data_batch[p] = (('y', 'x'), anc_tile[p].values)
        if 'AOD_GRAD_NATIVE' in anc_tile.variables:
            aod_grad_native = anc_tile['AOD_GRAD_NATIVE'].values
        iaer_best = anc_tile['iaer_best'].values
        iaer_month = anc_tile['iaer_month'].values
    else:
        filtre_tile = np.isnan(data_batch['SZA'])
        merra_tile = interp_merra_tile(merra_aer_g, merra_p2_g, latitude, longitude, date_time, merra_shift)
        for p in ['TOTEXTTAU','TQV','TO3','SLP','T10M','BC_FRAC','DU_FRAC','SS_FRAC','SU_FRAC','OC_FRAC']:
            data_batch[p] = merra_tile[p].where(~filtre_tile, other=np.nan)
        iaer_best = get_iaer(data_batch)
        iaer_month = calculate_monthly_aerosol(date_time, latitude, longitude, datasets=monthly_datasets)
    _now = time(); _PHASE_T['1_ancillary_read'] += _now - _tp; _tp = _now
    iaero = np.zeros((config_['nmodels']+1, iaer_best.shape[0], iaer_best.shape[1]), dtype=np.int32)
    iaero[0] = iaer_best
    iaero[1:] = iaer_month
    pression = data_batch['SLP'] * config_['k_p0']
    slope_err, slope_deg = get_slope_err(data_batch, data_batch['TOTEXTTAU'].values, pression.values, ca_, ca_ind, iaero[1], chunksize=batch_size, dem_ds=dem_ds)
    slope_err = np.maximum(slope_err, -2)
    slope_err = np.minimum(slope_err, 2)
    _now = time(); _PHASE_T['2_slope_err'] += _now - _tp; _tp = _now

    # suppression des lignes supplémentaires (halo)
    if y_min != 0: y_min_2 = 1
    else: y_min_2 = 0
    if y_max != s1: y_max_2 = -1
    else: y_max_2 = s1
    if x_min != 0: x_min = 1
    else: x_min = 0
    if x_max != s2: x_max = -1
    else: x_max = s2

    data_batch = data_batch.isel(x=slice(x_min, x_max), y=slice(y_min_2, y_max_2))
    iaero = iaero[:, y_min_2:y_max_2, x_min:x_max]
    ds_out = run(S, data_batch, iaero)
    _now = time(); _PHASE_T['3_smaccl_run'] += _now - _tp; _tp = _now
    slope_err = slope_err[:, y_min_2:y_max_2, x_min:x_max]
    slope_deg = slope_deg[y_min_2:y_max_2, x_min:x_max]
    ds_out['iaero'] = (('y','x'), iaer_best[y_min_2:y_max_2, x_min:x_max])

    # AOD-gradient proxy for artefact flagging.
    grad_source = str(config_.get('aod_grad_source', 'merra_native')).lower()
    if grad_source == 'merra_native':
        if aod_grad_native is None:
            aod_grad = interp_merra_native_aod_grad(
                merra_aer_g,
                data_batch['lat'],
                data_batch['lon'],
                date_time,
                merra_shift,
                include_temporal=bool(config_.get('aod_grad_native_use_temporal', False)),
                temporal_scale_steps=float(config_.get('aod_grad_temporal_scale_steps', 1.0)),
            )
        else:
            aod_grad = aod_grad_native[y_min_2:y_max_2, x_min:x_max]
    else:
        ygrad, xgrad = da.gradient(data_batch['TOTEXTTAU'].data)
        aod_grad = np.sqrt(ygrad**2 + xgrad**2)
        del ygrad
        del xgrad
    ds_out = ds_out.assign({'aod_grad': (('y','x'), aod_grad)})

    urtoc_rtm_slope = np.abs(ds_out['rTOC'].data * (slope_err - 1))
    ds_out['slope_err'] = (('bands','y', 'x'), slope_err)
    ds_out['urtoc_terrain'] = (('bands','y', 'x'), urtoc_rtm_slope)
    if debug:
        ds_out['slope_deg'] = (('y', 'x'), slope_deg.astype(np.float32))
    flag = build_flag(data_batch, ds_out, config)
    ds_out['flag'] = flag
    _now = time(); _PHASE_T['4_flag_aodgrad'] += _now - _tp; _tp = _now

    if config_.get('enable_brdf_uncertainty', True):
        urtoc_rtm_brdf = np.abs(ds_out['rTOC_0'].values - ds_out['rTOC'].values)
    else:
        urtoc_rtm_brdf = np.zeros_like(ds_out['rTOC'].values, dtype=np.float32)
    ds_out['UrTOC_rtm_brdf'] = (('bands', 'y', 'x'), urtoc_rtm_brdf)
    raa = data_batch['SAA'].values - data_batch['VAA'].values
    raa = raa % 360
    f = (raa > 180)
    raa[f] = 360-raa[f]
    raa_ir = data_batch['SAA'].values - data_batch['VAA_IR'].values
    raa_ir = raa_ir % 360
    f = (raa_ir > 180)
    raa_ir[f] = 360-raa_ir[f]
    urtoc_rtm_fit = compute_rtm_fit(rtm_session, data_batch['SLP']*config_['k_p0'], ds_out['iaero'], raa, raa_ir, ds_out['rTOC'].values, data_batch['SZA'].values, data_batch['TOTEXTTAU'].values, data_batch['VZA'].values, data_batch['VZA_IR'].values, data_batch.wavelengths)
    ds_out['UrTOC_rtm_fit'] = (('bands', 'y', 'x'), urtoc_rtm_fit)
    _now = time(); _PHASE_T['5_rtm_fit_onnx'] += _now - _tp; _tp = _now

    dtaup_for_unc = ds_out['Dtaup'].data
    if config_.get('enable_koppen_geiger_aod_multiplier', True):
        kg_ds = g.get('kg_ds')
        if kg_ds is None:
            kp_map_file = config_.get('kp_map_file')
            if kp_map_file and rasterio is not None:
                try:
                    kg_ds = rasterio.open(kp_map_file)
                    g['kg_ds'] = kg_ds
                except Exception:
                    g['kg_ds'] = False
                    kg_ds = False
            else:
                g['kg_ds'] = False
                kg_ds = False
        if kg_ds not in (None, False) and kg_zone_multipliers:
            kg_mult = sample_koppen_geiger_multiplier(
                data_batch['lat'].values,
                data_batch['lon'].values,
                kg_ds,
                kg_zone_multipliers,
                default=1.0,
            )
            dtaup_for_unc = dtaup_for_unc * kg_mult

    urtoc, unc = compute_urtoc(ds_out['Jrtoa'].data, ds_out['Drtoa'].data, ds_out['Juh2o'].data, ds_out['Duh2o'].data, ds_out['Juo3'].data, ds_out['Duo3'].data, ds_out['Jpre'].data, ds_out['Dpre'].data, ds_out['Jtau550'].data, dtaup_for_unc, ds_out['UrTOC_ens'].data, urtoc_rtm_brdf, urtoc_rtm_fit, config_['u2_0'])
    ds_out['UrTOC'] = (('bands','y','x'), urtoc)
    ds_out['unc_h2o'] = (('bands','y','x'), unc[0])
    ds_out['unc_o3']  = (('bands','y','x'), unc[1])
    ds_out['unc_ps']  = (('bands','y','x'), unc[2])
    ds_out['unc_aot'] = (('bands','y','x'), unc[3])

    ds_out = apply_cf_attributes_from_json(ds_out, config_['cf_json_path'])
    data_batch = apply_cf_attributes_from_json(data_batch, config_['cf_json_path'])

    _now = time(); _PHASE_T['6_uncertainty'] += _now - _tp; _tp = _now
    # Keep only what save needs and load to numpy (compute in the worker). In
    # debug mode also keep elev/Delev so they can be written to the output file.
    _save_in_vars = _SAVE_IN_VARS + ['elev', 'Delev'] if debug else _SAVE_IN_VARS
    data_batch = data_batch[[v for v in _save_in_vars if v in data_batch.variables]].load()
    ds_out = ds_out.load()
    _now = time(); _PHASE_T['7_cf_load'] += _now - _tp; _tp = _now
    return iband, jband, data_batch, ds_out


# MERRA2 fields cached to the local ancillary Zarr store (interpolated once).
_MERRA_CACHE_VARS = ['TOTEXTTAU','TQV','TO3','SLP','T10M',
                     'BC_FRAC','DU_FRAC','SS_FRAC','SU_FRAC','OC_FRAC',
                     'AOD_GRAD_NATIVE']
# Keep pressure/temperature in float32: ISmaccl->Ps computes log(R*T), and
# float16 overflows for typical T10M (~290 K) because R*T > 65504.
_MERRA_CACHE_FLOAT32_VARS = {'SLP', 'T10M'}


def precompute_ancillary(data, config_, merra_globals, monthly_datasets,
                         batch_size, tmp_root=None):
    """
    Interpolate the MERRA2 fields, the best aerosol-model index and the AMIP
    monthly aerosol-model indices onto the FULL satellite grid exactly once and
    stream the result, tile by tile, to a local Zarr store on temporary disk.

    Rationale: the source ancillary files live on a network FS. Interpolating
    them per tile (and, for AMIP, running closest-model per tile) both re-reads
    the network and repeats work. By materialising the interpolated products
    once to a *local* Zarr store, every processing tile then just slices its
    (halo'd) window from local disk -- no per-tile network I/O, no repeated
    interpolation.

    The precompute is streamed one tile at a time so peak memory stays bounded
    (a whole-grid interpolation would otherwise materialise ~2.8 GB), keeping it
    safe on the 4 GB deployment nodes. The per-tile interpolation is pointwise,
    so reading a halo'd window from the assembled full-grid store is numerically
    identical to the previous per-tile interpolation of a halo'd batch.

    Returns (tmpdir, store_path); the caller must remove tmpdir when finished.
    """
    merra_aer_g, merra_p2_g, merra_shift = merra_globals
    s1, s2 = data['SZA'].shape
    nmonths = len(monthly_datasets)

    if tmp_root:
        os.makedirs(tmp_root, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix='fdr4vgt_anc_', dir=tmp_root)
    store = os.path.join(tmpdir, 'ancillary.zarr')

    # Write the store schema/chunking up-front (metadata only); the values are
    # filled in by the per-tile region writes below. zarr_format=2 avoids an
    # xarray/zarr-v3 _FillValue incompatibility that breaks region writes. Most
    # MERRA fields are stored as float16 to halve cache size, but SLP/T10M stay
    # float32 to avoid overflow in pressure correction (Ps uses log(R*T)).
    # The aerosol-model indices (uint8) are exact.
    template = xa.Dataset(
        {v: (('y', 'x'), da.zeros((s1, s2),
                                  dtype=(np.float32 if v in _MERRA_CACHE_FLOAT32_VARS else np.float16),
                                  chunks=(batch_size, batch_size)))
         for v in _MERRA_CACHE_VARS}
    )
    template['iaer_best'] = (('y', 'x'),
                             da.zeros((s1, s2), dtype=np.uint8,
                                      chunks=(batch_size, batch_size)))
    template['iaer_month'] = (('month', 'y', 'x'),
                              da.zeros((nmonths, s1, s2), dtype=np.uint8,
                                       chunks=(nmonths, batch_size, batch_size)))
    template.to_zarr(store, compute=False, mode='w',
                     zarr_format=2, consolidated=False)

    t0 = time()
    ntiles = (((s1 + batch_size - 1) // batch_size) *
              ((s2 + batch_size - 1) // batch_size))
    # Profiling aid: cap the precomputed tiles so a FDR4VGT_MAXTILES-limited run
    # stays bounded (halo reads of un-precomputed neighbours are irrelevant to
    # timing). Gated, default off.
    _maxtiles = int(os.environ.get('FDR4VGT_MAXTILES', 0))
    done = 0
    for i in range(0, s1, batch_size):
        if _maxtiles and done >= _maxtiles:
            break
        for j in range(0, s2, batch_size):
            if _maxtiles and done >= _maxtiles:
                break
            y_sl = slice(i, min(s1, i + batch_size))
            x_sl = slice(j, min(s2, j + batch_size))
            tile = data.isel(y=y_sl, x=x_sl)
            lat = tile.lat
            lon = tile.lon
            dt = tile['mean-time']
            filtre = np.isnan(tile['SZA'].values)

            merra_tile = interp_merra_tile(merra_aer_g, merra_p2_g,
                                           lat, lon, dt, merra_shift)
            aod_grad_native = interp_merra_native_aod_grad(
                merra_aer_g,
                lat,
                lon,
                dt,
                merra_shift,
                include_temporal=bool(config_.get('aod_grad_native_use_temporal', False)),
                temporal_scale_steps=float(config_.get('aod_grad_temporal_scale_steps', 1.0)),
            )
            region_vars = {}
            for v in _MERRA_CACHE_VARS:
                if v == 'AOD_GRAD_NATIVE':
                    arr = np.asarray(aod_grad_native, dtype=np.float32)
                else:
                    arr = np.asarray(merra_tile[v].values, dtype=np.float32)
                arr[filtre] = np.nan
                region_vars[v] = (('y', 'x'), arr)
            merra_masked = xa.Dataset(region_vars)

            # iaer indices computed from the float32 fields (accuracy), then the
            # MERRA fields are cast to float16 for storage.
            iaer_best = get_iaer(merra_masked).astype(np.uint8)
            iaer_month = calculate_monthly_aerosol(
                dt, lat, lon, datasets=monthly_datasets).astype(np.uint8)

            region_ds = xa.Dataset({
                **{v: (('y', 'x'), region_vars[v][1].astype(np.float32 if v in _MERRA_CACHE_FLOAT32_VARS else np.float16))
                   for v in _MERRA_CACHE_VARS},
                'iaer_best': (('y', 'x'), iaer_best),
                'iaer_month': (('month', 'y', 'x'), iaer_month),
            })
            region_ds.to_zarr(store, region={'y': y_sl, 'x': x_sl},
                              consolidated=False)
            done += 1
    print("[precompute_ancillary] {}/{} tiles -> {} in {:.1f}s".format(
        done, ntiles, store, time() - t0), flush=True)
    return tmpdir, store


#@memory_tracker
#def process_batched(data, frac_aer_model, ca_, ca_ind, config_, list_aerosol, batch_size=512):
def process_batched(data, frac_aer_model, ca_, ca_ind, config_, batch_size=512, merra_globals=None, kg_zone_multipliers=None):
    dtype = np.dtype([(k, 'f4') for k in ca_ind.keys()])
    ca2_ = np.ones((ca_.shape[0], ca_.shape[1]), dtype=dtype, order='C')
    for idx, name in enumerate(ca_ind.keys()):
        ca2_[name] = ca_[:,:,idx]

    s1, s2 = data['SZA'].shape

    # Write the output NetCDF to a *local* temp file and move it to the final
    # (network) destination once at the end. The serial per-tile append writes
    # are latency-bound and degrade badly on the ceph network FS (save time grew
    # ~2s -> 5s/tile as the file filled); a local scratch file keeps them fast,
    # replacing many slow network appends with one sequential copy at the end.
    # Toggle via [Sizes] local_output (default on); [Paths] tmp_dir picks scratch.
    final_output = config_['output']
    try:
        _local_out = int(config_.get('local_output', 1))
    except (KeyError, TypeError, ValueError):
        _local_out = 1
    out_tmpdir = None
    if _local_out:
        _out_root = config_.get('tmp_dir') or None
        if _out_root:
            os.makedirs(_out_root, exist_ok=True)
        out_tmpdir = tempfile.mkdtemp(prefix='fdr4vgt_out_', dir=_out_root)
        output_path = os.path.join(out_tmpdir, os.path.basename(final_output))
        atexit.register(shutil.rmtree, out_tmpdir, ignore_errors=True)
        print("[process_batched] output -> local scratch {}".format(output_path), flush=True)
    else:
        output_path = final_output

    out = create_nc(output_path, (s1,s2), data.attrs['bands'], [], 1.0)
    # Open the output handle ONCE and reuse it for every tile. save_nc_batch used
    # to re-open Dataset(fn,'a') + close() each tile (100x HDF5 open/close/flush,
    # the dominant writer cost); keeping one handle open flushes only at the end.
    nc_out = Dataset(output_path, 'a', format='NETCDF4')
    date_time = data["mean-time"]

    # --- One-time initialisation shared with worker processes (built in the parent,
    #     inherited copy-on-write via fork so it is not pickled per task). ---
    # Monthly aerosol grids loaded once into memory (small, tile-count independent).
    monthly_datasets = preload_monthly_aerosol(date_time)
    # DEM opened once and loaded into RAM so per-tile terrain computation is a
    # numpy operation, not an HDF5/netCDF read every tile (that per-tile DEM read
    # was ~5 s/tile). Crop to the product's latitude AND longitude extent: the
    # global GTOPO30 has 43200 lon columns but the product spans a narrow lon
    # range, so lat-only cropping wasted ~1.7 GB (kept all lon). Cropping lon too
    # drops the DEM from ~1.9 GB to ~0.2 GB.
    dem_ds = xa.open_dataset(config.dem_path, chunks="auto")
    dem_ds = xrcrop(dem_ds, lat=data['y'], lon=data['x']).load()

    try:
        nworkers = int(config_.get('nworkers'))
    except (KeyError, TypeError, ValueError):
        nworkers = 1
    nworkers = max(1, min(nworkers, (os.cpu_count() or 1)))

    # Interpolate MERRA + AMIP ancillary to the full grid once and stream it to a
    # local Zarr cache on temp disk (built before the fork so workers just slice
    # their halo'd window from local disk -- no per-tile network I/O). Toggle via
    # [Sizes] anc_cache (default on); optional [Paths] tmp_dir picks the scratch.
    try:
        _anc_cache = int(config_.get('anc_cache', 1))
    except (KeyError, TypeError, ValueError):
        _anc_cache = 1
    anc_tmpdir = None
    anc_store = None
    if _anc_cache:
        anc_tmpdir, anc_store = precompute_ancillary(
            data, config_, merra_globals, monthly_datasets, batch_size,
            tmp_root=config_.get('tmp_dir') or None)
        atexit.register(shutil.rmtree, anc_tmpdir, ignore_errors=True)

    # 'robust' Bit-3 threshold: compute it ONCE from the whole-scene cached
    # AOD_GRAD_NATIVE grid rather than per-tile (the previous per-tile
    # statistics were found to be self-masking -- a tile overlapping the
    # scene's strongest real gradient feature has its own median/MAD inflated
    # by that very feature, raising its local bar and hiding what should be
    # flagged; a global threshold from the same k/quantile recipe avoids this
    # and was empirically validated to track a fixed ~0.15 threshold closely
    # on a high-AOT test scene while remaining scene-adaptive). Falls back to
    # the existing per-tile computation in build_flag() when the ancillary
    # cache is unavailable (anc_cache disabled).
    if anc_store is not None and str(config_.get('aod_grad_threshold_method', 'fixed')).lower() == 'robust':
        grad_full = xa.open_zarr(anc_store, consolidated=False)['AOD_GRAD_NATIVE'].values
        gvalid = np.isfinite(grad_full)
        if np.any(gvalid):
            g = grad_full[gvalid].astype(np.float32)
            med = np.median(g)
            mad = np.median(np.abs(g - med))
            sigma = 1.4826 * mad
            k = float(config_.get('aod_grad_robust_k', 6.0))
            q = min(max(float(config_.get('aod_grad_robust_quantile', 0.995)), 0.5), 0.999999)
            qv = float(np.quantile(g, q))
            grad_floor = float(config_.get('aodmax_grad', config_.get('aod_max_grad', 1.4e-4)))
            thr_global = max(grad_floor, float(med + k * sigma), qv)
            config.set('aod_grad_robust_thr_global', thr_global)
            print("[process_batched] global robust aod_grad threshold = {:.5g} "
                  "(med={:.4g} sigma={:.4g} q{:.1f}={:.4g})".format(
                      thr_global, med, sigma, 100 * q, qv), flush=True)
        del grad_full

    _SHARED.update(dict(
        data=data, batch_size=batch_size, config_=config_, ca_=ca_, ca_ind=ca_ind,
        ca2_=ca2_, frac_aer_model=frac_aer_model, monthly_datasets=monthly_datasets,
        dem_ds=dem_ds, merra_globals=merra_globals, s1=s1, s2=s2, nworkers=nworkers,
        anc_store=anc_store, kg_zone_multipliers=kg_zone_multipliers))

    tasks = [(iband, jband, i, j)
             for iband, i in enumerate(range(0, s1, batch_size))
             for jband, j in enumerate(range(0, s2, batch_size))]
    ntiles_total = len(tasks)

    proc = psutil.Process(os.getpid())
    peak_rss = proc.memory_info().rss
    t_loop_start = time()
    print("[process_batched] grid={}x{} batch={} -> {} tiles, {} workers".format(
        s1, s2, batch_size, ntiles_total, nworkers), flush=True)

    gc.collect()

    save_total = 0.0

    def _report(done, iband, jband, t_save):
        nonlocal peak_rss, save_total
        save_total += t_save
        peak_rss = max(peak_rss, proc.memory_info().rss)
        elapsed = time() - t_loop_start
        eta = elapsed / done * (ntiles_total - done)
        print("[tile {:>3d}/{:<3d}] saved y={:>2d} x={:>2d} | save {:4.2f}s "
              "| RSS {:5.0f}MB | elapsed {:6.1f}s ETA {:6.1f}s".format(
                  done, ntiles_total, iband, jband, t_save,
                  proc.memory_info().rss / 1024 / 1024, elapsed, eta), flush=True)

    done = 0
    _maxtiles = int(os.environ.get('FDR4VGT_MAXTILES', 0))
    if _maxtiles:
        tasks = tasks[:_maxtiles]
        ntiles_total = len(tasks)
    # The single open NetCDF handle keeps written (compressed) chunks dirty in the
    # HDF5 cache until they are flushed; without periodic flushing the whole output
    # (all variables) accumulates in RAM and breaks the 4 GB budget. Flush every
    # The single open NetCDF handle keeps written (compressed) chunks in the HDF5
    # cache; NetCDF sync() does not release it, so the whole output accumulates in
    # RAM (breaks the 4 GB budget). Close+reopen every flush_interval tiles fully
    # frees the HDF5 memory while still avoiding a per-tile open/close.
    try:
        flush_interval = max(1, int(config_.get('flush_interval', 4)))
    except (KeyError, TypeError, ValueError):
        flush_interval = 4
    if nworkers == 1:
        # Serial, in-process (bounded memory, no fork). Reuses the worker body.
        for task in tasks:
            iband, jband, ds_in, ds_out = _worker_compute_tile(task)
            t0 = time()
            save_nc_batch(nc_out, ds_in, ds_out, iband, jband, batch_size, config_['jacobian'], config_.get('debug', False))
            if done % flush_interval == 0:
                nc_out.close()
                nc_out = Dataset(output_path, 'a', format='NETCDF4')
            done += 1
            _report(done, iband, jband, time() - t0)
            del ds_in, ds_out
            gc.collect()
    else:
        # Parallel across worker processes (fork => read-only base shared COW).
        # Submissions are throttled so completed results do not pile up in the
        # main process (bounds memory); the NetCDF write stays single-writer.
        ctx = get_context('fork')
        max_inflight = nworkers * 2
        with ProcessPoolExecutor(max_workers=nworkers, mp_context=ctx) as ex:
            it = iter(tasks)
            inflight = []
            for _ in range(max_inflight):
                try:
                    inflight.append(ex.submit(_worker_compute_tile, next(it)))
                except StopIteration:
                    break
            while inflight:
                fut = inflight.pop(0)
                iband, jband, ds_in, ds_out = fut.result()
                t0 = time()
                save_nc_batch(nc_out, ds_in, ds_out, iband, jband, batch_size, config_['jacobian'], config_.get('debug', False))
                if done % flush_interval == 0:
                    nc_out.close()
                    nc_out = Dataset(output_path, 'a', format='NETCDF4')
                done += 1
                _report(done, iband, jband, time() - t0)
                del ds_in, ds_out
                try:
                    inflight.append(ex.submit(_worker_compute_tile, next(it)))
                except StopIteration:
                    pass

    total = time() - t_loop_start
    # Close the single output handle once (flushes any buffered HDF5 chunks).
    t_close = time()
    nc_out.close()
    close_dt = time() - t_close
    print("\n==== process_batched done: {} tiles in {:.1f}s ({:.2f}s/tile) "
          "peak RSS {:.0f} MB, {} workers ====\n".format(
              ntiles_total, total, total / max(1, ntiles_total),
              peak_rss / 1024 / 1024, nworkers), flush=True)

    # Per-phase wall-clock breakdown (accurate; accumulated in the tile worker).
    if _PHASE_T:
        nt = max(1, done)
        phase_sum = sum(_PHASE_T.values())
        print("---- per-phase compute breakdown ({} tiles) ----".format(done))
        for name in sorted(_PHASE_T):
            t = _PHASE_T[name]
            print("  {:22s} {:8.1f}s  {:6.2f}s/tile  {:5.1f}%".format(
                name, t, t / nt, 100 * t / phase_sum), flush=True)
        print("  {:22s} {:8.1f}s  {:6.2f}s/tile".format(
            'compute subtotal', phase_sum, phase_sum / nt), flush=True)
        print("  {:22s} {:8.1f}s  {:6.2f}s/tile".format(
            '8_save_nc (writer)', save_total, save_total / nt), flush=True)
        print("  {:22s} {:8.1f}s".format('9_nc_close_flush', close_dt), flush=True)

    # Move the locally-written output to its final (network) destination once.
    if out_tmpdir is not None:
        t_mv = time()
        dst_dir = os.path.dirname(final_output)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
        # copyfile (data only) not move/copy2: the destination may be owned by
        # another user, so copystat (utime/perms) would raise PermissionError.
        # Group-write is enough to overwrite the file contents in place.
        shutil.copyfile(output_path, final_output)
        print("[process_batched] copied output -> {} in {:.1f}s".format(
            final_output, time() - t_mv), flush=True)
        shutil.rmtree(out_tmpdir, ignore_errors=True)

    if anc_tmpdir is not None:
        shutil.rmtree(anc_tmpdir, ignore_errors=True)

    return None, None

def process(configfile):

    dask.config.set(scheduler='synchronous') 
#    chunks = 256
    config_ = readConfig(configfile)
    # Use consistent naming for AOD thresholds.
    # Primary names: aodmax, aodmax_grad.
    # Legacy aliases accepted: aotmax, aod_max_grad, taot.
    if 'aodmax' not in config_:
        config_['aodmax'] = config_.get('aotmax', config_.get('taot', 0.6))
    if 'aodmax_grad' not in config_:
        config_['aodmax_grad'] = config_.get('aod_max_grad', 1.4e-4)
    # Separate fixed-threshold constant for aod_grad_source=merra_native: the
    # native-grid index-based gradient is ~2-3 orders of magnitude larger than
    # the legacy interpolated-grid gradient that `aodmax_grad` was calibrated
    # for, so reusing `aodmax_grad` as a fixed threshold there flags ~100% of
    # pixels. Only used when aod_grad_threshold_method='fixed'; the 'robust'
    # method is scale-adaptive and unaffected.
    config_.setdefault('aodmax_grad_native', 0.15)
    config_.setdefault('aod_grad_source', 'merra_native')
    config_.setdefault('aod_grad_threshold_method', 'robust')
    config_.setdefault('aod_grad_robust_k', 6.0)
    config_.setdefault('aod_grad_robust_quantile', 0.995)
    config_.setdefault('aod_grad_native_use_temporal', False)
    config_.setdefault('aod_grad_temporal_scale_steps', 1.0)
    config_.setdefault('aod_grad_morph_enable', False)
    config_.setdefault('aod_grad_morph_radius_px', 1)
    config_.setdefault('aod_grad_morph_closing_radius_px', 0)
    config_.setdefault('aod_grad_morph_guard_rel', 0.5)
    config_.setdefault('aod_grad_morph_aod_rel', 0.8)
    # SMACCL still expects taot for its internal high-AOD process flag.
    config_['taot'] = float(config_['aodmax'])
    config.set('amip_path', config_['amip_path'])
    _coef_path = Path(config_['smaccoef_dir']) / f"{config_['sensor']}_smac_coeffs_v3.0.npy"
    config.set('aerosol_model_fraction', config_['faer'])
    config.set('dem_path', config_['dem'])
    config.set('kept_data_path', config_['kept_data_path'])
    config.set('aodmax', config_['aodmax'])
    config.set('aotmax', config_['aodmax'])
    config.set('tocmin', config_['tocmin'])
    config.set('tocmax', config_['tocmax'])
    config.set('szamax', config_['szamax'])
    config.set('aodmax_grad', config_['aodmax_grad'])
    config.set('aod_max_grad', config_['aodmax_grad'])
    config.set('aodmax_grad_native', config_['aodmax_grad_native'])
    config.set('aod_grad_source', config_['aod_grad_source'])
    config.set('aod_grad_threshold_method', config_['aod_grad_threshold_method'])
    config.set('aod_grad_robust_k', config_['aod_grad_robust_k'])
    config.set('aod_grad_robust_quantile', config_['aod_grad_robust_quantile'])
    config.set('aod_grad_morph_enable', config_['aod_grad_morph_enable'])
    config.set('aod_grad_morph_radius_px', config_['aod_grad_morph_radius_px'])
    config.set('aod_grad_morph_closing_radius_px', config_['aod_grad_morph_closing_radius_px'])
    config.set('aod_grad_morph_guard_rel', config_['aod_grad_morph_guard_rel'])
    config.set('aod_grad_morph_aod_rel', config_['aod_grad_morph_aod_rel'])

    # Optional uncertainty switches / parameters.
    config_.setdefault('enable_brdf_uncertainty', True)
    config_.setdefault('enable_koppen_geiger_aod_multiplier', True)
    config_.setdefault('sensor_spatial_resolution_m', 1000)
    config_.setdefault('terrain_geolocation_error_pixels', 0.5)
    config_.setdefault('terrain_slope_window', 3)

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
    # UNC_* bands are only needed to build ERROR; drop them to free ~1.1 GB.
    data = data.drop_vars([v for v in ['UNC_RANDOM','UNC_STRUCTURED','UNC_SYSTEMATIC'] if v in data.variables])
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

    # read MERRA2 (opened once, small & global; interpolated per tile to bound memory)
    merra_aer_g, merra_p2_g, merra_shift = open_merra_global(merra_aer, merra_p2, data['lon'].values, data['mean-time'])

    # read smac coeffs
    frac_aer_model = pre_aer_models(aer_file)

    smac_coeffs_file = data.attrs['smac_coeffs_file']
    ca_, ca_ind = read_smac_coefficients(smac_coeffs_file)

    kg_zone_multipliers = None
    if config_.get('enable_koppen_geiger_aod_multiplier', True):
        mapping_file = config_.get('mapping_file')
        if mapping_file and os.path.exists(mapping_file):
            kg_zone_multipliers = load_koppen_geiger_zone_multipliers(mapping_file)

    data = data.assign_attrs({'jac_name' : ['Juh2o','Juo3','Jrtoa','Jpre','Drsurf']})
    data = data.chunk({'y':chunks, 'x':chunks, 'bands':-1})

    # On memory-rich hosts, load the whole L1 product into RAM once so that each
    # tile is a pure numpy slice instead of re-reading HDF5 chunks every tile
    # (the ~5 s/tile "loading" cost). Keep off (0) for the 4 GB deployment nodes.
    try:
        _preload = int(config_.get('preload', 0))
    except (KeyError, TypeError, ValueError):
        _preload = 0
    if _preload:
        print("[process] preloading full L1 data into RAM...", flush=True)
        data = data.load()

    _toc,  _jac = process_batched(data, frac_aer_model, ca_, ca_ind, config_, batch_size=chunks,
                                merra_globals=(merra_aer_g, merra_p2_g, merra_shift),
                                kg_zone_multipliers=kg_zone_multipliers)
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python process.py <configfile>")
        exit(1)
    configfile = sys.argv[1]
    process(configfile)