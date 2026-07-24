import xarray as xr
import numpy as np
from pathlib import Path
import xdem.terrain as xdem_terrain
from core.interpolate import interp, Linear
from core.tools import xrcrop
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import maximum_filter, binary_closing, gaussian_filter, distance_transform_edt
import psutil
import os
from time import time
import gc
from glob import glob
import csv

try:
    import rasterio
except Exception:
    rasterio = None

class config_class:
    def __init__(self):
        pass

    def set(self, name, value):
        self.__dict__[name] = value

    def getfloat(self, name):
        assert(name in self.__dict__.keys())
        return self.__dict__[name]


config = config_class()


def _nearest_valid_fill(arr, valid):
    """Fill invalid pixels with the value of their nearest valid neighbor.

    Filling invalid/no-data regions (ocean, lakes, DEM gaps) with a single
    global constant (e.g. nanmedian) creates an artificial cliff at the
    boundary of every invalid patch: a flat plateau surrounded by real
    terrain. Differentiating across that boundary (slope/aspect) then
    produces spurious high-gradient artefacts that trace the shape of the
    invalid region -- a single stripe along a roughly linear coastline, or a
    ring/arc around a blob-shaped island/lake/no-data patch. Nearest-valid
    fill keeps the filled value locally continuous with the surrounding real
    data, so no artificial edge is introduced.
    """
    if np.all(valid):
        return arr
    if not np.any(valid):
        return np.zeros_like(arr)
    _, indices = distance_transform_edt(~valid, return_indices=True)
    return arr[tuple(indices)]


def shift_lon_to_360(ds, lon_sat):
    '''
    Convert dataset longitudes from [-180, 180] to [0, 360] when needed.

    If the satellite longitude array contains only positive values and the
    dataset longitudes include negative values, this function shifts the
    negative longitudes by +360 and sorts the coordinate values.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with a longitude coordinate named 'lon'.
    lon_sat : xarray.DataArray or array-like
        Satellite longitudes used to determine whether the dataset longitudes
        should be converted to the [0, 360] range.

    Returns
    -------
    xarray.Dataset
        Dataset with adjusted longitude coordinates when conversion is applied.
    '''
    if np.nanmax(lon_sat) > np.max(ds.lon.values):
        filtre = ds.lon.values < 0
        new_lon = ds.lon.values.copy()
        new_lon[filtre] = new_lon[filtre] + 360
        ds = ds.assign_coords(lon=new_lon)
        ds = ds.sortby('lon')
    return ds

def build_flag(ds_input, ds_output, config_data):
#    flag_aot = ds_input.rtoa.max(dim="bands") >= config_data.getfloat("Coefficients", "aotmax")
    aodmax = config_data.__dict__.get('aodmax', config_data.__dict__.get('aotmax'))
    aodmax_grad = config_data.__dict__.get('aodmax_grad', config_data.__dict__.get('aod_max_grad'))
    grad_method = str(config_data.__dict__.get('aod_grad_threshold_method', 'fixed')).lower()
    grad_source = str(config_data.__dict__.get('aod_grad_source', 'merra_native')).lower()
    flag_aot = np.asarray(ds_input['TOTEXTTAU']) >= float(aodmax)
#    flag_toc_min = ds_output.rtoc_run.min(dim="bands") < config_data.getfloat("Coefficients", "tocmin")
    _flag_toc_min = np.min(ds_output['rTOC'], axis=0) < config_data.getfloat("tocmin")
#    flag_toc_max = ds_output.rtoc_run.max(dim="bands") > config_data.getfloat("Coefficients", "tocmax")
    _flag_toc_max = np.max(ds_output['rTOC'], axis=0) > config_data.getfloat("tocmax")
#    flag_sza_max = ds_input.tetas > config_data.getfloat("Coefficients", "szamax")
    flag_sza_max = ds_input['SZA'] > config_data.getfloat("szamax")
    # `aodmax_grad` (default 1.4e-4) was calibrated for the legacy
    # `aod_grad_source=interpolated` gradient (spatial gradient of AOD already
    # interpolated onto the ~1 km sensor grid, so per-pixel deltas are tiny).
    # The `merra_native` gradient is an index-based derivative on the native
    # ~0.5-0.625 deg MERRA2 grid, which is ~2-3 orders of magnitude larger in
    # typical magnitude (observed scene stats: median ~0.04, p99 ~0.17 vs a
    # 1.4e-4 floor). Using `aodmax_grad` as-is for a *fixed* threshold against
    # `merra_native` therefore flags ~100% of pixels (verified empirically) and
    # makes Bit 3 useless as a discriminator. Keep `aodmax_grad` as the floor
    # used by the (scale-adaptive) 'robust' method -- unaffected here, since
    # 1.4e-4 is always dominated there by the data-driven med/quantile terms --
    # and use a separately-calibrated `aodmax_grad_native` constant for the
    # 'fixed' method when the native source is selected.
    aodmax_grad_native = config_data.__dict__.get('aodmax_grad_native', 0.15)
    grad_floor = float(aodmax_grad)
    # Scene-wide (whole-image) robust threshold, precomputed once in
    # process_batched() from the full cached AOD_GRAD_NATIVE grid -- see there
    # for why: per-tile robust statistics were found to be self-masking (a
    # tile that itself contains the strongest real gradient feature has its
    # own median/MAD inflated by that very feature, raising its bar and
    # hiding the feature it should flag). When available, always prefer it
    # over the per-tile fallback below.
    grad_thr_global = config_data.__dict__.get('aod_grad_robust_thr_global')
    if grad_method == 'robust' and grad_thr_global is not None:
        grad_thr = float(grad_thr_global)
    elif grad_method == 'robust':
        grad = np.asarray(ds_output['aod_grad']).astype(np.float32)
        valid = np.isfinite(grad)
        if np.any(valid):
            g = grad[valid]
            med = np.median(g)
            mad = np.median(np.abs(g - med))
            sigma = 1.4826 * mad
            k = float(config_data.__dict__.get('aod_grad_robust_k', 6.0))
            q = float(config_data.__dict__.get('aod_grad_robust_quantile', 0.995))
            q = min(max(q, 0.5), 0.999999)
            qv = float(np.quantile(g, q))
            grad_thr = max(grad_floor, float(med + k * sigma), qv)
        else:
            grad_thr = grad_floor
    elif grad_source == 'merra_native':
        grad_thr = float(aodmax_grad_native)
    else:
        grad_thr = grad_floor
    flag_aot_grad = ds_output['aod_grad'] > grad_thr

    # Optional morphology-assisted expansion around high-gradient cores to
    # better capture halo-like artefacts while avoiding unconstrained growth.
    if bool(config_data.__dict__.get('aod_grad_morph_enable', False)):
        seed = np.asarray(flag_aot_grad).astype(bool)
        radius = int(config_data.__dict__.get('aod_grad_morph_radius_px', 1))
        if radius > 0:
            size = 2 * radius + 1
            expanded = maximum_filter(seed.astype(np.uint8), size=size, mode='nearest') > 0
        else:
            expanded = seed

        close_radius = int(config_data.__dict__.get('aod_grad_morph_closing_radius_px', 0))
        if close_radius > 0:
            close_size = 2 * close_radius + 1
            structure = np.ones((close_size, close_size), dtype=bool)
            expanded = binary_closing(expanded, structure=structure)

        grad = np.asarray(ds_output['aod_grad']).astype(np.float32)
        guard_rel = float(config_data.__dict__.get('aod_grad_morph_guard_rel', 0.5))
        guard = np.isfinite(grad) & (grad >= guard_rel * grad_thr)

        # Use interpolated MERRA AOD when available to constrain expansions to
        # physically plausible aerosol-risk neighborhoods.
        if 'TOTEXTTAU' in ds_input:
            tau = np.asarray(ds_input['TOTEXTTAU']).astype(np.float32)
            tau_rel = float(config_data.__dict__.get('aod_grad_morph_aod_rel', 0.8))
            guard = guard | (np.isfinite(tau) & (tau >= tau_rel * float(aodmax)))

        flag_aot_grad = seed | (expanded & guard)

    # Explicit AOD-slope "contour" criterion: the gradient-based criterion
    # above uses `aod_grad`, which by default is computed on the coarse
    # native-MERRA grid (`aod_grad_source=merra_native`) and only loosely
    # aligns with the actual AOD transition on the fine sensor grid -- a
    # broad, gradual AOD ramp can have its steepest LOCAL slope well inside or
    # outside the aodmax boundary, so Bit 3 can miss most of the true
    # transition zone even with morphology enabled (measured recall of the
    # true aodmax boundary was only ~11% on a real high-AOD scene). This adds
    # a SEPARATE criterion based on `aod_grad_fine` -- the gradient of AOD
    # already interpolated onto the actual sensor grid (Appendix F.7) -- so
    # the flagged region naturally covers the FULL width of wherever the real
    # AOD slope is elevated (a gradual transition flags a wide band; a sharp
    # one flags a narrow band), rather than an arbitrary fixed-width ring.
    if bool(config_data.__dict__.get('aod_contour_enable', True)) and 'aod_grad_fine' in ds_output:
        fine_grad = np.asarray(ds_output['aod_grad_fine']).astype(np.float32)
        thr_fine_global = config_data.__dict__.get('aod_contour_grad_thr_global')
        if thr_fine_global is not None:
            thr_fine = float(thr_fine_global)
        else:
            # Per-tile fallback (e.g. anc_cache disabled): same recipe as the
            # global one, computed locally.
            valid_fine = np.isfinite(fine_grad)
            if np.any(valid_fine):
                gf = fine_grad[valid_fine]
                med_f = np.median(gf)
                mad_f = np.median(np.abs(gf - med_f))
                sigma_f = 1.4826 * mad_f
                k_f = float(config_data.__dict__.get('aod_contour_grad_k', 2.0))
                q_f = min(max(float(config_data.__dict__.get('aod_contour_grad_quantile', 0.90)), 0.5), 0.999999)
                qv_f = float(np.quantile(gf, q_f))
                thr_fine = max(float(med_f + k_f * sigma_f), qv_f)
            else:
                thr_fine = np.inf
        aod_contour = np.isfinite(fine_grad) & (fine_grad > thr_fine)
        flag_aot_grad = np.asarray(flag_aot_grad) | aod_contour

    flag_cloud = ds_input['clm'] != 0 # cloud contaminated flag
#    flag = ((flag_aot.data.astype(np.int16) << 0) | #lsb
#            (flag_toc_min.data.astype(np.int16) << 1) |
#            (flag_toc_max.data.astype(np.int16) << 2) |
#            (flag_sza_max.data.astype(np.int16) << 3) |
#            (flag_aot_grad.data.astype(np.int16) << 4)) #msb
#    
    flag = ( (flag_cloud.astype(np.int16) << 0) | #Bit 0 :cloud_contaminated
             (flag_aot.astype(np.int16) << 1) | #Bit 1: High AOD
             (flag_sza_max.astype(np.int16) << 2) | #Bit 2: High SZA
             (flag_aot_grad.astype(np.int16) << 3) #| #Bit 3: High AOD gradient 
#             (flag_ac_fail.astype(np.int16) << 4) |   #Bit 4: AC algorithm failure    
#             (flag_missing_aux.astype(np.int16) << 5) |  #Bit 5: Missing auxiliary data
#             (flag_out_lut.astype(np.int16) << 6))    #Bit 6: Out of LUT range
    )
    return flag

def compute_atmospheric_transmissions(cos_sun,
                                    aot_550, pressure_eq,
                                    smac_coeffs, ca_ind):
    """
    Compute atmospheric transmission components for SMAC.
    
    Args:
        cos_sun: Cosine of solar zenith angle
        aot_550: Aerosol optical thickness at 550nm
        pressure_eq: Equivalent pressure (normalized by 1013 hPa)
        smac_coeffs: SMAC coefficient array
        ca_ind: Dictionary mapping coefficient names to indices
        
    Returns:
        Tuple of (total_transmission, direct_transmission, diffuse_transmission)
        
    Note:
        This function implements the transmission model used in SMAC for
        both total scattering and direct beam transmissions.
    """
    # Aerosol optical depth in spectral band
    tau_aerosol = smac_coeffs[ca_ind['a0taup']] + smac_coeffs[ca_ind['a1taup']] * aot_550
    
    # Rayleigh optical depth  
    tau_rayleigh = smac_coeffs[ca_ind['taur']] * pressure_eq 
    
    # Total transmission
    total_transmission = (smac_coeffs[ca_ind['a0T']] + 
                         smac_coeffs[ca_ind['a1T']] * aot_550 / cos_sun +
                         (smac_coeffs[ca_ind['a2T']] * pressure_eq + 
                          smac_coeffs[ca_ind['a3T']]) / (1.0 + cos_sun))
    
    # Direct transmission
    total_optical_depth = tau_aerosol + tau_rayleigh
    direct_transmission = np.exp(-total_optical_depth / cos_sun)
    
    # Diffuse transmission
    diffuse_transmission = total_transmission - direct_transmission
    
    return total_transmission, direct_transmission, diffuse_transmission

def read_smac_coefficients(filepath, exclude_first_field=True, verbose=False):
    """
    Read SMAC coefficients from either .npy (structured array) or .nc (NetCDF) format.
    
    This function loads SMAC atmospheric correction coefficients from disk and 
    returns them in a standardized format compatible with the SMAC-NEO library.
    It supports both NumPy structured arrays (.npy) and NetCDF (.nc) formats.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the coefficient file. Supported formats: '.npy' or '.nc'.
    exclude_first_field : bool, default True
        Whether to exclude the first band-name field (typically contains string 
        identifiers rather than numerical coefficients). When True, skips the 
        first field which usually contains band names or metadata.
    verbose : bool, default False
        Whether to print detailed information during processing, including
        file format detection, variable shapes, and processing steps.
        
    Returns
    -------
    ca_ : numpy.ndarray
        Coefficient array with shape (bands, aerosols, coeffs) for 3D data or 
        (records, coeffs) for 2D data. Contains the numerical SMAC coefficients
        needed for atmospheric correction calculations.
    ca_ind : dict
        Dictionary mapping coefficient names to their corresponding indices in 
        the coefficient array. Keys are coefficient names (e.g., 'a0taup', 'a1taup')
        and values are integer indices.
        
    Notes
    -----
    Supported file formats:
    
    - **NetCDF (.nc)**: Uses xarray for reading. Variables are automatically 
      detected and can have different shapes. The function homogenizes shapes
      by broadcasting or truncating as needed.
    - **NumPy structured array (.npy)**: Standard NumPy format with named fields.
      Each field represents a coefficient type.
      
    The function automatically handles shape inconsistencies in NetCDF files by:
    
    1. Identifying the most common 2D shape across variables
    2. Broadcasting smaller arrays to match target shape
    3. Truncating larger arrays to fit target shape
    4. Filling with zeros when broadcasting is not possible
    
    Examples
    --------
    Load coefficients from NetCDF file:
    
    >>> ca_, ca_ind = read_smac_coefficients('S3A_OLCI_smac_coeffs.nc')
    >>> print(f"Coefficient array shape: {ca_.shape}")
    >>> print(f"Available coefficients: {list(ca_ind.keys())}")
    
    Load with verbose output:
    
    >>> ca_, ca_ind = read_smac_coefficients(
    ...     'coefficients.npy', 
    ...     exclude_first_field=False, 
    ...     verbose=True
    ... )
    
    Access specific coefficients:
    
    >>> # Get aerosol optical depth coefficients
    >>> a0_index = ca_ind['a0taup']
    >>> a1_index = ca_ind['a1taup']
    >>> a0_coeffs = ca_[:, :, a0_index]  # Shape: (bands, aerosols)
    >>> a1_coeffs = ca_[:, :, a1_index]  # Shape: (bands, aerosols)
    """
    import os
    from pathlib import Path
    
    file_ext = Path(filepath).suffix.lower()
    
    if verbose:
        print(f"Reading SMAC coefficients from: {filepath}")
        print(f"Detected format: {file_ext}")
    
    if file_ext == '.nc':
        # NetCDF format using xarray
        if verbose:
            print("Using NetCDF format reader")
            
        ca_data = xr.open_dataset(filepath)
        
        # Get variable names (excluding coordinates)
        all_vars = list(ca_data.data_vars.keys())
        if verbose:
            print(f"All coefficient variables: {all_vars}")
        
        # Optionally exclude first field
        if exclude_first_field and len(all_vars) > 1:
            coeff_vars = all_vars[1:]
            if verbose:
                print(f"Excluded first field: {all_vars[0]}")
        else:
            coeff_vars = all_vars
            
        if verbose:
            print(f"Using coefficient variables: {coeff_vars}")
            print(f"Number of variables: {len(coeff_vars)}")
        
        # Analyze variable shapes
        shapes = []
        if verbose:
            print("\nOriginal variable shapes:")
        for var in coeff_vars:
            shape = ca_data[var].shape
            shapes.append(shape)
            if verbose:
                print(f"  {var}: {shape}")
        
        # Determine target shape (lambda, iaer)
        two_d_shapes = [s for s in shapes if len(s) == 2]
        if two_d_shapes:
            target_shape = max(set(two_d_shapes), key=two_d_shapes.count)
        else:
            # Fallback strategies
            if 'lambda' in ca_data.dims and 'iaer' in ca_data.dims:
                target_shape = (ca_data.dims['lambda'], ca_data.dims['iaer'])
            elif 'band' in ca_data.dims and 'aer_model' in ca_data.dims:
                target_shape = (ca_data.dims['band'], ca_data.dims['aer_model'])
            else:
                target_shape = ca_data[coeff_vars[0]].shape
        
        if verbose:
            print(f"\nTarget shape (lambda, iaer): {target_shape}")
            
        n_lambda, n_iaer = target_shape
        
        # Homogenize all variables to target shape
        if verbose:
            print("\nHomogenizing variables to target shape:")
            
        homogenized_data = {}
        for var in coeff_vars:
            var_data = ca_data[var].values
            original_shape = var_data.shape
            
            if original_shape == target_shape:
                homogenized_data[var] = var_data
                if verbose:
                    print(f"  {var}: {original_shape} -> kept as is")
                    
            elif len(original_shape) == 1:
                if original_shape[0] == n_lambda:
                    homogenized_data[var] = np.broadcast_to(var_data[:, np.newaxis], target_shape)
                    if verbose:
                        print(f"  {var}: {original_shape} -> {target_shape} (broadcast lambda)")
                elif original_shape[0] == n_iaer:
                    homogenized_data[var] = np.broadcast_to(var_data[np.newaxis, :], target_shape)
                    if verbose:
                        print(f"  {var}: {original_shape} -> {target_shape} (broadcast iaer)")
                else:
                    if verbose:
                        print(f"  Warning: {var} has 1D shape {original_shape} that doesn't match target dimensions")
                    if original_shape[0] == 1:
                        homogenized_data[var] = np.full(target_shape, var_data[0], dtype=np.float32)
                    else:
                        homogenized_data[var] = np.zeros(target_shape, dtype=np.float32)
                        if verbose:
                            print(f"    -> filled with zeros")
                            
            elif len(original_shape) == 0:
                homogenized_data[var] = np.full(target_shape, var_data.item(), dtype=np.float32)
                if verbose:
                    print(f"  {var}: scalar -> {target_shape} (broadcast scalar)")
                    
            else:
                try:
                    if np.prod(original_shape) == np.prod(target_shape):
                        homogenized_data[var] = var_data.reshape(target_shape)
                        if verbose:
                            print(f"  {var}: {original_shape} -> {target_shape} (reshaped)")
                    else:
                        homogenized_data[var] = np.broadcast_to(var_data, target_shape)
                        if verbose:
                            print(f"  {var}: {original_shape} -> {target_shape} (broadcast)")
                except (ValueError, TypeError) as e:
                    if verbose:
                        print(f"  Warning: Cannot homogenize {var} with shape {original_shape}: {e}")
                    homogenized_data[var] = np.zeros(target_shape, dtype=np.float32)
                    if verbose:
                        print(f"    -> filled with zeros")
        
        # Create coefficient arrays
        ca_ind = {var: i for i, var in enumerate(coeff_vars)}
        Nkeys = len(coeff_vars)
        ca_ = np.zeros((*target_shape, Nkeys), dtype=np.float32)
        
        for i, var in enumerate(coeff_vars):
            ca_[..., i] = homogenized_data[var]
            
    elif file_ext == '.npy':
        # NumPy structured array format
        if verbose:
            print("Using NumPy structured array format reader")
            
        ca_data = np.load(filepath)
        
        # Get field names
        all_field_names = ca_data.dtype.names
        if verbose:
            print(f"All fields: {all_field_names}")
        
        # Optionally exclude first field
        if exclude_first_field and len(all_field_names) > 1:
            field_names = all_field_names[1:]
            if verbose:
                print(f"Excluded first field: {all_field_names[0]}")
        else:
            field_names = all_field_names
            
        if verbose:
            print(f"Using fields: {field_names}")
            print(f"Number of fields: {len(field_names)}")
            print(f"Array shape: {ca_data.shape}")
        
        # Create coefficient arrays
        ca_ind = {name: i for i, name in enumerate(field_names)}
        Nkeys = len(field_names)
        ca_ = np.zeros((*ca_data.shape, Nkeys), dtype=np.float32)
        
        for i, field_name in enumerate(field_names):
            ca_[..., i] = ca_data[field_name]
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .nc, .npy")
    
    if verbose:
        print(f"\nBuilt arrays:")
        print(f"ca_ shape: {ca_.shape}")
        print(f"Number of coefficients: {len(ca_ind)}")
        print(f"First 10 coefficient names: {list(ca_ind.keys())[:10]}")
    
    return ca_, ca_ind

def closest_model_low(image_data, lut_data):
    pixels = image_data.T
    indices = np.zeros((pixels.shape[0],), dtype=np.uint8)
    i_min = np.zeros((pixels.shape[0],), dtype=np.float32)+1e10
    for i in range(lut_data.shape[1]):
        lut = lut_data[:, i]
        distances = np.sum((pixels - lut) ** 2, axis=1)
        i_min = np.minimum(i_min, distances)
        f = np.where(i_min == distances)
        indices[f] = i

    return indices

def regular_interp(da, tgt_lat, tgt_lon, tgt_time=None):
    """
    Fast interpolation of a DataArray defined on a REGULAR (lat, lon[, time])
    grid onto target lat/lon points, optionally at a single time.

    MERRA2 and the monthly climatology are on regular spatial+temporal grids, so
    we use scipy.RegularGridInterpolator on plain numpy arrays. This avoids the
    dask task-graph / per-pixel weight-location overhead of the generic
    core.interpolate.interp (which dominated the per-tile cost).
    """
    lat = np.asarray(da['lat'].values, dtype=np.float64)
    lon = np.asarray(da['lon'].values, dtype=np.float64)
    tgt_lat = np.asarray(tgt_lat, dtype=np.float64)
    tgt_lon = np.asarray(tgt_lon, dtype=np.float64)
    shape = tgt_lat.shape

    use_time = ('time' in da.dims) and (tgt_time is not None)
    if 'time' in da.dims:
        da = da.transpose('time', 'lat', 'lon')
    else:
        da = da.transpose('lat', 'lon')
    vals = np.asarray(da.values, dtype=np.float64)

    # RegularGridInterpolator requires strictly ascending axes.
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        vals = vals[..., ::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        vals = vals[..., :, ::-1]

    flat_lat = tgt_lat.ravel()
    flat_lon = tgt_lon.ravel()
    nanmask = ~(np.isfinite(flat_lat) & np.isfinite(flat_lon))
    if nanmask.any():
        flat_lat = flat_lat.copy()
        flat_lon = flat_lon.copy()
        flat_lat[nanmask] = lat[0]
        flat_lon[nanmask] = lon[0]

    if use_time:
        t0 = da['time'].values[0]
        tsrc = (da['time'].values - t0) / np.timedelta64(1, 's')
        ttgt = (np.datetime64(tgt_time) - t0) / np.timedelta64(1, 's')
        rgi = RegularGridInterpolator((tsrc, lat, lon), vals,
                                      method='linear', bounds_error=False, fill_value=None)
        pts = np.empty((flat_lat.size, 3), dtype=np.float64)
        pts[:, 0] = float(ttgt)
        pts[:, 1] = flat_lat
        pts[:, 2] = flat_lon
    else:
        if vals.ndim == 3:      # singleton time (monthly climatology)
            vals = vals[0]
        rgi = RegularGridInterpolator((lat, lon), vals,
                                      method='linear', bounds_error=False, fill_value=None)
        pts = np.empty((flat_lat.size, 2), dtype=np.float64)
        pts[:, 0] = flat_lat
        pts[:, 1] = flat_lon

    out = rgi(pts)
    if nanmask.any():
        out[nanmask] = np.nan
    return out.astype(np.float32).reshape(shape)


def get_aer_interpolated(dsAER: xr.Dataset, latitude, longitude, date_time=None) -> xr.Dataset:
    """
    Interpolate aerosol data (regular grid) to the target lat/lon points.
    """
    vars_ = ["TOTEXTTAU", "SU_FRAC", "DU_FRAC", "OC_FRAC", "SS_FRAC", "BC_FRAC"]
    lat_v = np.asarray(latitude.values if hasattr(latitude, 'values') else latitude)
    lon_v = np.asarray(longitude.values if hasattr(longitude, 'values') else longitude)
    tv = None
    if date_time is not None:
        tv = np.datetime64(date_time.values) if hasattr(date_time, 'values') else np.datetime64(date_time)
    interp_vars = {}
    for v in vars_:
        interp_vars[v] = (('y', 'x'), regular_interp(dsAER[v], lat_v, lon_v, tv))
    return xr.Dataset(interp_vars)


_pre_aer_models_cache = {}


def pre_aer_models(faer, match):
    """
    Read MERRA2/CAMS aerosols components fraction of the aerosol models
    """
    cache_key = (faer, tuple(match.keys()))
    if cache_key in _pre_aer_models_cache:
        return _pre_aer_models_cache[cache_key]
    rh = {'sulf': 80., 'dust': 80., 'oc': 80., 'ssalt': 80., 'bc': 0.}
    Ha = {'sulf': 8., 'dust': 2., 'oc': 8., 'ssalt': 1., 'bc': 8.}
    f = open(faer, 'r')
    frac_aer_model = {}
    for key in match.keys():
        f.readline()
        line = f.readline()
        frac_aer_model[key] = np.array(line.split()).astype(np.float32)
    f.close()

    _pre_aer_models_cache[cache_key] = (frac_aer_model, rh, Ha)
    return frac_aer_model, rh, Ha

def get_iaer(data):
    """
    Calculate aerosol model index from MERRA2/CAMS data.
    """
    faer = config.aerosol_model_fraction
    match = {'sulf': 'SU', 'dust': 'DU', 'oc': 'OC', 'ssalt': 'SS', 'bc': 'BC'}
    match2 = {'sulf': 'SU_FRAC', 'dust': 'DU_FRAC',
              'oc': 'OC_FRAC', 'ssalt': 'SS_FRAC', 
              'bc': 'BC_FRAC'}
    # Load aerosol models
    frac_aer_model, _rh_aer_model, _Ha_aer_model = pre_aer_models(faer, match)
    # Get shape and flatten TOTEXTTAU once
    totexttau = data["TOTEXTTAU"]
    shp = totexttau.shape
    totexttau_flat = totexttau.values.reshape(-1)

    # Pre-allocate and compute all divisions in one go
    keys_list = list(frac_aer_model.keys())

    # Stack all aerosol data first, then divide all at once
    aer_data = np.stack([data[match2[key]].values.flatten() for key in keys_list], axis=0)
    xm = aer_data / totexttau_flat  # Vectorized division for all models
    
    # Stack xb
    xb = np.stack([frac_aer_model[key] for key in keys_list], axis=0, dtype=np.float32)
    # Find closest model
#    result = closest_model(xm, xb).reshape(shp)
    result = closest_model_low(xm, xb).reshape(shp) #.astype(np.float32)
#    result = closest_model_pixelwise(xm, xb).reshape(shp) #.astype(np.float32)
    
    return result


def get_mensual_faers(date):
    yyyymm = "".join(str(date).split("-")[:2])
    base = config.amip_path + "/m2amip01.tavgM_2d_aer_Nx.*.nc4"
    # find the closest date in the available files
    filenames = glob(base)
    dates = [int(f.split('.')[-2]) for f in filenames]
    diff_dates = np.argmin([abs(d - int(yyyymm)) for d in dates])
    yyyymm = dates[diff_dates]
    base = config.amip_path + "/m2amip{}.tavgM_2d_aer_Nx.{}.nc4"
    paths = []
    for i in range(1, 11):
        path = base.format(f"{i:02}", yyyymm)
        paths.append(path)
    return paths

def preload_monthly_aerosol(date_time):
    """
    Open, select, rename and load into memory (once) the monthly MERRA2 aerosol
    datasets so they can be reused across all tiles instead of being re-read
    from disk for every tile. The returned datasets are small global grids and
    are independent of the number of tiles.
    """
    if int(str(date_time.values)[:4]) <= 2017:
        date = str(date_time.values)[0:4]+str(date_time.values)[5:7]
    else:
        date = "2017"+str(date_time.values)[5:7]
    datasets = []
    for aer_path in get_mensual_faers(date):
        dsMensualAER = xr.open_dataset(aer_path, chunks={"lat" : -1, "lon" : -1, "time" : 1})
        dsMensualAER = dsMensualAER[["TOTEXTTAU", "SUEXTTAU", "DUEXTTAU",
                                     "OCEXTTAU", "SSEXTTAU", "BCEXTTAU"]].astype(np.float32)
        dsMensualAER = dsMensualAER.rename({"BCEXTTAU":"BC_FRAC", "DUEXTTAU":"DU_FRAC","SSEXTTAU":"SS_FRAC","OCEXTTAU":"OC_FRAC","SUEXTTAU":"SU_FRAC"})
        datasets.append(dsMensualAER.compute())
    return datasets

#@memory_tracker
def calculate_monthly_aerosol(
    date_time, 
    latitude, 
    longitude,
    datasets=None,
):

    # ``datasets`` may be preloaded once (see preload_monthly_aerosol) to avoid
    # re-reading the 10 monthly aerosol files from disk for every tile.
    if datasets is None:
        datasets = preload_monthly_aerosol(date_time)

    mensual_iaer = []
    for dsMensualAER in datasets:
        # Interpolate the aerosol data to the observation's geometry
        mensual_faer = get_aer_interpolated(
            dsMensualAER,
            latitude,
            longitude,
            None # Time interpolation is not used here for monthly data
        ) #.transpose()

        mensual_iaer.append(get_iaer(mensual_faer))

    # Stack the per-month model indices into (nmonths, y, x).
    return np.stack(mensual_iaer, axis=0)

def _load_or_compute_terrain(
    latitude,
    longitude,
    dsDEM,
    spatial_resolution_m=1000.0,
    slope_method="Horn",
    gaussian_sigma=0.0,
):
    """Load cached terrain data or compute if not available."""
#    file_name = Path(config.kept_data_path) / "slope" / \
#                f"{str(latitude.y[0].values)[:10]}_{str(longitude.x[0].values)[:10]}_slope.nc"
#    
#    if file_name.exists():
#        return xr.open_dataset(file_name).chunk({"y" : longitude.chunksizes['y'][0], "x" : longitude.chunksizes['x'][0]})
    
    # Compute terrain data
#    file_name.parent.mkdir(parents=True, exist_ok=True)
    
    elev = interp(dsDEM["elev"], lat=Linear(latitude), lon=Linear(longitude))
    elev_arr = np.asarray(elev.values, dtype=np.float32)

    valid = np.isfinite(elev_arr)
    elev_fill = _nearest_valid_fill(elev_arr, valid).astype(np.float32)

    sigma = float(gaussian_sigma)
    if sigma > 0.0:
        elev_proc = gaussian_filter(elev_fill, sigma=sigma).astype(np.float32)
    else:
        elev_proc = elev_fill

    method = str(slope_method)
    slope = xdem_terrain.slope(
        elev_proc,
        resolution=float(spatial_resolution_m),
        method=method,
    )
    aspect = xdem_terrain.aspect(elev_proc, method=method)
    slope = slope.astype(np.float32)
    aspect = aspect.astype(np.float32)
    # Preserve DEM no-data mask; derivative on filled values can otherwise
    # create synthetic circular artefacts around invalid regions.
    slope[~valid] = np.nan
    aspect[~valid] = np.nan
    
    # Cache results
    terrain_ds = xr.Dataset(
        {
#            "elev": (["y", "x"], elev.transpose().data.astype(np.float32)),
            "elev": (["y", "x"], elev.data.astype(np.float32)),
            "slope": (["y", "x"], slope),
            "aspect": (["y", "x"], aspect),
        },
        coords={"y": elev.y, "x": elev.x}
    )
    #terrain_ds.to_netcdf(file_name)
    
    return terrain_ds


def compute_delta_elevation_from_elev(
    elev,
    spatial_resolution_m,
    geolocation_error_pixels=0.5,
    slope_window=3,
):
    """
    Compute delta elevation from elevation and local maximum slope.

    The altitude error is estimated from a geolocation error expressed in
    pixels and a local maximum terrain slope:
    delta_z = (geolocation_error_pixels * spatial_resolution_m) * tan(slope_max)
    """
    elev_arr = np.asarray(elev, dtype=np.float32)
    valid = np.isfinite(elev_arr)
    if not np.any(valid):
        return np.full(elev_arr.shape, np.nan, dtype=np.float32)

    filled = _nearest_valid_fill(elev_arr, valid).astype(np.float32)

    slope_deg = xdem_terrain.slope(filled, resolution=float(spatial_resolution_m)).astype(np.float32)
    slope_tan = np.tan(np.radians(np.clip(slope_deg, 0.0, 89.9))).astype(np.float32)
    slope_window = max(1, int(slope_window))
    slope_max = maximum_filter(slope_tan, size=slope_window, mode="nearest")

    geolocation_error_m = float(geolocation_error_pixels) * float(spatial_resolution_m)
    delev = geolocation_error_m * slope_max
    delev = delev.astype(np.float32)
    delev[~valid] = np.nan
    return delev


def _kg_multiplier_from_zone_code(zone_code):
    code = (zone_code or "").strip()
    if code in {"Ocean", "EF", "ET"}:
        return 0.7
    if code.startswith("B"):
        return 1.5
    if code.startswith("A"):
        return 1.4
    if code.startswith("C") or code.startswith("D"):
        return 1.3
    return 1.0


def load_koppen_geiger_zone_multipliers(mapping_file):
    """Load zone-number to AOD uncertainty multiplier mapping."""
    zone_to_multiplier = {}
    with open(mapping_file, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            zone_num = int(row["zoneNum"])
            zone_code = row.get("kg_zone", "")
            zone_to_multiplier[zone_num] = _kg_multiplier_from_zone_code(zone_code)
    return zone_to_multiplier


def sample_koppen_geiger_multiplier(lat, lon, kg_dataset, zone_to_multiplier, default=1.0):
    """
    Sample Köppen-Geiger classes at lat/lon and convert them to multipliers.
    """
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    out = np.full(lat_arr.shape, float(default), dtype=np.float32)

    valid = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    if not np.any(valid):
        return out

    if rasterio is None or kg_dataset is None:
        return out

    rows, cols = rasterio.transform.rowcol(
        kg_dataset.transform,
        lon_arr[valid],
        lat_arr[valid],
        op=np.floor,
    )
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    inside = (
        (rows >= 0) & (rows < kg_dataset.height) &
        (cols >= 0) & (cols < kg_dataset.width)
    )
    if not np.any(inside):
        return out

    r0 = int(rows[inside].min())
    r1 = int(rows[inside].max())
    c0 = int(cols[inside].min())
    c1 = int(cols[inside].max())

    window = rasterio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
    zones_window = kg_dataset.read(1, window=window)
    z_rows = rows[inside] - r0
    z_cols = cols[inside] - c0
    zone_values = zones_window[z_rows, z_cols].astype(np.int64)

    sampled = np.fromiter(
        (zone_to_multiplier.get(int(z), float(default)) for z in zone_values),
        dtype=np.float32,
        count=zone_values.size,
    )

    valid_flat = np.flatnonzero(valid)
    target_flat = valid_flat[inside]
    out_flat = out.ravel()
    out_flat[target_flat] = sampled
    return out

def _stack_viewing_angles(vnir_angle, swir_angle, n_bands):
    # For 4-band products, SWIR geometry applies to the last band only.
    n_bands = int(n_bands)
    n_vnir = max(0, n_bands - 1)
    return np.stack([vnir_angle] * n_vnir + [swir_angle])

def slope_err(ds, slope, aspect, TOTEXTTAU, pression, ca_, ca_ind, iaer):
    """
    Calculate error from terrain slope and aspect.
    """
    mtetas = ds.SZA.data.astype(np.float32).ravel()#.compute()
    mtetav = _stack_viewing_angles(ds.VZA.data, ds.VZA_IR.data, len(ds.attrs["bands"])).reshape(len(ds.attrs["bands"]), -1)#.compute()
    mphis = ds.SAA.data.astype(np.float32).ravel()#.compute()
    maspect = aspect.data.ravel()#.compute()
    mslope = slope.data.ravel()#.compute()
    mpression = pression.ravel()
    mTOTEXTTAU = TOTEXTTAU.ravel()
    miaer = iaer.ravel()
    
    shape = ds.SZA.data.shape
    
    slope_errs = []

    for b in range(mtetav.shape[0]) :
        mus = np.cos(np.radians(mtetas))
        mu = np.cos(np.radians(mslope))
        mui = mus * mu + np.sqrt(1 - mus ** 2) * np.sqrt(1 - mu ** 2) * np.cos(np.radians(mphis - maspect))
        fsky = (1 + mu) / 2.
        fground = (1 - mu) / 2.
        
        # Prepare CA data
        ca_band = ca_[b, :, :]
        ca_iaer = ca_band[miaer, :]
        ca_final = np.moveaxis(ca_iaer, -1, 0)
        ca_final = ca_final.reshape(ca_final.shape[0], -1)
        
        # Compute atmospheric transmissions
        T, Tdir, Tdif = compute_atmospheric_transmissions(
            np.cos(np.radians(mtetas)),
            mTOTEXTTAU, 
            mpression, 
            ca_final, 
            ca_ind
        )

        # Calculate error
        # Precompute denominator components
        # When surface faces away from sun (mui <= 0), direct term contributes
        # nothing. We also clamp to >= 0 for numerical robustness.
        direct_raw = np.where(mui > 0, Tdir * mui / mus, 0.0)
        term1 = np.maximum(direct_raw, 0.0)

        term2 = Tdif * fsky

        term3 = T * fground * 0.3

        # Single division operation
        err = T / (term1 + term2 + term3)

        slope_errs.append(err.reshape((shape)))

    return np.array(slope_errs)

#@memory_tracker
def get_slope_err(ds_probav, TOTEXTTAU, pression, ca_, ca_ind, iaer, chunksize, dem_ds=None,
                   wide_lat=None, wide_lon=None, crop_offset=None):
    # ``dem_ds`` may be an already-open (lazy) DEM dataset preloaded once to
    # avoid re-opening the file for every tile. Cropping + compute stays
    # per-tile so peak memory remains bounded.
    #
    # ``wide_lat``/``wide_lon`` (optional): a WIDER lat/lon window than
    # ``ds_probav``'s own (typically 1-pixel-halo) grid, used only for the
    # slope/aspect derivative. This matters when ``terrain_gaussian_sigma`` > 0:
    # a Gaussian pre-smoothing needs several pixels of spatial context on every
    # side to be accurate, but the per-tile halo elsewhere in the pipeline is
    # only 1 pixel. Computing the smoothing on a too-narrow per-tile window
    # creates a spurious discontinuity at every tile boundary (a periodic
    # stripe pattern), because each tile's edge is (incorrectly) treated as if
    # it were the edge of the whole scene. When given, ``crop_offset`` (row,
    # col) gives the position of ``ds_probav``'s own window inside the wider
    # array, and the computed slope/aspect are cropped back down to
    # ``ds_probav``'s extent before use, so the terrain-effect formula still
    # operates on arrays of the expected (small) tile shape.
    if dem_ds is None:
        dem_ds = xr.open_dataset(config.dem_path, chunks="auto")

    lat_for_terrain = wide_lat if wide_lat is not None else ds_probav.lat
    lon_for_terrain = wide_lon if wide_lon is not None else ds_probav.lon
    crop_y = ds_probav.y if wide_lat is None else wide_lat.y

    dsDEM = xrcrop(dem_ds, lat = crop_y).chunk({"lat" : chunksize, "lon" : chunksize})
    dsDEM = dsDEM.compute()
    terrain_data = _load_or_compute_terrain(
        lat_for_terrain,
        lon_for_terrain,
        dsDEM,
        spatial_resolution_m=float(config.__dict__.get('sensor_spatial_resolution_m', 1000.0)),
        slope_method=str(config.__dict__.get('terrain_slope_method', 'Horn')),
        gaussian_sigma=float(config.__dict__.get('terrain_gaussian_sigma', 0.0)),
    )


#    elev = terrain_data["elev"]
#    delev = terrain_data["delev"]
    slope = terrain_data["slope"]
    aspect = terrain_data["aspect"]

    if crop_offset is not None:
        oy, ox = crop_offset
        ny = ds_probav.sizes['y']
        nx = ds_probav.sizes['x']
        slope = slope.isel(y=slice(oy, oy + ny), x=slice(ox, ox + nx))
        aspect = aspect.isel(y=slice(oy, oy + ny), x=slice(ox, ox + nx))

    # Also return the local terrain slope (degrees) used to derive the terrain
    # uncertainty term, so callers can save it for debugging (e.g. when the
    # uncertainty_from_terrain contribution looks too large).
    slope_deg = slope.data.astype(np.float32)

    return slope_err(ds_probav, slope, aspect, TOTEXTTAU, pression, ca_, ca_ind, iaer), slope_deg