import xarray as xr
import numpy as np
from pathlib import Path
import xdem.terrain as xdem_terrain
#import config
import probav_vito as probav
from core.interpolate import interp, Linear
from core.tools import xrcrop
from scipy.signal import convolve2d
from scipy.interpolate import RegularGridInterpolator
import logging
import datetime
import functools
import psutil
import os
from time import time
import gc
from glob import glob

class config_class:
    def __init__(self):
        pass

    def set(self, name, value):
        self.__dict__[name] = value

    def getfloat(self, part, name):
        assert(name in self.__dict__.keys())
        return self.__dict__[name]


config = config_class()

def setup_logger():
    """Configure le logger avec un format plus détaillé"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(f'memory_usage_{datetime.datetime.now():%Y%m%d_%H%M}.log'),
            logging.StreamHandler()
        ]
    )

def memory_tracker(func):
    """Décorateur pour suivre l'utilisation de la mémoire d'une fonction"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pc = psutil.Process(os.getpid())
        
        # Mesure avant exécution
        mem_before = pc.memory_info().rss / 1024 / 1024  # En MB
        start_time = time()
        peak_memory = mem_before
        
        # Fonction pour mesurer la mémoire pendant l'exécution
        def get_current_memory():
            return pc.memory_info().rss / 1024 / 1024
        
        try:
            # Exécution de la fonction
            result = func(*args, **kwargs)
            
            # Mesure finale
            peak_memory = max(peak_memory, get_current_memory())
            end_time = time()
            mem_after = get_current_memory()
            
            # Calcul des différences
            duration = end_time - start_time
            mem_diff = mem_after - mem_before
            
            logging.info(
                f"\n{'='*50}\n"
                f"Function: {func.__name__}\n"
                f"Memory before: {mem_before:,.2f} MB\n"
                f"Memory after: {mem_after:,.2f} MB\n"
                f"Peak memory: {peak_memory:,.2f} MB\n"
                f"Memory used: {mem_diff:,.2f} MB\n"
                f"Duration: {duration:.2f} seconds\n"
                f"{'='*50}"
            )
            
            return result
            
        except Exception as e:
            # En cas d'erreur, on log quand même l'utilisation mémoire
            end_time = time()
            mem_after = get_current_memory()
            peak_memory = max(peak_memory, mem_after)
            
            logging.error(
                f"\n{'='*50}\n"
                f"Function: {func.__name__} (FAILED)\n"
                f"Error: {str(e)}\n"
                f"Memory before: {mem_before:,.2f} MB\n"
                f"Memory after: {mem_after:,.2f} MB\n"
                f"Peak memory: {peak_memory:,.2f} MB\n"
                f"Duration: {end_time - start_time:.2f} seconds\n"
                f"{'='*50}"
            )
            raise
            
    return wrapper

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

def shift_lon_sat_to_360(lon_sat):
    """Convert satellite longitude values from a mixed [-180, 180] range
    to a consistent [0, 360] range.

    The function examines the input ``lon_sat`` array.  If the array
    contains both negative and positive values (i.e. a mix of western
    and eastern longitudes), it assumes the data are in the standard
    geographic convention where longitudes span from -180° to +180°.
    In that case the function shifts all negative values by +360° so
    that the resulting array contains only values in the 0–360° range.
    This is useful when the satellite data need to be merged with
    other datasets that use the 0–360° convention.

    Parameters
    ----------
    lon_sat : xarray.DataArray or array‑like
        Satellite longitude values.  The function operates on the
        underlying NumPy array via ``lon_sat.values``.

    Returns
    -------
    numpy.ndarray
        The longitude array with all negative values shifted by +360°.
        If the input array already contains only non‑negative values,
        it is returned unchanged.
    """
    # If the satellite longitudes span both negative and positive
    # values, shift the negative part by +360 to obtain a 0–360 range.
    lon_max = np.nanmax(lon_sat)
    lon_min = np.nanmin(lon_sat)
    if lon_max > 0 and lon_min < 0 and lon_max-lon_min > 300:
        # Create a copy to avoid modifying the original array.
        new_lon_sat = lon_sat.values.copy()
        # Add 360 to all negative longitudes.
        new_lon_sat[new_lon_sat < 0] += 360
        return xr.DataArray(new_lon_sat)
    # No conversion needed; return the original array.
    return xr.DataArray(lon_sat)

def shift_lon_to_180(ds):
    """Convert longitudes from a 0–360° range back to the standard
    [-180, 180] range.

    The function inspects the longitude coordinate of the provided
    :class:`xarray.Dataset`.  If the maximum longitude value exceeds
    180°, it assumes the dataset uses a 0–360° convention and
    subtracts 360° from all values greater than 180°.  The updated
    coordinate is then assigned back to the dataset and the coordinates
    are sorted.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing a longitude coordinate named ``lon`` or
        ``x``.

    Returns
    -------
    xarray.Dataset
        The dataset with longitudes converted to the [-180, 180] range.
    """
    # Determine which coordinate holds the longitude values.
    if 'x' in ds.dims:
        lon = ds.x
    else:
        lon = ds.lon
    # If the dataset uses a 0–360° convention, shift values > 180°.
    if np.nanmax(lon) > 180:
        # Create a copy to avoid modifying the original array.
        old_lon = lon.values.copy()
        old_lon -= 360
        ds = ds.assign_coords(lon=old_lon)
        ds = ds.sortby('lon')
    return ds

def build_flag(ds_input, ds_output, config_data):
#    flag_aot = ds_input.rtoa.max(dim="bands") >= config_data.getfloat("Coefficients", "aotmax")
    flag_aot = np.max(ds_input['TOA'], axis=0) >= config_data.getfloat("Coefficients", "aotmax")
#    flag_toc_min = ds_output.rtoc_run.min(dim="bands") < config_data.getfloat("Coefficients", "tocmin")
    flag_toc_min = np.min(ds_output['rTOC'], axis=0) < config_data.getfloat("Coefficients", "tocmin")
#    flag_toc_max = ds_output.rtoc_run.max(dim="bands") > config_data.getfloat("Coefficients", "tocmax")
    flag_toc_max = np.max(ds_output['rTOC'], axis=0) > config_data.getfloat("Coefficients", "tocmax")
#    flag_sza_max = ds_input.tetas > config_data.getfloat("Coefficients", "szamax")
    flag_sza_max = ds_input['SZA'] > config_data.getfloat("Coefficients", "szamax")
    flag_aot_grad = ds_output['aod_grad'] > config_data.getfloat("Coefficients", "aod_max_grad")
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

def compute_atmospheric_transmissions(cos_sun, cos_view,
                                    aot_550, pressure_eq,
                                    smac_coeffs, ca_ind):
    """
    Compute atmospheric transmission components for SMAC.
    
    Args:
        cos_sun: Cosine of solar zenith angle
        cos_view: Cosine of view zenith angle
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

def closest_model(image_data, lut_data):
    # distance compute https://www.mdpi.com/2227-7390/12/23/3787 3.2.5. Parallelization as of 21/07/2025
    pixels = image_data.T 
    lut = lut_data  
    lut_norms = np.sum(lut.T ** 2, axis=1) 
    pixel_norms = np.sum(pixels ** 2, axis=1)[:, np.newaxis] 
    dot_products = pixels @ lut
    distances = pixel_norms + lut_norms - 2 * dot_products 
    
    indices = np.argmin(distances, axis=1).astype(np.uint8)

    return indices

def closest_model_optimized(image_data, lut_data):
    pixels = image_data.T 
    lut = lut_data  
    lut_norms = np.sum(lut.T ** 2, axis=1) 
#    pixel_norms = np.sum(pixels ** 2, axis=1)[:, np.newaxis] 
    dot_products = pixels @ lut
    distances = lut_norms - 2 * dot_products 
    
    indices = np.argmin(distances, axis=1).astype(np.uint8)

    return indices

def closest_model_pixelwise(image_data, lut_data):
    pixels = image_data.T 
#    lut = lut_data.astype(np.float32)
    lut_norms = np.sum(lut_data.T ** 2, axis=1) 
    nb_pixels = pixels.shape[0]
    indices = np.zeros((nb_pixels), dtype=np.uint8)
    for i in range(nb_pixels):
        pixel = pixels[i]
        pixel_norm = np.sum(pixel ** 2).astype(np.float32)
        dot_products = pixel @ lut_data
        distances = pixel_norm + lut_norms - 2 * dot_products 
        indices[i] = np.argmin(distances).astype(np.uint8)

    return indices

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
    frac_aer_model, rh_aer_model, Ha_aer_model = pre_aer_models(faer, match)
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

#date_time = dsProbav["mean-time"]
# latitude = dsProbav.lat
# longitude = dsProbav.lon

def open_monthly_aerosol(date_time):
    print("Loading monthly aerosol data...")
    if int(str(date_time.values)[:4]) <= 2017:
        date = str(date_time.values)[0:4]+str(date_time.values)[5:7]
    else:
        date = "2017"+str(date_time.values)[5:7]
    dsMensualAERs = []
    for aer_path in get_mensual_faers(date): 
        dsMensualAER = xr.open_dataset(aer_path, chunks={"lat" : -1, "lon" : -1, "time" : 1})
        dsMensualAER = dsMensualAER[["TOTEXTTAU", "SUEXTTAU", "DUEXTTAU", 
                                     "OCEXTTAU", "SSEXTTAU", "BCEXTTAU"]].astype(np.float32)
        dsMensualAERs.append(dsMensualAER)
    return dsMensualAERs

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

def _load_or_compute_terrain(latitude, longitude, dsDEM):
    """Load cached terrain data or compute if not available."""
#    file_name = Path(config.kept_data_path) / "slope" / \
#                f"{str(latitude.y[0].values)[:10]}_{str(longitude.x[0].values)[:10]}_slope.nc"
#    
#    if file_name.exists():
#        return xr.open_dataset(file_name).chunk({"y" : longitude.chunksizes['y'][0], "x" : longitude.chunksizes['x'][0]})
    
    # Compute terrain data
#    file_name.parent.mkdir(parents=True, exist_ok=True)
    
    elev = interp(dsDEM["elev"], lat=Linear(latitude), lon=Linear(longitude))
    delev = interp(dsDEM["Delev"], lat=Linear(latitude), lon=Linear(longitude))
    slope = xdem_terrain.slope(elev.values, resolution=10)
    aspect = xdem_terrain.aspect(elev.values)
    slope = slope.astype(np.float32)
    aspect = aspect.astype(np.float32)
    
    # Cache results
    terrain_ds = xr.Dataset(
        {
#            "elev": (["y", "x"], elev.transpose().data.astype(np.float32)),
#            "delev": (["y", "x"], delev.transpose().data.astype(np.float32)),
            "elev": (["y", "x"], elev.data.astype(np.float32)),
            "delev": (["y", "x"], delev.data.astype(np.float32)),
            "slope": (["y", "x"], slope),
            "aspect": (["y", "x"], aspect),
        },
        coords={"y": elev.y, "x": elev.x}
    )
    #terrain_ds.to_netcdf(file_name)
    
    return terrain_ds

def _stack_viewing_angles(vnir_angle, swir_angle):
    return np.stack([vnir_angle, vnir_angle, swir_angle, swir_angle])

def slope_err(ds, slope, aspect, TOTEXTTAU, pression, ca_, ca_ind, iaer):
    """
    Calculate error from terrain slope and aspect.
    """
    mtetas = ds.SZA.data.astype(np.float32).ravel()#.compute()
    mtetav = _stack_viewing_angles(ds.VZA.data,ds.VZA_IR.data).reshape(len(ds.attrs["bands"]), -1)#.compute()
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
            np.cos(np.radians(mtetav[b])),
            mTOTEXTTAU, 
            mpression, 
            ca_final, 
            ca_ind
        )

        # Calculate error
        # Precompute denominator components
        term1 = Tdir * mui / mus

        term2 = Tdif * fsky

        term3 = T * fground * 0.3

        # Single division operation
        err = T / (term1 + term2 + term3)

        slope_errs.append(err.reshape((shape)))

    return np.array(slope_errs)

#@memory_tracker
def get_slope_err(ds_probav, TOTEXTTAU, pression, ca_, ca_ind, iaer, chunksize, dem_ds=None):
    # ``dem_ds`` may be an already-open (lazy) DEM dataset preloaded once to
    # avoid re-opening the file for every tile. Cropping + compute stays
    # per-tile so peak memory remains bounded.
    if dem_ds is None:
        dem_ds = xr.open_dataset(config.dem_path, chunks="auto")
    dsDEM = xrcrop(dem_ds, lat = ds_probav.y).chunk({"lat" : chunksize, "lon" : chunksize})
    dsDEM = dsDEM.compute()
    terrain_data = _load_or_compute_terrain(ds_probav.lat, ds_probav.lon, dsDEM)

#    elev = terrain_data["elev"]
#    delev = terrain_data["delev"]
    slope = terrain_data["slope"]
    aspect = terrain_data["aspect"]

    return slope_err(ds_probav, slope, aspect, TOTEXTTAU, pression, ca_, ca_ind, iaer)

def convolution_err(input_data: np.ndarray, size: int = 5, sigma: float = 0.9) -> np.ndarray:
    """
    Calculate error from adjacent pixels using Gaussian convolution.
    Uses a 5x5 matrix filled with a Gaussian at 99% at 2.5 pixels then normalized.
    """
    ax = np.arange(size) - (size // 2)
    xx, yy = np.meshgrid(ax, ax)
    kernel = (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    convolved = convolve2d(np.where(np.isnan(input_data), 0, input_data), kernel, mode='same', boundary='fill')
    
    return np.abs(input_data - convolved)

if __name__ == '__main__':
#    probav_folder = config.probav_path
    k_p0 = 1e-2

    probav_folder = '/mnt/ceph/proj/FDR4VGT/input/probav'
    dsProbav = probav.read_ProbaV(probav_folder, 1000).isel(y=slice(1000, 1100), x=slice(1000,1100))

#    coef_path = Path(config.smac_coef_path) / f"{config.sensor}_smac_coeffs_v3.0.npy"
#    ca_, ca_ind = read_smac_coefficients(coef_path)

    latitude = dsProbav.lat
    longitude = dsProbav.lon
    date_time = dsProbav["mean-time"]
    SHAPE = (len(latitude.y), len(latitude.x))
    bands_count = 4

#    aer_file = f"{config.path_aer}/MERRA2_400.tavg1_2d_aer_Nx.20141111.nc4"
#    slv_file = f"{config.path_slv}/MERRA2_400.tavg1_2d_slv_Nx.20141111.nc4"
    slv_file = '/archive2/data/MERRA2/surf_pression_water_vapor/2014/MERRA2_400.tavg1_2d_slv_Nx.20141111.nc4'

#    dsAER =  xr.open_dataset(aer_file, chunks={"lat" : -1, "lon" : -1, "time" : 1})
#    dsAER = dsAER[["TOTEXTTAU", "SUEXTTAU", "DUEXTTAU", "OCEXTTAU", "SSEXTTAU", "BCEXTTAU"]].compute()
#    faer = get_aer_interpolated(dsAER, Linear(latitude), Linear(longitude), Linear(date_time)).transpose()

    dsSLV = xr.open_dataset(slv_file, chunks={"lat" : -1, "lon" : -1, "time" : 1})
    dsSLV = dsSLV[["TQV", "TO3", "SLP"]].compute()
    # uh2o = interp(dsSLV["TQV"] * config.k_uh2o, lat=Linear(latitude), lon=Linear(longitude), time=Linear(date_time)).T
    # uo3 = interp(dsSLV["TO3"] * config.k_uo3, lat=Linear(latitude), lon=Linear(longitude), time=Linear(date_time)).T
    pression = interp(dsSLV["SLP"] * k_p0, lat=Linear(latitude), lon=Linear(longitude), time=Linear(date_time)).T

    iaer_month, mean_totex_month, std_totex_month = calculate_monthly_aerosol(date_time, latitude, longitude)

    iaer_run = iaer_month[0] #FOR THE EXAMPLE

    err = get_slope_err(dsProbav, faer.TOTEXTTAU.values, pression.values, ca_, ca_ind, iaer_run)

    #convolution_err(rtoc run (band by band))
    print()