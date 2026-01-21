"""
CF-1.8 Compliance Utilities for FDR4VGT

Functions to apply CF-compliant attributes to xarray Datasets
using JSON configuration files.
"""

import json
import xarray as xr
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def load_cf_attributes(json_path: Union[str, Path]) -> dict:
    """
    Load CF attributes from a JSON file.
    
    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file containing CF attributes
        
    Returns
    -------
    dict
        Dictionary containing CF attributes
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def apply_cf_attributes_from_json(
    ds: xr.Dataset,
    json_path: Union[str, Path],
    output_path: Optional[str] = None,
    add_history: bool = True
) -> xr.Dataset:
    """
    Apply CF-1.8 compliant attributes to a dataset using a JSON configuration.
    
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to make CF-compliant
    json_path : str or Path
        Path to the JSON file containing CF attributes
    output_path : str, optional
        Path to save the CF-compliant file. If None, file is not saved.
    add_history : bool, default True
        Whether to add a history timestamp to global attributes
        
    Returns
    -------
    xr.Dataset
        CF-compliant copy of the dataset
    """
    # Load CF attributes from JSON
    cf_config = load_cf_attributes(json_path)
    
    # Create a deep copy
    ds_cf = ds.copy(deep=True)
    
    # Apply global attributes
    global_attrs = cf_config.get('global_attributes', {}).copy()
    if add_history:
        global_attrs['history'] = f'{datetime.utcnow().isoformat()}Z: Applied CF-1.8 attributes'
    ds_cf.attrs.update(global_attrs)
    
    # Apply variable attributes
    var_attrs = cf_config.get('variables', {})
    for var_name, attrs in var_attrs.items():
        if var_name in ds_cf.data_vars or var_name in ds_cf.coords:
            # Filter out None values
            clean_attrs = {k: v for k, v in attrs.items() if v is not None}
            ds_cf[var_name].attrs.update(clean_attrs)
    
    # Apply coordinate attributes
    coord_attrs = cf_config.get('coordinates', {})
    for coord_name, attrs in coord_attrs.items():
        if coord_name in ds_cf.dims:
            # Create coordinate if it doesn't exist
            if coord_name not in ds_cf.coords:
                if coord_name == 'bands':
                    ds_cf = ds_cf.assign_coords(bands=('bands', list(range(ds_cf.dims['bands']))))
            if coord_name in ds_cf.coords:
                clean_attrs = {k: v for k, v in attrs.items() if v is not None}
                ds_cf[coord_name].attrs.update(clean_attrs)
    
    # Save if output path provided
    if output_path:
        # Get encoding configuration
        encoding_config = cf_config.get('encoding', {})
        default_encoding = encoding_config.get('default', {})
        
        encoding = {}
        for var in ds_cf.data_vars:
            dtype = ds_cf[var].dtype
            if dtype in ['float32', 'float64']:
                encoding[var] = default_encoding.copy()
            elif dtype in ['int32', 'int64', 'uint8', 'uint32']:
                encoding[var] = {
                    'dtype': str(dtype),
                    'zlib': default_encoding.get('zlib', True),
                    'complevel': default_encoding.get('complevel', 4),
                }
        
        ds_cf.to_netcdf(output_path, encoding=encoding)
        print(f"CF-compliant file saved to: {output_path}")
    
    return ds_cf


def validate_cf_compliance(ds: xr.Dataset, json_path: Union[str, Path]) -> dict:
    """
    Validate a dataset against CF attribute requirements from JSON.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate
    json_path : str or Path
        Path to JSON file with CF requirements
        
    Returns
    -------
    dict
        Validation report with 'missing', 'present', and 'extra' keys
    """
    cf_config = load_cf_attributes(json_path)
    var_attrs = cf_config.get('variables', {})
    
    report = {
        'missing_variables': [],
        'present_variables': [],
        'extra_variables': [],
        'missing_attributes': {},
    }
    
    # Check variables
    expected_vars = set(var_attrs.keys())
    actual_vars = set(ds.data_vars) | set(ds.coords)
    
    report['missing_variables'] = list(expected_vars - actual_vars)
    report['present_variables'] = list(expected_vars & actual_vars)
    report['extra_variables'] = list(actual_vars - expected_vars)
    
    # Check attributes for present variables
    for var_name in report['present_variables']:
        expected_attrs = set(var_attrs[var_name].keys())
        actual_attrs = set(ds[var_name].attrs.keys())
        missing = expected_attrs - actual_attrs
        if missing:
            report['missing_attributes'][var_name] = list(missing)
    
    return report