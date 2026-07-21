# fdr4vgt_opencl

[**Quickstart**](#Usage)
| [**Install guide**](#installation)

## Introduction

FDR4VGT is an atmospheric correction processing chain for **VEGETATION** (SPOT4/5) and **Proba-V** satellite sensors, producing **Top-of-Canopy (TOC) reflectance** with full **FIDUCEO-compliant uncertainty propagation**. The pipeline implements the **Simplified Model for Atmospheric Corrections (SMAC)** using an OpenCL-accelerated kernel (SMAC-CL), combined with MERRA-2 reanalysis data for atmospheric state variables and a comprehensive uncertainty budget following the FIDUCEO methodology.

### Key Features

- **SMAC-CL OpenCL kernel**: GPU-accelerated atmospheric correction for 4 spectral bands (Blue, Red, NIR, SWIR)
- **FIDUCEO uncertainty framework**: Full propagation of 7 uncertainty effects (TOA, O₃, H₂O, pressure, AOD, aerosol model, RTM fit)
- **Tile-based parallel processing**: Memory-efficient batch processing with configurable tile size and worker count
- **Ancillary Zarr caching**: Optimized local cache for interpolated MERRA-2 and terrain data
- **ONNX RTM fit uncertainty**: Neural-network-based radiative transfer model residual estimation
- **CF-1.8 compliant output**: NetCDF files with standard attributes and metadata

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FDR4VGT Processing Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Level-1C Input          Auxiliary Data                                     │
│  ┌──────────┐           ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ TOA Refl │           │ MERRA-2  │  │ DEM      │  │ SMAC     │           │
│  │ Geometry │           │ AER+SLV  │  │ (Slope/  │  │ Coeffs   │           │
│  │ Bands    │           │ (AOT,O3, │  │  Aspect) │  │ (.npy/.nc)│          │
│  └────┬─────┘           │  H2O,SLP)│  └────┬─────┘  └────┬─────┘           │
│       │                 └──────────┘       │            │                   │
│       │                                    │            │                   │
│       ▼                                    ▼            ▼                   │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                    Tile-Based Processing                     │          │
│  │  ┌────────────────────────────────────────────────────────┐  │          │
│  │  │ 1. MERRA-2 interpolation (per tile, RegularGridInterpolator)│     │  │          │
│  │  │ 2. Aerosol model index (iaer) from MERRA-2 fractions   │  │          │
│  │  │ 3. Terrain slope/aspect error (xdem_terrain)           │  │          │
│  │  │ 4. SMAC-CL atmospheric correction (OpenCL kernel)      │  │          │
│  │  │ 5. Jacobian computation (analytical + finite difference)│      │  │          │
│  │  │ 6. Uncertainty propagation (7 effects)                 │  │          │
│  │  │ 7. RTM fit uncertainty (ONNX runtime)                  │  │          │
│  │  └────────────────────────────────────────────────────────┘  │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                    NetCDF Output                              │          │
│  │  rTOC_B1..B4, UrTOC, Jacobians, Quality Flags, Ancillary     │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- **Python ≥ 3.10**
- **pixi** package manager (https://pixi.sh)
- **OpenCL-capable GPU** (for SMAC-CL acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/hygeos/fdr4vgt_opencl.git
cd fdr4vgt_opencl

# Install dependencies via pixi
pixi install

# Verify installation
pixi run tests
```

### Dependencies

| Category | Packages |
|----------|----------|
| Core | `xarray`, `dask`, `numpy`, `scipy` |
| SMAC-CL | `smaccl` (OpenCL kernel) |
| Uncertainty | `onnxruntime` (RTM fit estimation) |
| Terrain | `xdem_terrain` (slope/aspect from DEM) |
| I/O | `netCDF4`, `h5py` |
| Visualization | `matplotlib` |
| Testing | `pytest`, `pytest-xdist`, `pytest-html` |

### Environment

The project uses **pixi** for environment management. The environment is defined in `pyproject.toml` with three feature groups:
- **sys**: System-level dependencies
- **sub**: Sub-dependencies (SMAC-CL, core libraries)
- **dev**: Development tools (testing, linting)

## Usage

### Quick Start

```bash
# Run atmospheric correction with a configuration file
pixi exec python -m fdr4vgt.process fdr4vgt/spotvgt1_config.cfg
```

### Configuration File Structure

Configuration files use INI format with five sections:

#### `[Paths]` — Data and Output Locations
| Parameter | Description | Example |
|-----------|-------------|---------|
| `input` | Level-1C input directory | `/data/input/INPUT_ATMCOR_X00Y00/` |
| `output` | Output NetCDF file path | `/data/output/toc.nc` |
| `merraaero` | MERRA-2 aerosol extinction file | `/data/MERRA2/aer/...nc4` |
| `merraptwo` | MERRA-2 surface pressure/water vapor file | `/data/MERRA2/slv/...nc4` |
| `amip_path` | MERRA-2 AMIP ensemble directory | `/data/MERRA2/amip/` |
| `smaccoef_dir` | SMAC coefficients directory | `/data/smac/` |
| `dem` | Digital Elevation Model file | `/data/DEM/GTOPO30.nc` |
| `faer` | Aerosol model fractions file | `ANCILLARY/Aerosol_model_fraction.txt` |
| `brdf_dir` | C3S BRDF coefficients directory | `/data/brdf/` |
| `cf_json_path` | CF-1.8 attributes JSON file | `ANCILLARY/cf_attributes_res2.json` |
| `rmt_fit_file` | ONNX RTM fit model file | `ANCILLARY/model.onnx` |
| `tmp_dir` | Local scratch for Zarr cache | `/home/user/tmp` |

#### `[Sensor]` — Sensor Configuration
| Parameter | Description | Values |
|-----------|-------------|--------|
| `sensor` | Sensor identifier | `SPOTVGT1`, `SPOTVGT2`, `PROBAV` |
| `smaccoef_version` | SMAC coefficients version | `3.0` |

#### `[Coefficients]` — Uncertainty Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `k_uh2o` | Water vapor scaling factor | `1e-1` |
| `k_uo3` | Ozone scaling factor | `1e-3` |
| `k_p0` | Pressure scaling factor | `1e-2` |
| `Etoa`, `ERtoa` | TOA absolute/relative uncertainty | `0`, `0.01` |
| `Etaup`, `ERtaup` | AOT absolute/relative uncertainty | `0.03`, `0.15` |
| `Euo3`, `ERuo3` | Ozone absolute/relative uncertainty | `0.0`, `0.06` |
| `Euh2o`, `ERuh2o` | Water vapor absolute/relative uncertainty | `0.0`, `0.2` |
| `Epre`, `ERpre` | Pressure absolute/relative uncertainty | `1.0`, `0.0` |
| `tocmin`, `tocmax` | Valid TOC range | `0.0`, `1.0235` |
| `aodmax` | Maximum AOD for quality flag | `0.6` |
| `szamax` | Maximum solar zenith angle | `80.0` |
| `aodmax_grad` | Maximum AOD gradient | `1.4e-4` |
| `u2_0` | Second-order effects residual | `7e-3` |

#### `[Sizes]` — Processing Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `nworkers` | Number of parallel workers | `1` (serial) |
| `nmodels` | Number of aerosol models | `10` |
| `preload` | Preload L1 data into RAM (0/1) | `0` |
| `chunks_size` | Tile chunk size (pixels) | `512` |

#### `[Output]` — Output Options
| Parameter | Description | Values |
|-----------|-------------|--------|
| `jacobian` | Include Jacobian fields in output | `True`, `False` |

### Processing Workflow

1. **Configuration loading**: Read config file, parse all sections
2. **Level-1C ingestion**: Load satellite data via sensor-specific reader (`Level1_probav` or `Level1_spotvgt`)
3. **MERRA-2 global open**: Load aerosol + surface fields once, apply longitude convention
4. **BRDF loading**: Load C3S albedo coefficients for the acquisition date
5. **Tile grid generation**: Divide satellite grid into chunks of `chunks_size × chunks_size`
6. **Per-tile processing** (in worker processes):
   - MERRA-2 interpolation to satellite grid (per tile)
   - Aerosol model index computation from MERRA-2 fractions
   - Terrain slope/aspect error computation
   - SMAC-CL atmospheric correction (OpenCL kernel)
   - Jacobian computation (analytical for Rtoa/UO3/UH2O, finite difference for Ps/tau550)
   - Uncertainty propagation (7 effects)
   - RTM fit uncertainty (ONNX runtime)
7. **NetCDF output**: Tile-aligned batch writing with chunk optimization

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FDR4VGT_MAXTILES` | Maximum number of tiles to process (for testing) | All tiles |

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Memory footprint | ~4 GB per worker (with preload=0) |
| Processing speed | ~2.4 s/tile (512×512, single worker) |
| Ancillary cache | Local Zarr format, float16 (float32 for SLP/T10M) |
| Parallel scaling | Linear with `nworkers` (CPU-bound) |

## Output Format

### NetCDF Variables

| Variable | Dimensions | Description | Units |
|----------|------------|-------------|-------|
| `rTOC_B1..B4` | (y, x) | Top-of-canopy reflectance, bands 1-4 | dimensionless |
| `UrTOC_B1..B4` | (y, x) | Total uncertainty on rTOC | dimensionless |
| `UrTOC_ens` | (bands, y, x) | Aerosol ensemble uncertainty | dimensionless |
| `UrTOC_rtm_fit` | (bands, y, x) | RTM fit uncertainty (ONNX) | dimensionless |
| `UrTOC_rtm_brdf` | (bands, y, x) | BRDF model uncertainty | dimensionless |
| `urtoc_terrain` | (bands, y, x) | Terrain slope/aspect uncertainty | dimensionless |
| `unc_h2o` | (bands, y, x) | Water vapor contribution | dimensionless |
| `unc_o3` | (bands, y, x) | Ozone contribution | dimensionless |
| `unc_ps` | (bands, y, x) | Pressure contribution | dimensionless |
| `unc_aot` | (bands, y, x) | AOD contribution | dimensionless |
| `Jrtoa` | (bands, y, x) | Jacobian w.r.t. TOA reflectance | — |
| `Juh2o` | (bands, y, x) | Jacobian w.r.t. water vapor | — |
| `Juo3` | (bands, y, x) | Jacobian w.r.t. ozone | — |
| `Jpre` | (bands, y, x) | Jacobian w.r.t. pressure | — |
| `Jtau550` | (bands, y, x) | Jacobian w.r.t. AOD at 550nm | — |
| `flag` | (y, x) | Quality flag (bitfield) | — |
| `iaero` | (bands, y, x) | Aerosol model index | — |

### Quality Flag Bits

| Bit | Meaning |
|-----|---------|
| 0 | Cloud contaminated |
| 1 | High AOD (> aotmax) |
| 2 | High SZA (> szamax) |
| 3 | High AOD gradient |

## Project Structure

```
fdr4vgt_opencl/
├── fdr4vgt/
│   ├── process.py          # Main processing entry point
│   ├── funcs.py            # Core computation functions
│   ├── in_out.py           # I/O utilities (Level1, NetCDF, BRDF)
│   ├── CF_from_json.py     # CF-1.8 compliance utilities
│   ├── probav_vito.py      # Proba-V Level-1 reader
│   ├── spotvgt_vito.py     # SPOT-VGT Level-1 reader
│   ├── probav_config.cfg   # Proba-V configuration example
│   └── spotvgt1_config.cfg # SPOT-VGT1 configuration example
├── ANCILLARY/
│   ├── Aerosol_model_fraction.txt
│   ├── cf_attributes_res2.json
│   ├── kg_zoneNumGueymard.csv
│   ├── legend.txt
│   └── model.onnx
├── tests/
├── docs/
├── scripts/
├── pyproject.toml
└── README.md
```

## Algorithm Theory

For detailed algorithm theory, see the **Algorithm Theoretical Basis Document**:
- **FDR4VGT-AC-ATBD v1.0**: Atmospheric correction algorithm description, uncertainty methodology, and validation strategy

Key algorithm components:
1. **SMAC Algorithm**: Simplified Model for Atmospheric Corrections using pre-computed radiative transfer coefficients
2. **Sensitivity Computation**: Analytical derivatives for TOA, O₃, H₂O; finite difference for pressure and AOD
3. **Uncertainty Propagation**: FIDUCEO framework with 7 uncertainty effects
4. **Aerosol Model Selection**: MERRA-2 fraction-based matching to discrete aerosol models
5. **Terrain Correction**: Slope/aspect effects from DEM using xdem_terrain

## References

- **HYGEOS**: https://hygeos.com
- **FIDUCEO**: FIDelity and Uncertainty in Climate data records from Earth Observation
- **SMAC**: Simplified Model for Atmospheric Corrections (Roujean et al., 1992)
- **MERRA-2**: Modern-Era Retrospective analysis for Research and Applications, Version 2
- **ATBD Document**: FDR4VGT-AC-ATBD v1.0 — Algorithm Theoretical Basis Document

## Licensing

© HYGEOS — Contact: contact@hygeos.com