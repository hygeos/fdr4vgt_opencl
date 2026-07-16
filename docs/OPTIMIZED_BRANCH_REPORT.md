# `optimized` branch report

Comparison of the new **`optimized`** branches against their base branches, for
both repositories. Date: 2026-07-16.

| Repo | Base branch | New branch | New commit | Merge-base |
|---|---|---|---|---|
| [`hygeos/fdr4vgt_opencl`](https://github.com/hygeos/fdr4vgt_opencl) | `main` | `optimized` | `ef70941` | `e26dcb5` |
| [`hygeos/smaccl`](https://github.com/hygeos/smaccl) | `opencl` | `optimized` | `2d78ac2` | `1e983ef` |

Both branches are pushed to `origin`. Open a PR:
- fdr4vgt_opencl → `https://github.com/hygeos/fdr4vgt_opencl/pull/new/optimized`
- smaccl → `https://github.com/hygeos/smaccl/pull/new/optimized`

Each `optimized` branch is exactly **one commit** ahead of its base (fast-forward,
no divergence).

---

## 1. `fdr4vgt_opencl`: `main` → `optimized`

Single commit `ef70941` — *"Optimize single-scene atmospheric correction: memory + speed"*.

```
 docs/PERFORMANCE_PLAN.md    |  76 +++++
 fdr4vgt/funcs.py            | 179 ++++++++----
 fdr4vgt/in_out.py           |  48 +++-
 fdr4vgt/process.py          | 660 ++++++++++++++++++++++++++++++++++++--------
 fdr4vgt/spotvgt1_config.cfg |   5 +
 fdr4vgt/spotvgt_vito.py     |  17 +-
 pyproject.toml              |   1 +
 7 files changed, 798 insertions(+), 188 deletions(-)
```

### What changed

**`process.py`** (largest change) — the tiled processing pipeline:
- **Ancillary Zarr cache** (`precompute_ancillary`): MERRA-2 fields + best/monthly
  aerosol-model indices are interpolated onto the full grid **once** and streamed
  to a local Zarr store on scratch disk; each tile then slices its halo'd window
  from local disk instead of re-interpolating and re-reading the network per tile.
  Toggle `[Sizes] anc_cache` (default on).
- **Batched processing with a single persistent NetCDF writer**: the output handle
  is opened once and reused across all tiles (with a periodic HDF5 flush), instead
  of reopening `Dataset(fn,'a')` every tile.
- **Tile-aligned NetCDF chunking + shrunk HDF5 chunk cache** (see `in_out.py`):
  the dominant memory + writer-time win.
- **`preload`** option (full L1 into RAM on memory-rich hosts) and **`nworkers`**
  process-pool knob; **local output staging** (write to scratch, one final copy
  to the network destination).
- DEM / monthly-aerosol datasets and the ONNX session are preloaded once.

**`in_out.py`**:
- `create_nc` now calls `set_chunk_cache(4 MiB)` before variables are created
  (default is 64 MiB/var × ~84 vars ≈ 5.25 GB ceiling).
- All `createVariable(...)` calls pass `chunksizes=(band_size, band_size)`
  (512×512), aligning HDF5 chunks with the write tiles.
- **Result:** writer peak RSS ~5 GB → ~0.2 GB and ~7× faster writer (eliminates
  read-modify-write of the oversized default chunks). Stored data is bit-identical.
- **Fix:** `load_brdf` inverted finite mask — it previously *kept* NaNs and
  *zeroed* the valid BRDF coefficients (`np.logical_not(np.isfinite(...))`);
  corrected to keep finite values and replace non-finite with 0.

**`funcs.py`**: per-tile regular-grid interpolation (`RegularGridInterpolator`
on numpy) for MERRA and climatology, slope/terrain error, DEM preload helper.

**`spotvgt_vito.py`**: lazy (dask) TOA/UNC loading instead of materialising the
full arrays up front (keeps the resident base small).

**`spotvgt1_config.cfg`**: adds `[Paths] tmp_dir` (local scratch), `[Sizes] nworkers`
and `[Sizes] preload`.

**`pyproject.toml`**: adds the `zarr` dependency (used by the ancillary cache).

**`docs/PERFORMANCE_PLAN.md`**: performance analysis / plan document.

### Measured performance (SPOTVGT1 X00Y00, 100 tiles, single process)

| Config | Wall time | Speedup vs 19 m 20 s | Peak RAM |
|---|---|---|---|
| `preload=1` (memory-rich) | 319 s (5 m 19 s) | **3.64×** | 4.8 GB |
| `preload=0` (4 GB-safe) | 382 s (6 m 22 s) | **3.04×** | 3.5 GB |

Writer cost dropped from 2.16 s/tile to ~0.3 s/tile.

---

## 2. `smaccl`: `opencl` → `optimized`

Single commit `2d78ac2` — *"Optimize OpenCL smaccl kernel + add gated profiling"*.

```
 smaccl/kernels/devicecl.cl | 113 +++++++++++++++++++++++++++------------------
 smaccl/smaccl.py           |  46 +++++++++++++++++-
 2 files changed, 112 insertions(+), 47 deletions(-)
```

### What changed

**`kernels/devicecl.cl`** (the OpenCL kernel) — ~7× faster `smaccl.run` on the
CPU (PoCL); validated bit-identical vs baseline on a fully-valid zone
(`Rtoc` max-abs-diff ≈ 2.4e-7):
- **Hoist the RTLS BRDF kernels** (`F1_rtls`/`F2_rtls` → `ax1d/ax2d/ax1u/ax2u`)
  out of the band/aerosol/run loops to per-pixel (were recomputed ~132×/pixel).
- **Hoist gaseous transmission** out of the aerosol-model loop (gas coefficients
  are aerosol-model invariant); precompute the base and pressure-perturbed gas
  products per band.
- **Replace `pow(x, int)` with explicit multiplies** (spherical albedo, aerosol
  phase, Rayleigh / aerosol / 6S residual squares and cubes).

**`smaccl.py`**:
- Build with `options=['-cl-mad-enable']` — **no `-cl-fast-relaxed-math`**, whose
  finite-math-only assumption is unsafe with NaN-padded invalid pixels.
- Optional fine-grained `run()` profiling gated by env `SMACCL_PROFILE` (inert
  unless set), with an `atexit` summary of `host_alloc / h2d / kernel / d2h`.

---

## 3. Known open issue (not resolved on these branches)

On a full real scene the output science variables (`Rtoc_*`, all Jacobians,
`uncertainty_from_aerosol_*`) currently come out **100% NaN**, including over land.
This is **independent of the changes above**:
- The writer/chunking change is proven inert — `TOA` (written by the same chunked
  writer) is valid; only the smaccl-kernel-derived fields are NaN.
- Removing `-cl-fast-relaxed-math` did not change it.
- It has been traced toward `pressure = Ps(alt, …)` with the DEM `elev` being NaN
  over the processed pixels, plus the kernel path; the exact cause on land is
  still under investigation (next step: probe a land tile's `elev`/`pressure`).

The `optimized` branches capture the intended performance work and should be
treated as **work-in-progress** until this NaN issue is fixed and the output is
re-validated.
