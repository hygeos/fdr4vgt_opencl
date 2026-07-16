# FDR4VGT Processing – Performance Plan

Reference product: `SPOTVGT1 X00Y00` (grid 5049×4669, tile size 512 → 100 tiles).
Dev machine: 32 cores, ~192 GB RAM, CPU-only OpenCL (PoCL). Deployment nodes: 32 cores, **4 GB RAM hard cap**, one product per node.

---

## 1. Done (delivered)

Single-product runtime **~26 min → ~13 min (≈2×)**, memory **~6 GB → ~3.1 GB** (`preload=0`), all 4 GB-safe.

### Memory (fits 4 GB nodes)
- Per-tile MERRA interpolation (`open_merra_global` + `interp_merra_tile`) instead of full-grid → avoids ~2.8 GB.
- Drop `UNC_RANDOM/STRUCTURED/SYSTEMATIC` after `calc_error` → −1.1 GB.
- Lazy `TOA`/`UNC` in `read_spotvgt` (`dask_stack`) → base 1.9 GB → 0.4 GB.

### Speed (the real bottlenecks, found by cProfile)
The interpolation *math* was ~0.5 s/tile; the cost was overhead:
- **Interpolation overhead** (~6 s/tile): `core.interpolate.interp` builds dask task-graphs + per-pixel weight location even though MERRA/climatology are **regular grids**. Fixed with `funcs.regular_interp` (`scipy.RegularGridInterpolator` on numpy). Used in `interp_merra_tile` and `get_aer_interpolated`.
- **DEM re-read every tile** (~5 s/tile): the `_operator.getitem` hotspot was `netCDF4._getitem` — the DEM read in `get_slope_err`. Fixed by preloading the DEM band once (`dem_ds = xrcrop(open_dataset(dem), lat=data['y']).load()`).
- **Save** (4 s → 2 s): zlib complevel 4 → 1.
- `preload=1` config option: load full L1 into RAM once (numpy) so each tile is a pure slice (memory-rich hosts only).
- Reusable ONNX session, cached `pre_aer_models`, preloaded monthly-aerosol datasets, per-tile live progress + ETA.

### Config knobs added (`[Sizes]`)
- `nworkers` (int) – process pool size (see §3).
- `preload` (0/1) – load full L1 into RAM. **Use 0 on 4 GB nodes, 1 on memory-rich hosts.**

---

## 2. Why simple multi-process parallelism does NOT help (measured)

A single process only uses **~5 of 32 cores**, yet 4/6/8 workers gave **no speedup** (~8 s/tile, save inflated 2 s → 8–16 s). Tested on local disk AND ceph, with `OMP/OPENBLAS/MKL` thread limits. Root causes:
1. Each worker's **PoCL OpenCL context grabs cores greedily** — no clean per-context CPU-thread cap in PoCL.
2. The **serial NetCDF writer's zlib compression is CPU-bound and starved** by the workers (local-disk test proved it is CPU starvation, not network I/O). Uncompressed output on ceph is *worse* (more bytes over the network FS).

Conclusion: the current "single compressed output file + one writer" design plus PoCL core-contention makes naive process-parallelism a dead end.

---

## 3. Potential plan to go beyond ~13 min

### Option A — Parallel compressed output (highest leverage, self-contained)
Goal: remove the serial-writer starvation so the ~5-core-per-tile compute can be spread across the 32 cores. Target ~3–5 min on a memory-rich node.
- Each worker computes AND **writes its own tile** (compression happens in the worker, distributed across cores).
- Assemble the single output file **without recompressing**, using **HDF5 direct chunk writes** (`h5py` `write_direct_chunk`) so pre-compressed tile chunks are pasted into the final file. Requires aligning NetCDF/HDF5 chunk layout to the 512 tile grid.
- Alternative if direct-chunk is too fiddly: keep per-tile temp files and do a final parallel merge, or expose the output as a virtual/kerchunk dataset.
- Must still cap per-worker OpenCL threads (investigate PoCL sub-devices / `cl.create_sub_devices` by partition) to avoid core contention.
- Keep `nworkers`/`preload` config-driven; on 4 GB nodes keep `nworkers=1`.

### Option B — Speed up the `smaccl` OpenCL kernel (biggest ceiling, out of this repo)
- `run` is ~4–5 s/tile and only extracts ~5 cores of parallelism (2 kernel launches VIS+IR × 11-model aerosol ensemble). It lives in the installed `smaccl` package.
- Investigate: larger global work-size / better work-group mapping to use all 32 CUs; avoid redundant host↔device copies (on CPU device they are wasted); fuse the VIS/IR launches; reduce the 11-model ensemble cost.
- Would lower the per-tile floor for BOTH single- and multi-process paths.

### Option C — Cheaper remaining per-tile items (incremental)
- `closest_model_low` (~0.4 s/tile) – vectorize carefully (reverted earlier due to memory spike with large `Nmod`; revisit with block-wise matmul and a memory cap).
- Halo (±1) causes 4× chunk reads when data is lazy; with `preload=1` this is moot, but for 4 GB nodes consider reading tiles chunk-aligned.

### Suggested order
1. Option A (unlocks the cores already available; targets ~3–5 min on rich nodes).
2. Option B (lifts the floor everywhere; needs `smaccl` package changes).
3. Option C (marginal).

---

## 4. Validation checklist for any change
- Wall time: `time python fdr4vgt/process.py <cfg>` (full 100 tiles).
- Memory: sample PSS (`/proc/<pid>/smaps_rollup`) + `MemAvailable` delta; must stay < 4 GB with `preload=0`.
- Correctness: `np.allclose` / nanmax-abs-diff on output `Rtoc_*`, `UrTOC*` vs a baseline run (negligible differences acceptable).
- Profiling: `FDR4VGT_MAXTILES=5 python -m cProfile -o prof.out ...` then `pstats` sort by `tottime`.

## 5. Cleanup / notes
- `read_merra()` in `process.py` is now dead (superseded by `open_merra_global` + `interp_merra_tile`) — safe to remove.
- `FDR4VGT_MAXTILES` env var is a profiling aid (gated, default off) — keep or remove.
- Output on `/mnt/ceph` is a network FS: prefer compressed writes there.
