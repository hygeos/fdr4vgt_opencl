#!/usr/bin/env python3
"""Test DEM smoothing impact on xdem slope/aspect noise.

This script compares terrain derivatives computed from:
1) Raw DEM interpolation on a scene window.
2) Gaussian-smoothed DEM (sigma=1) before derivatives.

It compares Horn(3x3) and Florinsky methods in xdem.
"""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xdem.terrain as xdem_terrain
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


def read_cfg(cfg_path: Path) -> dict:
    cf = configparser.ConfigParser()
    cf.read(cfg_path)
    return {
        "output": cf["Paths"]["output"],
        "dem": cf["Paths"]["dem"],
    }


def nanmedian_fill(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    valid = np.isfinite(out)
    if not np.any(valid):
        return np.zeros_like(out, dtype=np.float32)
    fill_val = np.float32(np.nanmedian(out[valid]))
    out[~valid] = fill_val
    return out


def edge_metric_std(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    dx = np.diff(arr, axis=1)
    dy = np.diff(arr, axis=0)
    gx = dx[np.isfinite(dx)]
    gy = dy[np.isfinite(dy)]
    if gx.size == 0 and gy.size == 0:
        return float("nan")
    g = np.concatenate([gx, gy])
    if g.size == 0:
        return float("nan")
    return float(np.std(g))


def valid_stats(name: str, arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float32)
    v = a[np.isfinite(a)]
    if v.size == 0:
        return {
            "name": name,
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "edge_std": None,
        }
    return {
        "name": name,
        "count": int(v.size),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "edge_std": edge_metric_std(a),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare xdem slope/aspect with and without DEM Gaussian smoothing")
    parser.add_argument("--cfg", default="tests/all_process_spotvgt1_2.cfg", help="Config file path")
    parser.add_argument("--window", type=int, default=512, help="Square window size around max AOD pixel")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma for DEM smoothing")
    parser.add_argument("--outdir", default="tests/tmp", help="Output directory for plots and stats")
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = read_cfg(cfg_path)
    output_nc = Path(cfg["output"])
    dem_nc = Path(cfg["dem"])

    print(f"Config: {cfg_path}")
    print(f"Output NC: {output_nc}")
    print(f"DEM NC: {dem_nc}")

    with xr.open_dataset(output_nc) as ds_out:
        aod = ds_out["AOD_550_used"].values.astype(np.float32)
        lat = ds_out["lat"].values.astype(np.float64)
        lon = ds_out["lon"].values.astype(np.float64)

    valid_aod = np.isfinite(aod)
    if not np.any(valid_aod):
        raise RuntimeError("No finite AOD_550_used pixel found.")

    aod_for_argmax = np.where(valid_aod, aod, -np.inf)
    cy, cx = np.unravel_index(np.argmax(aod_for_argmax), aod.shape)
    w = int(args.window)
    half = w // 2
    y0 = max(0, min(cy - half, aod.shape[0] - w))
    x0 = max(0, min(cx - half, aod.shape[1] - w))
    y1 = min(aod.shape[0], y0 + w)
    x1 = min(aod.shape[1], x0 + w)

    lat_w = lat[y0:y1, x0:x1]
    lon_w = lon[y0:y1, x0:x1]

    with xr.open_dataset(dem_nc) as ds_dem:
        dem_lat = ds_dem["lat"].values.astype(np.float64)
        dem_lon = ds_dem["lon"].values.astype(np.float64)
        dem_elev = ds_dem["elev"].values.astype(np.float32)

    # RegularGridInterpolator requires ascending coordinates.
    if dem_lat[0] > dem_lat[-1]:
        dem_lat = dem_lat[::-1]
        dem_elev = dem_elev[::-1, :]
    if dem_lon[0] > dem_lon[-1]:
        dem_lon = dem_lon[::-1]
        dem_elev = dem_elev[:, ::-1]

    interp = RegularGridInterpolator(
        (dem_lat, dem_lon),
        dem_elev,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    pts = np.column_stack([lat_w.ravel(), lon_w.ravel()])
    dem_win = interp(pts).reshape(lat_w.shape).astype(np.float32)
    dem_fill = nanmedian_fill(dem_win)

    dem_smooth = gaussian_filter(dem_fill, sigma=float(args.sigma)).astype(np.float32)

    methods = ["Horn", "Florinsky"]
    method_results = {}
    for method in methods:
        slope_raw = xdem_terrain.slope(dem_fill, method=method, resolution=10.0).astype(np.float32)
        aspect_raw = xdem_terrain.aspect(dem_fill, method=method).astype(np.float32)

        slope_smooth = xdem_terrain.slope(dem_smooth, method=method, resolution=10.0).astype(np.float32)
        aspect_smooth = xdem_terrain.aspect(dem_smooth, method=method).astype(np.float32)

        slope_diff = slope_smooth - slope_raw
        # Circular aspect difference in degrees [-180, 180].
        aspect_diff = ((aspect_smooth - aspect_raw + 180.0) % 360.0) - 180.0
        method_results[method] = {
            "slope_raw": slope_raw,
            "aspect_raw": aspect_raw,
            "slope_smooth": slope_smooth,
            "aspect_smooth": aspect_smooth,
            "slope_diff": slope_diff,
            "aspect_diff": aspect_diff,
        }

    stats = {
        "window": {
            "y0": int(y0),
            "y1": int(y1),
            "x0": int(x0),
            "x1": int(x1),
            "center_y": int(cy),
            "center_x": int(cx),
            "size": int(y1 - y0),
        },
        "params": {
            "gaussian_sigma": float(args.sigma),
            "xdem_methods": methods,
            "xdem_resolution": 10.0,
        },
        "dem_raw": valid_stats("dem_raw", dem_fill),
        "dem_smooth": valid_stats("dem_smooth", dem_smooth),
        "methods": {
            method: {
                "slope_raw": valid_stats("slope_raw", result["slope_raw"]),
                "slope_smooth": valid_stats("slope_smooth", result["slope_smooth"]),
                "aspect_raw": valid_stats("aspect_raw", result["aspect_raw"]),
                "aspect_smooth": valid_stats("aspect_smooth", result["aspect_smooth"]),
                "slope_diff": valid_stats("slope_diff", result["slope_diff"]),
                "aspect_diff": valid_stats("aspect_diff", result["aspect_diff"]),
            }
            for method, result in method_results.items()
        },
    }

    stats_path = outdir / "dem_smoothing_xdem_horn_florinsky_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Saved stats:", stats_path)
    for method in methods:
        s_raw = stats["methods"][method]["slope_raw"]["edge_std"]
        s_smooth = stats["methods"][method]["slope_smooth"]["edge_std"]
        print(f"{method} slope edge std raw/smooth:", s_raw, s_smooth)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)

    im0 = axes[0, 0].imshow(dem_fill, cmap="terrain")
    axes[0, 0].set_title("DEM raw")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(method_results["Horn"]["slope_raw"], cmap="viridis")
    axes[0, 1].set_title("Slope raw (Horn)")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(method_results["Horn"]["aspect_raw"], cmap="hsv", vmin=0, vmax=360)
    axes[0, 2].set_title("Aspect raw (Horn)")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im3 = axes[1, 0].imshow(dem_smooth, cmap="terrain")
    axes[1, 0].set_title(f"DEM smooth (gaussian sigma={args.sigma})")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(method_results["Horn"]["slope_smooth"], cmap="viridis")
    axes[1, 1].set_title("Slope smooth (Horn)")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    vmax_aspect_diff_h = np.nanpercentile(np.abs(method_results["Horn"]["aspect_diff"]), 99)
    im5 = axes[1, 2].imshow(method_results["Horn"]["aspect_diff"], cmap="coolwarm", vmin=-vmax_aspect_diff_h, vmax=vmax_aspect_diff_h)
    axes[1, 2].set_title("Horn aspect diff smooth-raw (deg)")
    axes[1, 2].axis("off")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im6 = axes[2, 0].imshow(method_results["Florinsky"]["slope_raw"], cmap="viridis")
    axes[2, 0].set_title("Slope raw (Florinsky)")
    axes[2, 0].axis("off")
    plt.colorbar(im6, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im7 = axes[2, 1].imshow(method_results["Florinsky"]["slope_smooth"], cmap="viridis")
    axes[2, 1].set_title("Slope smooth (Florinsky)")
    axes[2, 1].axis("off")
    plt.colorbar(im7, ax=axes[2, 1], fraction=0.046, pad=0.04)

    vmax_aspect_diff_f = np.nanpercentile(np.abs(method_results["Florinsky"]["aspect_diff"]), 99)
    im8 = axes[2, 2].imshow(method_results["Florinsky"]["aspect_diff"], cmap="coolwarm", vmin=-vmax_aspect_diff_f, vmax=vmax_aspect_diff_f)
    axes[2, 2].set_title("Florinsky aspect diff smooth-raw (deg)")
    axes[2, 2].axis("off")
    plt.colorbar(im8, ax=axes[2, 2], fraction=0.046, pad=0.04)

    fig.suptitle("DEM smoothing test before xdem slope/aspect (Horn + Florinsky)", fontsize=14)
    fig_path = outdir / "dem_smoothing_xdem_horn_florinsky.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print("Saved figure:", fig_path)


if __name__ == "__main__":
    main()
