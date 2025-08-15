from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import open3d as o3d
from loguru import logger


@dataclass
class DownsampleConfig:
    target_points_min: int = 200_000
    target_points_max: int = 400_000
    initial_voxel_size: float = 0.05
    adaptive_max_voxel_size: float = 0.25
    plane_guardrail_rmse: float = 0.02


def fit_plane_rmse(pcd: o3d.geometry.PointCloud, distance_threshold: float = 0.02) -> float:
    if len(pcd.points) < 1000:
        return 0.0
    _, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=200
    )
    if not inliers:
        return float("inf")
    inlier_cloud = pcd.select_by_index(inliers)
    plane_model, _ = inlier_cloud.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=50
    )
    a, b, c, d = plane_model
    pts = np.asarray(inlier_cloud.points)
    num = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d)
    den = np.sqrt(a * a + b * b + c * c) + 1e-9
    rmse = float(np.sqrt(np.mean((num / den) ** 2)))
    return rmse


def adaptive_voxel_downsample(
    pcd: o3d.geometry.PointCloud, cfg: DownsampleConfig
) -> Tuple[o3d.geometry.PointCloud, float]:
    """Downsample to target range while keeping planar fidelity.

    Returns the downsampled point cloud and the voxel size used.
    """
    num_points = len(pcd.points)
    if num_points == 0:
        return pcd, 0.0
    if cfg.target_points_min <= num_points <= cfg.target_points_max:
        return pcd, 0.0

    # Estimate baseline voxel size from point count ratio (heuristic)
    ratio = max(num_points / cfg.target_points_max, 1.0)
    voxel = min(cfg.initial_voxel_size * np.cbrt(ratio), cfg.adaptive_max_voxel_size)
    best = None
    for _ in range(6):
        ds = pcd.voxel_down_sample(voxel)
        rmse = fit_plane_rmse(ds, distance_threshold=max(0.5 * voxel, 0.01))
        logger.debug(f"voxel={voxel:.3f}, n={len(ds.points)}, rmse={rmse:.4f}")
        if rmse <= cfg.plane_guardrail_rmse and cfg.target_points_min <= len(ds.points) <= cfg.target_points_max:
            best = (ds, voxel)
            break
        # Adjust voxel: if too many points, increase voxel; if too few or rmse too high, decrease
        if len(ds.points) > cfg.target_points_max:
            voxel = min(voxel * 1.25, cfg.adaptive_max_voxel_size)
        else:
            voxel = max(voxel * 0.8, 0.01)
        best = (ds, voxel)
    assert best is not None
    return best


__all__ = ["DownsampleConfig", "adaptive_voxel_downsample", "fit_plane_rmse"]



