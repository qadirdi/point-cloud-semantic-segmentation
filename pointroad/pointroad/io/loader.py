from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import open3d as o3d
import laspy
from loguru import logger


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """Load a point cloud from PLY/PCD/LAS/LAZ into an Open3D PointCloud.

    Args:
        path: Input file path.

    Returns:
        Open3D PointCloud with points and optional colors/normals/intensity as channels.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".ply", ".pcd"]:
        logger.info(f"Reading point cloud via Open3D: {path}")
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            raise ValueError(f"Loaded empty point cloud: {path}")
        return pcd
    if suffix in [".las", ".laz"]:
        logger.info(f"Reading LAS via laspy: {path}")
        with laspy.open(str(path)) as lasf:
            points = lasf.read()
        xyz = np.vstack([points.x, points.y, points.z]).T.astype(np.float64)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        if hasattr(points, "red"):
            rgb = np.vstack([points.red, points.green, points.blue]).T.astype(np.float64)
            # Normalize 16-bit color to [0,1] if needed
            if rgb.max() > 1.0:
                rgb /= 65535.0 if rgb.max() > 255 else 255.0
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        return pcd
    raise ValueError(f"Unsupported point cloud format: {suffix}")


def estimate_memory_points(num_points: int, attrs: int = 3) -> float:
    """Estimate memory use in GB for given number of points and attributes.

    Args:
        num_points: Number of points.
        attrs: Attributes per point (3 xyz + optional channels). Assumes float64.

    Returns:
        Estimated memory in GB.
    """
    bytes_total = num_points * attrs * 8
    return bytes_total / (1024**3)


__all__ = ["load_point_cloud", "estimate_memory_points"]



