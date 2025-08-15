#!/usr/bin/env python3
"""Instance clustering for semantic segmentation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import open3d as o3d
from loguru import logger

from ..ml.model_loader import get_discrete_classes


@dataclass
class InstanceInfo:
    """Information about a detected instance."""
    class_name: str
    instance_id: int
    point_indices: np.ndarray
    centroid: np.ndarray
    aabb: o3d.geometry.AxisAlignedBoundingBox
    obb: o3d.geometry.OrientedBoundingBox
    mean_score: float
    num_points: int
    extent: np.ndarray


def cluster_per_class(
    pcd: o3d.geometry.PointCloud,
    class_names: np.ndarray,
    scores: np.ndarray,
    eps: float = 0.4,
    min_points: int = 20,
) -> List[InstanceInfo]:
    """Cluster points by semantic class to find instances."""
    points = np.asarray(pcd.points)
    instances: List[InstanceInfo] = []
    unique_classes = np.unique(class_names)
    instance_counter = 0
    
    discrete_classes = get_discrete_classes()
    
    logger.info(f"Clustering {len(unique_classes)} classes for instances")
    
    for cname in unique_classes:
        if cname not in discrete_classes:
            continue
            
        class_mask = class_names == cname
        idxs = np.where(class_mask)[0]
        
        if idxs.size == 0:
            continue
            
        logger.info(f"Clustering class '{cname}' with {idxs.size} points")
        
        # Create sub-pointcloud for this class
        sub_pcd = pcd.select_by_index(idxs)
        
        # Run DBSCAN clustering
        labels = np.array(sub_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        
        if labels.size == 0:
            continue
            
        max_label = labels.max()
        
        for lid in range(max_label + 1):
            lid_mask = labels == lid
            if not np.any(lid_mask):
                continue
                
            # Get points for this instance
            pidxs = idxs[lid_mask]
            cluster_points = points[pidxs]
            
            # Calculate instance properties
            centroid = cluster_points.mean(axis=0)
            mean_score = float(scores[pidxs].mean())
            
            # Create sub-pointcloud for bounding box calculation
            sub = o3d.geometry.PointCloud()
            sub.points = o3d.utility.Vector3dVector(cluster_points)
            
            # Calculate bounding boxes
            aabb = sub.get_axis_aligned_bounding_box()
            obb = sub.get_oriented_bounding_box()
            extent = aabb.get_extent()
            
            # Create instance info
            instance = InstanceInfo(
                class_name=str(cname),
                instance_id=instance_counter,
                point_indices=pidxs,
                centroid=centroid,
                aabb=aabb,
                obb=obb,
                mean_score=mean_score,
                num_points=len(pidxs),
                extent=extent
            )
            
            instances.append(instance)
            instance_counter += 1
            
            logger.info(f"  Instance {instance_counter-1}: {len(pidxs)} points, "
                       f"centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
    
    logger.info(f"Found {len(instances)} total instances across all classes")
    return instances


def cluster_all_points(
    pcd: o3d.geometry.PointCloud, eps: float = 0.5, min_points: int = 20
) -> np.ndarray:
    """Cluster the entire cloud with DBSCAN, returning instance ids per point (or -1 for noise)."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0:
        return np.full(len(pcd.points), -1, dtype=int)
    return labels.astype(int)


def cluster_semantic_instances(
    pcd: o3d.geometry.PointCloud,
    class_names: np.ndarray,
    scores: np.ndarray,
    eps_by_class: Dict[str, float] = None,
    min_points_by_class: Dict[str, int] = None
) -> List[InstanceInfo]:
    """Advanced clustering with class-specific parameters."""
    
    # Default parameters
    if eps_by_class is None:
        eps_by_class = {
            "car": 1.0,
            "truck": 1.5,
            "bicycle": 0.5,
            "motorcycle": 0.5,
            "person": 0.3,
            "bicyclist": 0.3,
            "motorcyclist": 0.3,
            "building": 2.0,
            "fence": 0.5,
            "vegetation": 1.0,
            "trunk": 0.3,
            "pole": 0.2,
            "traffic-sign": 0.3
        }
    
    if min_points_by_class is None:
        min_points_by_class = {
            "car": 50,
            "truck": 100,
            "bicycle": 20,
            "motorcycle": 20,
            "person": 30,
            "bicyclist": 20,
            "motorcyclist": 20,
            "building": 200,
            "fence": 30,
            "vegetation": 100,
            "trunk": 20,
            "pole": 10,
            "traffic-sign": 10
        }
    
    points = np.asarray(pcd.points)
    instances: List[InstanceInfo] = []
    unique_classes = np.unique(class_names)
    instance_counter = 0
    
    discrete_classes = get_discrete_classes()
    
    logger.info(f"Advanced clustering for {len(unique_classes)} classes")
    
    for cname in unique_classes:
        if cname not in discrete_classes:
            continue
            
        class_mask = class_names == cname
        idxs = np.where(class_mask)[0]
        
        if idxs.size == 0:
            continue
        
        # Get class-specific parameters
        eps = eps_by_class.get(cname, 0.5)
        min_points = min_points_by_class.get(cname, 20)
        
        logger.info(f"Clustering '{cname}' with eps={eps}, min_points={min_points}, total_points={idxs.size}")
        
        # Create sub-pointcloud for this class
        sub_pcd = pcd.select_by_index(idxs)
        
        # Run DBSCAN clustering
        labels = np.array(sub_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        
        if labels.size == 0:
            continue
            
        max_label = labels.max()
        
        for lid in range(max_label + 1):
            lid_mask = labels == lid
            if not np.any(lid_mask):
                continue
                
            # Get points for this instance
            pidxs = idxs[lid_mask]
            cluster_points = points[pidxs]
            
            # Calculate instance properties
            centroid = cluster_points.mean(axis=0)
            mean_score = float(scores[pidxs].mean())
            
            # Create sub-pointcloud for bounding box calculation
            sub = o3d.geometry.PointCloud()
            sub.points = o3d.utility.Vector3dVector(cluster_points)
            
            # Calculate bounding boxes
            aabb = sub.get_axis_aligned_bounding_box()
            obb = sub.get_oriented_bounding_box()
            extent = aabb.get_extent()
            
            # Create instance info
            instance = InstanceInfo(
                class_name=str(cname),
                instance_id=instance_counter,
                point_indices=pidxs,
                centroid=centroid,
                aabb=aabb,
                obb=obb,
                mean_score=mean_score,
                num_points=len(pidxs),
                extent=extent
            )
            
            instances.append(instance)
            instance_counter += 1
            
            logger.info(f"  Instance {instance_counter-1} ({cname}): {len(pidxs)} points, "
                       f"centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), "
                       f"extent=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})")
    
    logger.info(f"Found {len(instances)} total instances across all classes")
    return instances


def get_instance_statistics(instances: List[InstanceInfo]) -> Dict[str, dict]:
    """Get statistics about detected instances by class."""
    stats = {}
    
    for instance in instances:
        class_name = instance.class_name
        if class_name not in stats:
            stats[class_name] = {
                "count": 0,
                "total_points": 0,
                "mean_score": 0.0,
                "mean_extent": np.zeros(3),
                "size_range": [float('inf'), 0.0]
            }
        
        stats[class_name]["count"] += 1
        stats[class_name]["total_points"] += instance.num_points
        stats[class_name]["mean_score"] += instance.mean_score
        stats[class_name]["mean_extent"] += instance.extent
        
        # Update size range
        size = instance.extent[0] * instance.extent[1] * instance.extent[2]
        stats[class_name]["size_range"][0] = min(stats[class_name]["size_range"][0], size)
        stats[class_name]["size_range"][1] = max(stats[class_name]["size_range"][1], size)
    
    # Calculate averages
    for class_name in stats:
        count = stats[class_name]["count"]
        if count > 0:
            stats[class_name]["mean_score"] /= count
            stats[class_name]["mean_extent"] /= count
    
    return stats


__all__ = [
    "InstanceInfo", 
    "cluster_per_class", 
    "cluster_all_points",
    "cluster_semantic_instances",
    "get_instance_statistics"
]


