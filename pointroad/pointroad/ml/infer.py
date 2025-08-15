#!/usr/bin/env python3
"""Semantic segmentation inference using pretrained models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import open3d as o3d
from loguru import logger

from .model_loader import (
    get_semantic_colors, map_kitti_to_canonical, get_class_names,
    get_discrete_classes, CANONICAL_CLASSES, get_recommended_model
)
from .enhanced_infer import run_enhanced_segmentation, PretrainedModelManager


@dataclass
class InferenceResult:
    """Result of semantic segmentation inference."""
    labels: np.ndarray  # Class labels per point
    scores: np.ndarray  # Confidence scores per point
    class_names: list[str]  # List of class names
    colors: np.ndarray  # RGB colors per point


def run_segmentation_dummy(pcd: o3d.geometry.PointCloud) -> InferenceResult:
    """Enhanced dummy segmentation with improved heuristics for car detection."""
    points = np.asarray(pcd.points)
    num_points = len(points)
    
    if num_points == 0:
        return InferenceResult(
            labels=np.array([]),
            scores=np.array([]),
            class_names=CANONICAL_CLASSES,
            colors=np.array([]).reshape(0, 3)
        )
    
    # Create realistic segmentation based on geometric analysis
    labels = np.full(num_points, CANONICAL_CLASSES.index("unlabeled"), dtype=np.int32)
    scores = np.ones(num_points) * 0.8  # More consistent confidence
    
    z_coords = points[:, 2]
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Calculate local point density for better classification
    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(10, num_points)).fit(points)
        distances, _ = nbrs.kneighbors(points)
        local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-6)
        density_percentile = np.percentile(local_density, [25, 50, 75])
    except:
        # Fallback if sklearn not available
        local_density = np.ones(num_points)
        density_percentile = [1, 1, 1]
    
    # Ground level (roads and sidewalks)
    ground_height = np.percentile(z_coords, 10)  # Adaptive ground level
    ground_mask = z_coords <= (ground_height + 0.2)
    
    # Roads: low, flat, high density
    road_mask = ground_mask & (local_density > density_percentile[1])
    labels[road_mask] = CANONICAL_CLASSES.index("road")
    scores[road_mask] = 0.9
    
    # Sidewalks: slightly elevated from road, medium density
    sidewalk_mask = (z_coords > ground_height + 0.1) & (z_coords < ground_height + 0.5) & ~road_mask
    labels[sidewalk_mask] = CANONICAL_CLASSES.index("sidewalk") 
    scores[sidewalk_mask] = 0.85
    
    # Buildings: tall structures with high density
    building_height_threshold = ground_height + 3.0
    building_mask = (z_coords > building_height_threshold) & (local_density > density_percentile[2])
    labels[building_mask] = CANONICAL_CLASSES.index("building")
    scores[building_mask] = 0.9
    
    # Enhanced car detection using multiple criteria
    car_height_min = ground_height + 0.3
    car_height_max = ground_height + 2.2
    
    # Base car candidates: right height range
    car_candidates = (z_coords >= car_height_min) & (z_coords <= car_height_max)
    car_candidates &= ~road_mask & ~sidewalk_mask & ~building_mask
    
    if np.any(car_candidates):
        candidate_points = points[car_candidates]
        candidate_z = z_coords[car_candidates]
        candidate_density = local_density[car_candidates]
        
        logger.debug(f"Car candidates: {len(candidate_points)} points in height range {car_height_min:.2f}-{car_height_max:.2f}")
        
        # Analyze spatial distribution for car-like clusters
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.6, min_samples=15).fit(candidate_points)
            cluster_labels = clustering.labels_
            
            # Evaluate each cluster for car-likeness
            unique_clusters = np.unique(cluster_labels)
            valid_clusters = unique_clusters[unique_clusters >= 0]
            car_mask_candidates = np.zeros(len(candidate_points), dtype=bool)
            
            logger.debug(f"Found {len(valid_clusters)} valid clusters from {len(unique_clusters)} total clusters")
            
            for cluster_id in valid_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_points = candidate_points[cluster_mask]
                
                if len(cluster_points) < 15:  # Too few points
                    continue
                
                # Analyze cluster dimensions
                min_coords = np.min(cluster_points, axis=0)
                max_coords = np.max(cluster_points, axis=0)
                dimensions = max_coords - min_coords
                
                length, width, height = np.sort(dimensions)[::-1]  # Sort descending
                
                # Car-like dimension checks (more permissive)
                is_car_like = (
                    2.0 <= length <= 8.0 and      # Car length (wider range)
                    1.0 <= width <= 3.5 and       # Car width (wider range)
                    0.5 <= height <= 3.0 and      # Car height (wider range)
                    1.0 <= length/width <= 6.0    # More permissive aspect ratio
                )
                
                if is_car_like:
                    # Additional checks for car-like properties
                    cluster_z_range = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])
                    cluster_density_mean = np.mean(candidate_density[cluster_mask])
                    
                    density_check = cluster_density_mean > 0.1  # Simple fixed threshold
                    height_check = cluster_z_range > 0.3
                    
                    # Cars should have reasonable height variation and medium density (more permissive)
                    if height_check and density_check:
                        car_mask_candidates[cluster_mask] = True
            
        except:
            # Fallback: use density and height criteria only
            height_score = 1.0 - np.abs(candidate_z - (car_height_min + car_height_max) / 2) / ((car_height_max - car_height_min) / 2)
            density_score = np.clip(candidate_density / density_percentile[1], 0, 2) / 2
            combined_score = (height_score + density_score) / 2
            car_mask_candidates = combined_score > 0.6
        
        # Apply car labels
        car_indices = np.where(car_candidates)[0][car_mask_candidates]
        labels[car_indices] = CANONICAL_CLASSES.index("car")
        scores[car_indices] = 0.8
    
    # Vegetation: medium height, scattered, medium-low density
    veg_mask = (z_coords > ground_height + 0.5) & (z_coords < building_height_threshold)
    veg_mask &= (local_density < density_percentile[2]) & ~building_mask
    veg_mask &= labels == CANONICAL_CLASSES.index("unlabeled")  # Don't override other classes
    
    # Random sampling for vegetation (not all medium-height points are vegetation)
    if np.any(veg_mask):
        veg_indices = np.where(veg_mask)[0]
        # Sample based on position variation (more scattered = more likely vegetation)
        n_sample = min(len(veg_indices), int(len(veg_indices) * 0.4))
        sampled_veg = np.random.choice(veg_indices, n_sample, replace=False)
        labels[sampled_veg] = CANONICAL_CLASSES.index("vegetation")
        scores[sampled_veg] = 0.7
    
    # Poles: very high, thin structures with low point count
    pole_mask = (z_coords > building_height_threshold) & (local_density < density_percentile[0])
    pole_mask &= labels == CANONICAL_CLASSES.index("unlabeled")
    if np.any(pole_mask):
        # Randomly sample some pole candidates
        pole_indices = np.where(pole_mask)[0]
        n_sample = min(len(pole_indices), int(len(pole_indices) * 0.1))
        if n_sample > 0:
            sampled_poles = np.random.choice(pole_indices, n_sample, replace=False)
            labels[sampled_poles] = CANONICAL_CLASSES.index("pole")
            scores[sampled_poles] = 0.75
    
    # Generate colors
    colors = np.zeros((num_points, 3), dtype=np.float64)
    semantic_colors = get_semantic_colors()
    
    for i, class_name in enumerate(CANONICAL_CLASSES):
        mask = labels == i
        if np.any(mask) and class_name in semantic_colors:
            colors[mask] = semantic_colors[class_name]
    
    logger.info(f"Enhanced dummy segmentation complete: {np.sum(labels == CANONICAL_CLASSES.index('car'))} car points detected")
    
    return InferenceResult(
        labels=labels,
        scores=scores,
        class_names=CANONICAL_CLASSES,
        colors=colors
    )


def run_segmentation_open3d_ml(pcd: o3d.geometry.PointCloud, model_path: str) -> InferenceResult:
    """Run segmentation using Open3D-ML models."""
    try:
        import open3d.ml.torch as ml3d
        from open3d.ml.torch.models import MinkUNet, KPConv
        import torch
        
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Normalize points to [-1, 1] range
        points_centered = points - points.mean(axis=0)
        points_normalized = points_centered / (points_centered.std(axis=0) + 1e-8)
        
        # Load model
        model = MinkUNet(num_classes=20, in_channels=3)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Run inference
        with torch.no_grad():
            # Convert to tensor
            points_tensor = torch.from_numpy(points_normalized).unsqueeze(0)
            
            # Run model
            logits = model(points_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            confidence_scores = torch.max(probabilities, dim=-1)[0]
        
        # Convert to numpy
        labels = predictions.numpy()[0]
        scores = confidence_scores.numpy()[0]
        
        # Generate colors
        colors = np.zeros((len(points), 3), dtype=np.float64)
        semantic_colors = get_semantic_colors()
        
        for i, class_name in enumerate(CANONICAL_CLASSES):
            mask = labels == i
            if np.any(mask) and class_name in semantic_colors:
                colors[mask] = semantic_colors[class_name]
        
        return InferenceResult(
            labels=labels,
            scores=scores,
            class_names=CANONICAL_CLASSES,
            colors=colors
        )
        
    except ImportError:
        logger.error("Open3D-ML not available")
        return None
    except Exception as e:
        logger.error(f"Open3D-ML inference failed: {e}")
        return None


def run_segmentation_pretrained(
    pcd: o3d.geometry.PointCloud,
    model_name: Optional[str] = None,
    force_download: bool = False
) -> InferenceResult:
    """Run segmentation using pretrained models."""
    try:
        # Use enhanced inference with pretrained models
        enhanced_result = run_enhanced_segmentation(pcd, model_name, force_download)
        
        if enhanced_result is None:
            logger.warning("Enhanced segmentation failed, falling back to dummy segmentation")
            return run_segmentation_dummy(pcd)
        
        # Convert to InferenceResult format
        return InferenceResult(
            labels=enhanced_result.labels,
            scores=enhanced_result.scores,
            class_names=enhanced_result.class_names,
            colors=enhanced_result.colors
        )
        
    except Exception as e:
        logger.error(f"Pretrained model inference failed: {e}")
        logger.warning("Falling back to dummy segmentation")
        return run_segmentation_dummy(pcd)


def run_segmentation(
    pcd: o3d.geometry.PointCloud,
    method: str = "auto",
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    force_download: bool = False
) -> InferenceResult:
    """Run semantic segmentation using the specified method."""
    
    if method == "auto":
        # Try pretrained models first, then fallback to dummy
        logger.info("Using automatic method selection")
        result = run_segmentation_pretrained(pcd, model_name, force_download)
        if result is not None:
            return result
        else:
            logger.warning("Pretrained models failed, using dummy segmentation")
            return run_segmentation_dummy(pcd)
    
    elif method == "pretrained":
        logger.info("Using pretrained models")
        result = run_segmentation_pretrained(pcd, model_name, force_download)
        if result is None:
            raise RuntimeError("Pretrained model segmentation failed")
        return result
    
    elif method == "open3d_ml":
        if model_path is None:
            raise ValueError("model_path must be provided for Open3D-ML method")
        logger.info("Using Open3D-ML")
        result = run_segmentation_open3d_ml(pcd, model_path)
        if result is None:
            raise RuntimeError("Open3D-ML segmentation failed")
        return result
    
    elif method == "dummy":
        logger.info("Using dummy segmentation")
        return run_segmentation_dummy(pcd)
    
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def get_available_methods() -> list[str]:
    """Get list of available segmentation methods."""
    methods = ["auto", "dummy"]

    # Check if pretrained models pathway is usable (do not actually download)
    try:
        from .enhanced_infer import PretrainedModelManager
        _ = PretrainedModelManager()  # ensures torch available
        # Pretrained mode is always available logically; the manager handles runtime errors
        methods.append("pretrained")
    except Exception:
        pass

    # Check if Open3D-ML is importable without raising the confusing warning
    try:
        import importlib
        _o3d_ml = importlib.import_module("open3d.ml.torch")
        methods.append("open3d_ml")
    except Exception:
        # Do not expose Open3D-ML when not properly built
        pass

    return methods


def get_recommended_method() -> str:
    """Get the recommended segmentation method."""
    methods = get_available_methods()
    
    if "pretrained" in methods:
        return "pretrained"
    elif "open3d_ml" in methods:
        return "open3d_ml"
    else:
        return "dummy"


def get_class_statistics(result: InferenceResult) -> dict[str, dict[str, float]]:
    """Get statistics about the segmentation results."""
    stats = {}
    total_points = len(result.labels)
    
    if total_points == 0:
        return stats
    
    for i, class_name in enumerate(result.class_names):
        mask = result.labels == i
        count = np.sum(mask)
        
        if count > 0:
            class_stats = {
                'count': int(count),
                'percentage': float(count / total_points * 100),
                'mean_score': float(np.mean(result.scores[mask])),
                'min_score': float(np.min(result.scores[mask])),
                'max_score': float(np.max(result.scores[mask]))
            }
            stats[class_name] = class_stats
    
    return stats


__all__ = [
    "InferenceResult",
    "run_segmentation",
    "run_segmentation_dummy",
    "run_segmentation_open3d_ml",
    "run_segmentation_pretrained",
    "get_available_methods",
    "get_recommended_method",
    "get_class_statistics"
]


