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
    """Dummy segmentation that assigns random classes for testing."""
    points = np.asarray(pcd.points)
    num_points = len(points)
    
    # Create realistic dummy segmentation based on height and position
    labels = np.zeros(num_points, dtype=np.int32)
    scores = np.random.uniform(0.7, 0.95, num_points)
    
    # Simple height-based classification
    z_coords = points[:, 2]
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Ground level classes
    ground_mask = z_coords < 0.1
    labels[ground_mask] = CANONICAL_CLASSES.index("road")
    
    # Sidewalk (slightly elevated)
    sidewalk_mask = (z_coords >= 0.1) & (z_coords < 0.3)
    labels[sidewalk_mask] = CANONICAL_CLASSES.index("sidewalk")
    
    # Buildings (tall structures)
    building_mask = z_coords > 2.0
    labels[building_mask] = CANONICAL_CLASSES.index("building")
    
    # Vegetation (medium height, scattered)
    veg_mask = (z_coords >= 0.3) & (z_coords <= 2.0) & (np.random.random(num_points) < 0.3)
    labels[veg_mask] = CANONICAL_CLASSES.index("vegetation")
    
    # Poles (thin vertical structures)
    pole_mask = (z_coords > 1.5) & (np.random.random(num_points) < 0.05)
    labels[pole_mask] = CANONICAL_CLASSES.index("pole")
    
    # Cars (low height, on ground)
    car_mask = (z_coords >= 0.1) & (z_coords < 1.5) & (np.random.random(num_points) < 0.1)
    labels[car_mask] = CANONICAL_CLASSES.index("car")
    
    # Generate colors
    colors = np.zeros((num_points, 3), dtype=np.float64)
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
    
    # Check if pretrained models are available
    try:
        from .enhanced_infer import PretrainedModelManager
        manager = PretrainedModelManager()
        if len(manager.get_available_models()) > 0:
            methods.append("pretrained")
    except Exception:
        pass
    
    # Check if Open3D-ML is available
    try:
        import open3d.ml.torch
        methods.append("open3d_ml")
    except ImportError:
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


