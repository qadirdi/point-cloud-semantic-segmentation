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
    get_discrete_classes, CANONICAL_CLASSES
)


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
            pred = model(torch.from_numpy(points_normalized).unsqueeze(0))
            pred = pred.softmax(dim=-1)
            scores, labels = pred.max(dim=-1)
        
        labels = labels.numpy().flatten().astype(np.int32)
        scores = scores.numpy().flatten().astype(np.float32)
        
        # Map to canonical classes
        canonical_labels = map_kitti_to_canonical(labels)
        
        # Generate colors
        colors = np.zeros((len(points), 3), dtype=np.float64)
        semantic_colors = get_semantic_colors()
        
        for i, class_name in enumerate(CANONICAL_CLASSES):
            mask = canonical_labels == i
            if np.any(mask) and class_name in semantic_colors:
                colors[mask] = semantic_colors[class_name]
        
        return InferenceResult(
            labels=canonical_labels,
            scores=scores,
            class_names=CANONICAL_CLASSES,
            colors=colors
        )
        
    except ImportError:
        logger.warning("Open3D-ML not available, falling back to dummy segmentation")
        return run_segmentation_dummy(pcd)
    except Exception as e:
        logger.error(f"Open3D-ML inference failed: {e}")
        return run_segmentation_dummy(pcd)


def run_segmentation_onnx(pcd: o3d.geometry.PointCloud, model_path: str) -> InferenceResult:
    """Run segmentation using ONNX model."""
    try:
        import onnxruntime as ort
        
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Normalize points to [-1, 1] range
        points_centered = points - points.mean(axis=0)
        points_normalized = points_centered / (points_centered.std(axis=0) + 1e-8)
        
        # Create session
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get input name
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference
        pred = session.run([output_name], {input_name: points_normalized.reshape(1, -1, 3)})[0]
        
        # Get labels and scores
        labels = pred.argmax(axis=-1).flatten().astype(np.int32)
        scores = pred.max(axis=-1).flatten().astype(np.float32)
        
        # Map KITTI labels to canonical classes
        canonical_labels = map_kitti_to_canonical(labels)
        
        # Generate colors
        colors = np.zeros((len(points), 3), dtype=np.float64)
        semantic_colors = get_semantic_colors()
        
        for i, class_name in enumerate(CANONICAL_CLASSES):
            mask = canonical_labels == i
            if np.any(mask) and class_name in semantic_colors:
                colors[mask] = semantic_colors[class_name]
        
        return InferenceResult(
            labels=canonical_labels,
            scores=scores,
            class_names=CANONICAL_CLASSES,
            colors=colors
        )
        
    except ImportError:
        logger.warning("ONNX Runtime not available, falling back to dummy segmentation")
        return run_segmentation_dummy(pcd)
    except Exception as e:
        logger.error(f"ONNX inference failed: {e}")
        return run_segmentation_dummy(pcd)


def run_segmentation(
    pcd: o3d.geometry.PointCloud,
    backend: str = "dummy",
    model_path: Optional[str] = None
) -> InferenceResult:
    """Run semantic segmentation on point cloud."""
    logger.info(f"Running semantic segmentation with backend: {backend}")
    
    if backend == "open3d_ml":
        if model_path is None:
            logger.warning("No model path provided for Open3D-ML, using dummy")
            return run_segmentation_dummy(pcd)
        return run_segmentation_open3d_ml(pcd, model_path)
    
    elif backend == "onnx":
        if model_path is None:
            logger.warning("No model path provided for ONNX, using dummy")
            return run_segmentation_dummy(pcd)
        return run_segmentation_onnx(pcd, model_path)
    
    elif backend == "dummy":
        return run_segmentation_dummy(pcd)
    
    else:
        logger.warning(f"Unknown backend {backend}, using dummy")
        return run_segmentation_dummy(pcd)


def get_class_statistics(result: InferenceResult) -> dict:
    """Get statistics about detected classes."""
    unique_labels, counts = np.unique(result.labels, return_counts=True)
    
    stats = {}
    for label, count in zip(unique_labels, counts):
        if label < len(result.class_names):
            class_name = result.class_names[label]
            stats[class_name] = {
                "count": int(count),
                "percentage": float(count / len(result.labels) * 100),
                "mean_score": float(result.scores[result.labels == label].mean())
            }
    
    return stats


def filter_by_class(
    result: InferenceResult,
    class_names: list[str],
    min_score: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter points by class and confidence score."""
    class_indices = [result.class_names.index(name) for name in class_names if name in result.class_names]
    
    if not class_indices:
        return np.array([]), np.array([]), np.array([])
    
    mask = np.isin(result.labels, class_indices) & (result.scores >= min_score)
    
    return result.labels[mask], result.scores[mask], result.colors[mask]


__all__ = [
    "InferenceResult",
    "run_segmentation",
    "run_segmentation_dummy", 
    "run_segmentation_open3d_ml",
    "run_segmentation_onnx",
    "get_class_statistics",
    "filter_by_class"
]


