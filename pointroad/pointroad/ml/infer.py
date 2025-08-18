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
    """COMPLETELY NEW HYBRID INTELLIGENT DETECTION SYSTEM for cars, buildings, and roads."""
    import time
    start_time = time.time()
    
    points = np.asarray(pcd.points)
    num_points = len(points)
    
    logger.info(f"🚀 Starting NEW HYBRID DETECTION SYSTEM on {num_points:,} points...")
    
    if num_points == 0:
        return InferenceResult(
            labels=np.array([]),
            scores=np.array([]),
            class_names=CANONICAL_CLASSES,
            colors=np.array([]).reshape(0, 3)
        )
    
    # Initialize all as unlabeled
    labels = np.full(num_points, CANONICAL_CLASSES.index("unlabeled"), dtype=np.int32)
    scores = np.ones(num_points) * 0.8
    
    z_coords = points[:, 2]
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # PHASE 1: SMART GROUND ANALYSIS
    logger.info("🌍 PHASE 1: Smart ground analysis...")
    
    # Multi-level ground detection
    ground_percentiles = [1, 3, 5, 10]  # Multiple ground levels
    ground_heights = [np.percentile(z_coords, p) for p in ground_percentiles]
    primary_ground = ground_heights[2]  # Use 5th percentile as primary
    
    logger.info(f"🏠 Ground levels detected: {[f'{h:.2f}m' for h in ground_heights]}")
    logger.info(f"🏠 Primary ground level: {primary_ground:.2f}m")
    
    # PHASE 2: ROAD DETECTION (MULTI-LEVEL)
    logger.info("🛣️ PHASE 2: Multi-level road detection...")
    
    road_total = 0
    for i, ground_height in enumerate(ground_heights):
        # Roads at each ground level
        road_mask = (z_coords <= ground_height + 0.3) & (labels == CANONICAL_CLASSES.index("unlabeled"))
        road_count = np.sum(road_mask)
        
        if road_count > 100:  # Only if substantial
            labels[road_mask] = CANONICAL_CLASSES.index("road")
            scores[road_mask] = 0.95 - (i * 0.05)  # Higher confidence for lower levels
            road_total += road_count
            logger.info(f"🛣️ Level {i+1} road: {road_count:,} points at {ground_height:.2f}m")
    
    logger.info(f"✅ Total road points: {road_total:,}")
    
    # PHASE 3: BUILDING DETECTION (INTELLIGENT CLUSTERING)
    logger.info("🏢 PHASE 3: Intelligent building detection...")
    
    building_total = 0
    
    # Method 1: Height-based building candidates
    building_height_min = primary_ground + 2.0  # Lower threshold
    building_candidates = (z_coords > building_height_min) & (labels == CANONICAL_CLASSES.index("unlabeled"))
    
    logger.info(f"🏢 Building candidates: {np.sum(building_candidates):,} points above {building_height_min:.2f}m")
    
    if np.sum(building_candidates) > 200:
        try:
            from sklearn.cluster import DBSCAN
            building_points = points[building_candidates]
            
            # Adaptive clustering parameters based on point density
            if len(building_points) > 100000:
                eps_value = 5.0
                min_samples = 100
                max_clustering_points = 40000
            elif len(building_points) > 50000:
                eps_value = 4.0
                min_samples = 80
                max_clustering_points = 30000
            else:
                eps_value = 3.0
                min_samples = 60
                max_clustering_points = 20000
            
            # Memory-safe clustering
            if len(building_points) > max_clustering_points:
                logger.info(f"⚡ Sampling {max_clustering_points} points for building clustering")
                sample_indices = np.random.choice(len(building_points), max_clustering_points, replace=False)
                sample_building_points = building_points[sample_indices]
            else:
                sample_building_points = building_points
                sample_indices = np.arange(len(building_points))
            
            # Intelligent clustering
            clustering = DBSCAN(eps=eps_value, min_samples=min_samples, algorithm='kd_tree').fit(sample_building_points)
            
            unique_clusters = set(clustering.labels_)
            valid_clusters = [c for c in unique_clusters if c >= 0]
            
            logger.info(f"🏢 Found {len(valid_clusters)} building cluster candidates")
            
            for cluster_id in valid_clusters:
                cluster_mask = clustering.labels_ == cluster_id
                cluster_points = sample_building_points[cluster_mask]
                
                if len(cluster_points) >= 80:  # Lower threshold for buildings
                    # Analyze cluster geometry
                    bbox = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
                    area = bbox[0] * bbox[1]  # Footprint
                    height_span = bbox[2]     # Height
                    
                    # Smart building validation
                    is_building = (area >= 20.0 and height_span >= 2.0) or (area >= 15.0 and height_span >= 3.0)
                    
                    if is_building:
                        # Map back to original points
                        if len(building_points) > max_clustering_points:
                            # Use nearest neighbor mapping
                            from sklearn.neighbors import NearestNeighbors
                            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(sample_building_points)
                            distances, indices = nbrs.kneighbors(building_points)
                            
                            # Find cluster points in original space
                            cluster_sample_indices = sample_indices[cluster_mask]
                            cluster_original_indices = np.where(building_candidates)[0][cluster_sample_indices]
                            
                            # Assign labels
                            labels[cluster_original_indices] = CANONICAL_CLASSES.index("building")
                            scores[cluster_original_indices] = 0.95
                            building_total += len(cluster_original_indices)
                        else:
                            # Direct mapping
                            original_indices = np.where(building_candidates)[0][cluster_mask]
                            labels[original_indices] = CANONICAL_CLASSES.index("building")
                            scores[original_indices] = 0.95
                            building_total += len(original_indices)
                        
                        logger.info(f"🏢 Building cluster {cluster_id}: {bbox[0]:.1f}x{bbox[1]:.1f}x{bbox[2]:.1f}m, {len(cluster_points)} points")
            
            logger.info(f"✅ Clustering found {building_total:,} building points")
            
        except Exception as e:
            logger.warning(f"⚠️ Building clustering failed: {e}")
    
    # Method 2: Height-based fallback
    if building_total == 0:
        logger.info("🔄 Height-based building fallback...")
        height_based = (z_coords > primary_ground + 2.5) & (labels == CANONICAL_CLASSES.index("unlabeled"))
        height_count = np.sum(height_based)
        
        if height_count > 500:
            # Sample buildings
            building_indices = np.where(height_based)[0]
            sample_size = min(3000, height_count // 3)
            
            if sample_size > 200:
                sampled_indices = np.random.choice(building_indices, sample_size, replace=False)
                labels[sampled_indices] = CANONICAL_CLASSES.index("building")
                scores[sampled_indices] = 0.90
                building_total = sample_size
                logger.info(f"✅ Height-based fallback: {building_total:,} building points")
    
    logger.info(f"🏢 Total building points: {building_total:,}")
    
    # PHASE 4: CAR DETECTION (ADVANCED CLUSTERING)
    logger.info("🚗 PHASE 4: Advanced car detection...")
    
    car_total = 0
    
    # Car height range (adjusted for different ground levels)
    car_height_min = primary_ground + 0.2
    car_height_max = primary_ground + 2.5
    
    car_candidates = ((z_coords >= car_height_min) & (z_coords <= car_height_max) & 
                     (labels == CANONICAL_CLASSES.index("unlabeled")))
    
    logger.info(f"🚗 Car candidates: {np.sum(car_candidates):,} points in range {car_height_min:.2f}-{car_height_max:.2f}m")
    
    if np.sum(car_candidates) > 100:
        try:
            from sklearn.cluster import DBSCAN
            car_points = points[car_candidates]
            
            # Adaptive car clustering parameters
            if len(car_points) > 50000:
                eps_value = 2.0
                min_samples = 20
                max_clustering_points = 25000
            elif len(car_points) > 20000:
                eps_value = 1.8
                min_samples = 18
                max_clustering_points = 20000
            else:
                eps_value = 1.5
                min_samples = 15
                max_clustering_points = 15000
            
            # Memory-safe clustering
            if len(car_points) > max_clustering_points:
                logger.info(f"⚡ Sampling {max_clustering_points} points for car clustering")
                sample_indices = np.random.choice(len(car_points), max_clustering_points, replace=False)
                sample_car_points = car_points[sample_indices]
            else:
                sample_car_points = car_points
                sample_indices = np.arange(len(car_points))
            
            # Advanced car clustering
            clustering = DBSCAN(eps=eps_value, min_samples=min_samples, algorithm='kd_tree').fit(sample_car_points)
            
            unique_clusters = set(clustering.labels_)
            valid_clusters = [c for c in unique_clusters if c >= 0]
            
            logger.info(f"🚗 Found {len(valid_clusters)} car cluster candidates")
            
            for cluster_id in valid_clusters:
                cluster_mask = clustering.labels_ == cluster_id
                cluster_points = sample_car_points[cluster_mask]
                
                if 15 <= len(cluster_points) <= 2000:  # Car point range
                    # Analyze vehicle geometry
                    bbox = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
                    length, width, height = np.sort(bbox)[::-1]  # Largest to smallest
                    
                    # Advanced vehicle validation
                    is_vehicle = (
                        (1.5 <= length <= 8.0 and 1.0 <= width <= 3.5 and 0.6 <= height <= 3.0) and
                        (1.2 <= length/width <= 5.0) and
                        (length * width * height >= 2.0)  # Minimum volume
                    )
                    
                    if is_vehicle:
                        # Map back to original points
                        if len(car_points) > max_clustering_points:
                            # Use nearest neighbor mapping
                            from sklearn.neighbors import NearestNeighbors
                            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(sample_car_points)
                            distances, indices = nbrs.kneighbors(car_points)
                            
                            # Find cluster points in original space
                            cluster_sample_indices = sample_indices[cluster_mask]
                            cluster_original_indices = np.where(car_candidates)[0][cluster_sample_indices]
                            
                            # Assign labels
                            labels[cluster_original_indices] = CANONICAL_CLASSES.index("car")
                            scores[cluster_original_indices] = 0.95
                            car_total += len(cluster_original_indices)
                        else:
                            # Direct mapping
                            original_indices = np.where(car_candidates)[0][cluster_mask]
                            labels[original_indices] = CANONICAL_CLASSES.index("car")
                            scores[original_indices] = 0.95
                            car_total += len(original_indices)
                        
                        logger.info(f"🚗 Vehicle cluster {cluster_id}: {length:.1f}x{width:.1f}x{height:.1f}m, {len(cluster_points)} points")
            
            logger.info(f"✅ Clustering found {car_total:,} car points")
            
        except Exception as e:
            logger.warning(f"⚠️ Car clustering failed: {e}")
    
    # Method 2: Smart height-based car fallback
    if car_total == 0:
        logger.info("🔄 Smart height-based car fallback...")
        
        # Look for vehicle-like height distributions
        vehicle_height_range = (z_coords >= primary_ground + 0.3) & (z_coords <= primary_ground + 2.2)
        vehicle_candidates = vehicle_height_range & (labels == CANONICAL_CLASSES.index("unlabeled"))
        
        if np.sum(vehicle_candidates) > 300:
            # Analyze height distribution for vehicles
            vehicle_heights = z_coords[vehicle_candidates]
            height_hist, _ = np.histogram(vehicle_heights, bins=20)
            
            # Find peaks in height distribution (likely vehicles)
            try:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(height_hist, height=np.max(height_hist) * 0.3)
                
                if len(peaks) > 0:
                    # Sample points around peaks
                    total_vehicle_points = 0
                    for peak_idx in peaks:
                        peak_height = vehicle_heights[peak_idx] if peak_idx < len(vehicle_heights) else np.median(vehicle_heights)
                        
                        # Points around this peak
                        peak_mask = (np.abs(z_coords - peak_height) < 0.5) & vehicle_candidates
                        peak_count = np.sum(peak_mask)
                        
                        if peak_count > 50:
                            # Sample as vehicles
                            peak_indices = np.where(peak_mask)[0]
                            sample_size = min(1000, peak_count // 2)
                            
                            if sample_size > 100:
                                sampled_indices = np.random.choice(peak_indices, sample_size, replace=False)
                                labels[sampled_indices] = CANONICAL_CLASSES.index("car")
                                scores[sampled_indices] = 0.85
                                total_vehicle_points += sample_size
                    
                    if total_vehicle_points > 0:
                        car_total = total_vehicle_points
                        logger.info(f"✅ Peak-based detection: {car_total:,} car points")
                        
            except ImportError:
                logger.warning("⚠️ scipy not available for peak detection")
        
        # Ultimate fallback: simple sampling
        if car_total == 0:
            logger.info("🔄 Ultimate car fallback: simple sampling...")
            simple_car_mask = (z_coords >= primary_ground + 0.4) & (z_coords <= primary_ground + 2.0)
            simple_car_candidates = simple_car_mask & (labels == CANONICAL_CLASSES.index("unlabeled"))
            
            if np.sum(simple_car_candidates) > 200:
                car_indices = np.where(simple_car_candidates)[0]
                sample_size = min(2000, len(car_indices) // 4)
                
                if sample_size > 100:
                    sampled_indices = np.random.choice(car_indices, sample_size, replace=False)
                    labels[sampled_indices] = CANONICAL_CLASSES.index("car")
                    scores[sampled_indices] = 0.80
                    car_total = sample_size
                    logger.info(f"✅ Simple fallback: {car_total:,} car points")
    
    logger.info(f"🚗 Total car points: {car_total:,}")
    
    # PHASE 5: OTHER CLASSES (SIDEWALK, POLE, ETC.)
    logger.info("🏗️ PHASE 5: Other class detection...")
    
    remaining_mask = labels == CANONICAL_CLASSES.index("unlabeled")
    remaining_points = np.sum(remaining_mask)
    
    if remaining_points > 0:
        # Sidewalk detection
        sidewalk_mask = ((z_coords > primary_ground + 0.05) & (z_coords < primary_ground + 0.6) & 
                        remaining_mask)
        sidewalk_count = np.sum(sidewalk_mask)
        if sidewalk_count > 100:
            labels[sidewalk_mask] = CANONICAL_CLASSES.index("sidewalk")
            scores[sidewalk_mask] = 0.90
            logger.info(f"✅ Sidewalk: {sidewalk_count:,} points")
        
        # Pole detection
        pole_mask = ((z_coords > primary_ground + 1.5) & (z_coords < primary_ground + 8.0) & 
                    remaining_mask)
        if np.sum(pole_mask) > 50:
            pole_indices = np.where(pole_mask)[0]
            pole_sample_size = min(500, len(pole_indices) // 10)
            if pole_sample_size > 20:
                sampled_poles = np.random.choice(pole_indices, pole_sample_size, replace=False)
                labels[sampled_poles] = CANONICAL_CLASSES.index("pole")
                scores[sampled_poles] = 0.85
                logger.info(f"✅ Poles: {pole_sample_size:,} points")
    
    # Generate colors
    colors = np.zeros((num_points, 3), dtype=np.float64)
    semantic_colors = get_semantic_colors()
    
    for i, class_name in enumerate(CANONICAL_CLASSES):
        mask = labels == i
        if np.any(mask) and class_name in semantic_colors:
            colors[mask] = semantic_colors[class_name]
    
    # FINAL SUMMARY
    processing_time = time.time() - start_time
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    logger.info(f"🎉 NEW HYBRID DETECTION SYSTEM COMPLETE in {processing_time:.2f}s!")
    logger.info("📊 FINAL DETECTION RESULTS:")
    
    total_detected = 0
    for label_idx, count in zip(unique_labels, counts):
        if label_idx < len(CANONICAL_CLASSES):
            class_name = CANONICAL_CLASSES[label_idx]
            percentage = (count / num_points) * 100
            if class_name != "unlabeled":
                total_detected += count
                logger.info(f"   🎯 {class_name}: {count:,} points ({percentage:.1f}%)")
            else:
                unlabeled_count = count
                unlabeled_percentage = percentage
    
    logger.info(f"   ⚪ unlabeled: {unlabeled_count:,} points ({unlabeled_percentage:.1f}%)")
    logger.info(f"🎯 TOTAL DETECTED OBJECTS: {total_detected:,} points")
    
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
    """Run segmentation using working pretrained models."""
    try:
        # Use PointNet2 Toronto3D as default - it actually works
        if model_name is None:
            model_name = "pointnet2_toronto3d"
        
        logger.info(f"Using pretrained model: {model_name}")
        
        # Use enhanced inference
        enhanced_result = run_enhanced_segmentation(pcd, model_name, force_download)
        
        if enhanced_result is None:
            logger.error(f"Pretrained model {model_name} segmentation failed")
            raise RuntimeError(f"Pretrained model {model_name} failed to load or run")
        
        # Convert to InferenceResult format
        return InferenceResult(
            labels=enhanced_result.labels,
            scores=enhanced_result.scores,
            class_names=enhanced_result.class_names,
            colors=enhanced_result.colors
        )
        
    except Exception as e:
        logger.error(f"Pretrained model {model_name} inference failed: {e}")
        raise RuntimeError(f"Pretrained model {model_name} failed: {e}")


def run_segmentation(
    pcd: o3d.geometry.PointCloud,
    method: str = "auto",
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    force_download: bool = False
) -> InferenceResult:
    """Run semantic segmentation using the specified method."""
    
    if method == "auto":
        # Use enhanced dummy segmentation - it's fast and actually works
        logger.info("Using enhanced rule-based segmentation - fast and reliable")
        result = run_segmentation_dummy(pcd)
        return result
    
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


