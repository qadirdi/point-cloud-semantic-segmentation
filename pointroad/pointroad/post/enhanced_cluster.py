"""Enhanced clustering module with car-specific optimization."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os
import yaml
from pathlib import Path
from loguru import logger

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using simplified clustering")

from .cluster import InstanceInfo


def load_config() -> Dict[str, Any]:
    """Load enhanced detection configuration."""
    # Look for config in multiple locations
    config_paths = [
        # Current directory
        Path("enhanced_detection.yaml"),
        # Config directory
        Path(__file__).parent.parent / "config" / "enhanced_detection.yaml",
        # User-specified path via environment variable
        Path(os.environ.get("POINTROAD_CONFIG", "")) / "enhanced_detection.yaml"
    ]
    
    for path in config_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded enhanced detection config from {path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")
    
    # Return default config if no file found
    logger.info("Using default detection parameters")
    return {
        "car_detection": {
            "clustering": {
                "eps": 0.4,               # Increased for better connectivity
                "min_points": 15,         # Reduced for better small car detection
                "confidence_threshold": 0.50  # Lowered for better recall
            },
            "dimensions": {
                "length_range": [2.5, 7.5],  # Wider range for all vehicle types
                "width_range": [1.3, 3.0],   # More flexible width
                "height_range": [1.0, 2.8],  # More flexible height
                "volume_range": [4.0, 40.0], # Wider volume range
                "aspect_ratio_range": [1.2, 4.5]  # More flexible ratios
            }
        }
    }


def simple_distance_clustering(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Simple distance-based clustering fallback when sklearn is not available."""
    n_points = len(points)
    labels = np.full(n_points, -1)  # Initialize all as noise
    
    if n_points < min_points:
        return labels
    
    cluster_id = 0
    visited = np.zeros(n_points, dtype=bool)
    
    for i in range(n_points):
        if visited[i]:
            continue
            
        visited[i] = True
        
        # Find neighbors within eps distance
        distances = np.linalg.norm(points - points[i], axis=1)
        neighbors = np.where(distances <= eps)[0]
        
        if len(neighbors) >= min_points:
            # Start new cluster
            labels[i] = cluster_id
            
            # Expand cluster
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                neighbor_idx = seed_set[j]
                
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    
                    # Find neighbors of this neighbor
                    neighbor_distances = np.linalg.norm(points - points[neighbor_idx], axis=1)
                    neighbor_neighbors = np.where(neighbor_distances <= eps)[0]
                    
                    if len(neighbor_neighbors) >= min_points:
                        seed_set.extend(neighbor_neighbors)
                
                if labels[neighbor_idx] == -1:  # Not yet assigned to a cluster
                    labels[neighbor_idx] = cluster_id
                
                j += 1
            
            cluster_id += 1
    
    return labels


@dataclass
class EnhancedInstanceInfo:
    """Enhanced instance information with car-specific features."""
    instance_id: int
    class_name: str
    num_points: int
    centroid: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    extent: np.ndarray
    confidence_score: float
    
    # Car-specific features
    car_confidence: float
    geometric_features: Dict[str, float]
    is_likely_car: bool
    car_dimensions: Dict[str, float]


def analyze_car_dimensions(points: np.ndarray) -> Dict[str, float]:
    """Analyze dimensions to determine if cluster represents a car."""
    if len(points) < 10:
        return {"length": 0, "width": 0, "height": 0, "volume": 0, "aspect_ratio": 0}
    
    # Calculate oriented bounding box for better dimension analysis
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        obb = pcd.get_oriented_bounding_box()
        extent = obb.extent
        
        # Sort dimensions to get length, width, height
        sorted_dims = np.sort(extent)[::-1]  # Descending order
        length, width, height = sorted_dims[0], sorted_dims[1], sorted_dims[2]
        
    except Exception:
        # Fallback to axis-aligned bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        extent = max_coords - min_coords
        sorted_dims = np.sort(extent)[::-1]
        length, width, height = sorted_dims[0], sorted_dims[1], sorted_dims[2]
    
    volume = length * width * height
    aspect_ratio = length / width if width > 0 else 0
    
    return {
        "length": float(length),
        "width": float(width), 
        "height": float(height),
        "volume": float(volume),
        "aspect_ratio": float(aspect_ratio)
    }


def is_car_like_cluster(points: np.ndarray, dimensions: Dict[str, float], car_confidence: float) -> Tuple[bool, float]:
    """Determine if a cluster is likely to be a car based on multiple criteria."""
    
    # Load configuration
    config = load_config()
    car_config = config.get("car_detection", {})
    dim_config = car_config.get("dimensions", {})
    
    # More flexible car dimensions (in meters) - covers sedans to SUVs/trucks
    CAR_LENGTH_RANGE = tuple(dim_config.get("length_range", [2.5, 7.5]))  # Compact cars to large SUVs
    CAR_WIDTH_RANGE = tuple(dim_config.get("width_range", [1.3, 3.0]))   # Wider range for different vehicle types
    CAR_HEIGHT_RANGE = tuple(dim_config.get("height_range", [1.0, 2.8]))  # Lower cars to high SUVs/trucks
    CAR_VOLUME_RANGE = tuple(dim_config.get("volume_range", [4.0, 40.0])) # Wider volume range
    CAR_ASPECT_RATIO_RANGE = tuple(dim_config.get("aspect_ratio_range", [1.2, 4.5])) # More flexible ratios
    
    # Lower confidence threshold for better recall
    confidence_threshold = car_config.get("clustering", {}).get("confidence_threshold", 0.55)
    
    length = dimensions["length"]
    width = dimensions["width"]
    height = dimensions["height"]
    volume = dimensions["volume"]
    aspect_ratio = dimensions["aspect_ratio"]
    
    # Score each dimension
    scores = []
    
    # Length score
    if CAR_LENGTH_RANGE[0] <= length <= CAR_LENGTH_RANGE[1]:
        length_score = 1.0
    elif length < CAR_LENGTH_RANGE[0]:
        length_score = max(0, length / CAR_LENGTH_RANGE[0])
    else:
        length_score = max(0, CAR_LENGTH_RANGE[1] / length)
    scores.append(length_score)
    
    # Width score
    if CAR_WIDTH_RANGE[0] <= width <= CAR_WIDTH_RANGE[1]:
        width_score = 1.0
    elif width < CAR_WIDTH_RANGE[0]:
        width_score = max(0, width / CAR_WIDTH_RANGE[0])
    else:
        width_score = max(0, CAR_WIDTH_RANGE[1] / width)
    scores.append(width_score)
    
    # Height score
    if CAR_HEIGHT_RANGE[0] <= height <= CAR_HEIGHT_RANGE[1]:
        height_score = 1.0
    elif height < CAR_HEIGHT_RANGE[0]:
        height_score = max(0, height / CAR_HEIGHT_RANGE[0])
    else:
        height_score = max(0, CAR_HEIGHT_RANGE[1] / height)
    scores.append(height_score)
    
    # Volume score
    if CAR_VOLUME_RANGE[0] <= volume <= CAR_VOLUME_RANGE[1]:
        volume_score = 1.0
    elif volume < CAR_VOLUME_RANGE[0]:
        volume_score = max(0, volume / CAR_VOLUME_RANGE[0])
    else:
        volume_score = max(0, CAR_VOLUME_RANGE[1] / volume)
    scores.append(volume_score)
    
    # Aspect ratio score
    if CAR_ASPECT_RATIO_RANGE[0] <= aspect_ratio <= CAR_ASPECT_RATIO_RANGE[1]:
        aspect_score = 1.0
    elif aspect_ratio < CAR_ASPECT_RATIO_RANGE[0]:
        aspect_score = max(0, aspect_ratio / CAR_ASPECT_RATIO_RANGE[0])
    else:
        aspect_score = max(0, CAR_ASPECT_RATIO_RANGE[1] / aspect_ratio)
    scores.append(aspect_score)
    
    # Point density score (more flexible for varying scan densities)
    point_density = len(points) / volume if volume > 0 else 0
    if 20 <= point_density <= 800:  # More flexible density range for different scan resolutions
        density_score = 1.0
    elif point_density < 20:
        density_score = point_density / 20  # Gradually decrease score for sparse scans
    else:
        density_score = 800 / point_density  # Gradually decrease score for very dense scans
    scores.append(density_score)
    
    # Combine geometric scores
    geometric_score = np.mean(scores)
    
    # Combine with car confidence from segmentation
    final_score = 0.6 * geometric_score + 0.4 * car_confidence
    
    # Use configurable threshold for car classification
    is_car = final_score > confidence_threshold
    
    logger.debug(f"Car analysis - Dims: {length:.1f}x{width:.1f}x{height:.1f}, "
                f"Density: {point_density:.1f}, Scores: geo={geometric_score:.3f}, "
                f"conf={car_confidence:.3f}, final={final_score:.3f}, is_car={is_car}")
    
    return is_car, final_score


def enhanced_car_clustering(pcd: o3d.geometry.PointCloud, 
                           car_confidence_scores: np.ndarray,
                           car_mask: np.ndarray,
                           eps: float = None,
                           min_points: int = None) -> Tuple[np.ndarray, List[EnhancedInstanceInfo]]:
    """Enhanced clustering specifically optimized for car detection."""
    # Load configuration
    config = load_config()
    car_config = config.get("car_detection", {}).get("clustering", {})
    
    # Use provided parameters or fall back to config values
    if eps is None:
        eps = car_config.get("eps", 0.25)  # Use tighter clustering by default
    if min_points is None:
        min_points = car_config.get("min_points", 20)  # Reduced minimum points
    
    points = np.asarray(pcd.points)
    total_points = len(points)
    
    if total_points == 0 or not np.any(car_mask):
        return np.full(total_points, -1), []
    
    logger.info(f"Running enhanced car clustering on {np.sum(car_mask):,} potential car points")
    
    # Extract potential car points
    car_points = points[car_mask]
    car_confidences = car_confidence_scores[car_mask]
    
    # Optimized clustering parameters for better car detection
    car_eps = eps * 0.8  # Slightly looser clustering to catch separated car parts
    car_min_points = max(10, min_points // 2)  # Reduced minimum points for better detection of smaller cars
    
    # Run DBSCAN on potential car points
    if SKLEARN_AVAILABLE:
        clustering = DBSCAN(eps=car_eps, min_samples=car_min_points).fit(car_points)
        car_labels = clustering.labels_
    else:
        # Fallback: use simplified clustering based on distance
        car_labels = simple_distance_clustering(car_points, car_eps, car_min_points)
    
    # Map back to full point cloud
    full_labels = np.full(total_points, -1)
    full_labels[car_mask] = car_labels
    
    # Analyze each cluster
    enhanced_instances = []
    unique_labels = np.unique(car_labels)
    valid_labels = unique_labels[unique_labels >= 0]
    
    logger.info(f"Found {len(valid_labels)} potential car clusters")
    
    for cluster_id in valid_labels:
        # Get cluster points
        cluster_mask_in_cars = car_labels == cluster_id
        cluster_indices = np.where(car_mask)[0][cluster_mask_in_cars]
        cluster_points = points[cluster_indices]
        cluster_confidences = car_confidence_scores[cluster_indices]
        
        if len(cluster_points) < car_min_points:
            continue
        
        # Analyze dimensions
        dimensions = analyze_car_dimensions(cluster_points)
        
        # Calculate mean confidence
        mean_confidence = np.mean(cluster_confidences)
        
        # Determine if this is likely a car
        is_car, car_score = is_car_like_cluster(cluster_points, dimensions, mean_confidence)
        
        # Calculate other properties
        centroid = np.mean(cluster_points, axis=0)
        min_coords = np.min(cluster_points, axis=0)
        max_coords = np.max(cluster_points, axis=0)
        extent = max_coords - min_coords
        
        # Geometric features
        geometric_features = {
            "point_density": len(cluster_points) / dimensions["volume"] if dimensions["volume"] > 0 else 0,
            "compactness": dimensions["volume"] / (dimensions["length"] * dimensions["width"] * dimensions["height"]) if all(dimensions[k] > 0 for k in ["length", "width", "height"]) else 0,
            "elongation": dimensions["aspect_ratio"],
            "height_ratio": dimensions["height"] / dimensions["length"] if dimensions["length"] > 0 else 0
        }
        
        # Create enhanced instance info
        instance = EnhancedInstanceInfo(
            instance_id=int(cluster_id),
            class_name="car",
            num_points=len(cluster_points),
            centroid=centroid,
            aabb_min=min_coords,
            aabb_max=max_coords,
            extent=extent,
            confidence_score=float(np.mean(car_confidence_scores[cluster_indices])),
            car_confidence=float(mean_confidence),
            geometric_features=geometric_features,
            is_likely_car=is_car,
            car_dimensions=dimensions
        )
        
        enhanced_instances.append(instance)
        
        # Update labels based on car likelihood
        if not is_car:
            # If not car-like, mark as noise
            full_labels[cluster_indices] = -1
    
    # Filter out non-car clusters
    car_instances = [inst for inst in enhanced_instances if inst.is_likely_car]
    
    logger.info(f"Enhanced car clustering complete: {len(car_instances)} high-confidence cars detected")
    
    return full_labels, car_instances


def memory_safe_clustering(points: np.ndarray, eps: float, min_pts: int, class_name: str) -> np.ndarray:
    """Perform memory-safe clustering to prevent allocation errors."""
    max_points_for_clustering = 120000  # Increased for better precision while maintaining safety
    
    if len(points) <= max_points_for_clustering:
        # Safe to cluster directly
        if SKLEARN_AVAILABLE:
            try:
                clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(points)
                return clustering.labels_
            except MemoryError:
                logger.warning(f"DBSCAN memory error for {class_name}, using fallback")
                return simple_distance_clustering(points, eps, min_pts)
        else:
            return simple_distance_clustering(points, eps, min_pts)
    else:
        # Too many points - use sampling approach
        logger.info(f"Class {class_name} has {len(points):,} points, using memory-safe sampling")
        
        # Sample points for clustering
        sample_indices = np.random.choice(len(points), max_points_for_clustering, replace=False)
        sample_points = points[sample_indices]
        
        try:
            if SKLEARN_AVAILABLE:
                # Cluster the sample
                clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(sample_points)
                sample_labels = clustering.labels_
                
                # Map results back to full point cloud using nearest neighbors
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(sample_points)
                distances, indices = nbrs.kneighbors(points)
                
                # Map cluster labels back
                full_labels = sample_labels[indices.flatten()]
                
                # Filter out points that are too far from their nearest cluster center
                max_distance = eps * 2.0  # Points further than this are considered noise
                noise_mask = distances.flatten() > max_distance
                full_labels[noise_mask] = -1
                
                logger.info(f"Memory-safe clustering: {len(np.unique(full_labels)) - 1} clusters found")
                return full_labels
            else:
                return simple_distance_clustering(points, eps, min_pts)
                
        except MemoryError as e:
            logger.error(f"Memory error in clustering {class_name}: {e}")
            # Fallback: create simple clusters based on spatial proximity
            return simple_spatial_clustering(points, eps, min_pts)


def simple_spatial_clustering(points: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Ultra-simple clustering based on spatial grids to avoid memory issues."""
    logger.info("Using ultra-safe grid-based clustering")
    
    # Create spatial grid
    grid_size = eps * 2.0
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # Calculate grid dimensions
    grid_dims = np.ceil((max_coords - min_coords) / grid_size).astype(int)
    
    # Assign points to grid cells
    grid_indices = np.floor((points - min_coords) / grid_size).astype(int)
    
    # Convert multi-dimensional indices to single index
    flat_indices = np.ravel_multi_index(
        grid_indices.T, grid_dims, mode='clip'
    )
    
    # Count points in each cell
    unique_cells, counts = np.unique(flat_indices, return_counts=True)
    
    # Create cluster labels
    labels = np.full(len(points), -1)
    cluster_id = 0
    
    for cell_idx, count in zip(unique_cells, counts):
        if count >= min_pts:
            cell_mask = flat_indices == cell_idx
            labels[cell_mask] = cluster_id
            cluster_id += 1
    
    logger.info(f"Grid-based clustering found {cluster_id} clusters")
    return labels


def enhanced_clustering_all_classes(pcd: o3d.geometry.PointCloud,
                                  labels: np.ndarray,
                                  class_names: List[str],
                                  car_confidence_scores: np.ndarray = None,
                                  eps_by_class: Optional[Dict[str, float]] = None,
                                  min_points_by_class: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, List[EnhancedInstanceInfo]]:
    """Enhanced clustering for all classes with car-specific optimization."""
    # Load configuration
    config = load_config()
    
    # If car confidence scores not provided, use dummy ones
    if car_confidence_scores is None:
        car_confidence_scores = np.ones(len(labels)) * 0.8
    
    points = np.asarray(pcd.points)
    total_points = len(points)
    
    if total_points == 0:
        return np.full(total_points, -1), []
    
    # Default parameters optimized for each class
    if eps_by_class is None:
        eps_by_class = {
            "car": config.get("car_detection", {}).get("clustering", {}).get("eps", 0.4),  # Optimized for cars
            "building": 1.2,    # Increased for large complex buildings (like L-shapes)
            "road": 2.0,       # Very loose for road segments
            "sidewalk": 0.8,   # Medium for sidewalks
            "pole": 0.3,       # Tighter for poles
            "unlabeled": 0.5   # Default
        }
    
    # Default minimum points per class - optimized for better detection
    default_min_points = {
        "car": config.get("car_detection", {}).get("clustering", {}).get("min_points", 15),  # Further reduced for smaller cars
        "building": 80,    # Increased for substantial buildings
        "road": 150,       # Much higher for meaningful road segments
        "sidewalk": 30,    # Medium for sidewalks
        "pole": 10,        # Lower for thin poles
        "unlabeled": 15    # Default - lower threshold
    }
    
    # Use provided min_points or merge defaults with config
    if min_points_by_class is None:
        min_points_by_class = {}
        for class_name, default_min in default_min_points.items():
            class_config = config.get("class_parameters", {}).get(class_name, {})
            min_points_by_class[class_name] = class_config.get("min_points", default_min)
    
    all_cluster_labels = np.full(total_points, -1)
    all_instances = []
    current_cluster_id = 0
    
    for class_idx, class_name in enumerate(class_names):
        class_mask = labels == class_idx
        if not np.any(class_mask):
            continue
        
        class_points = points[class_mask]
        logger.info(f"Clustering {class_name}: {len(class_points):,} points")
        
        # Get class-specific parameters
        eps = eps_by_class.get(class_name, 0.5)
        min_pts = min_points_by_class.get(class_name, 20)
        
        if class_name == "car":
            # Use enhanced car clustering
            car_labels, car_instances = enhanced_car_clustering(
                pcd, car_confidence_scores, class_mask, eps, min_pts
            )
            
            # Update cluster IDs to be unique
            for instance in car_instances:
                instance.instance_id = current_cluster_id
                current_cluster_id += 1
                
                # Update labels
                instance_mask = (car_labels == instance.instance_id) & class_mask
                all_cluster_labels[instance_mask] = instance.instance_id
            
            all_instances.extend(car_instances)
            
        else:
            # Memory-safe clustering for other classes
            if len(class_points) >= min_pts:
                cluster_labels = memory_safe_clustering(class_points, eps, min_pts, class_name)
                
                unique_labels = np.unique(cluster_labels)
                valid_labels = unique_labels[unique_labels >= 0]
                
                for cluster_id in valid_labels:
                    cluster_mask_in_class = cluster_labels == cluster_id
                    cluster_indices = np.where(class_mask)[0][cluster_mask_in_class]
                    cluster_points = points[cluster_indices]
                    
                    # Calculate properties
                    centroid = np.mean(cluster_points, axis=0)
                    min_coords = np.min(cluster_points, axis=0)
                    max_coords = np.max(cluster_points, axis=0)
                    extent = max_coords - min_coords
                    
                    # Create standard instance info
                    instance = EnhancedInstanceInfo(
                        instance_id=current_cluster_id,
                        class_name=class_name,
                        num_points=len(cluster_points),
                        centroid=centroid,
                        aabb_min=min_coords,
                        aabb_max=max_coords,
                        extent=extent,
                        confidence_score=0.8,  # Default confidence
                        car_confidence=0.0,    # Not a car
                        geometric_features={},
                        is_likely_car=False,
                        car_dimensions={"length": 0, "width": 0, "height": 0, "volume": 0, "aspect_ratio": 0}
                    )
                    
                    all_instances.append(instance)
                    
                    # Update labels
                    all_cluster_labels[cluster_indices] = current_cluster_id
                    current_cluster_id += 1
    
    logger.info(f"Enhanced clustering complete: {len(all_instances)} instances found "
               f"({len([i for i in all_instances if i.class_name == 'car'])} cars)")
    
    return all_cluster_labels, all_instances


__all__ = [
    "EnhancedInstanceInfo",
    "enhanced_car_clustering",
    "enhanced_clustering_all_classes",
    "analyze_car_dimensions",
    "is_car_like_cluster",
    "load_config"
]