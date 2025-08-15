#!/usr/bin/env python3
"""Model loader for pretrained semantic segmentation models."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import open3d as o3d
from loguru import logger

# Toronto3D class definitions (more comprehensive than SemanticKITTI)
TORONTO3D_CLASSES = {
    0: "unlabeled",
    1: "road",
    2: "road_marking",
    3: "natural",
    4: "building",
    5: "utility_line",
    6: "pole",
    7: "car",
    8: "fence",
    9: "traffic_sign",
    10: "traffic_light",
    11: "vegetation",
    12: "terrain",
    13: "other_ground",
    14: "other_object"
}

# SemanticKITTI class definitions (keeping for compatibility)
SEMANTICKITTI_CLASSES = {
    0: "unlabeled",
    1: "car",
    2: "bicycle", 
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign"
}

# Enhanced canonical classes combining both datasets
CANONICAL_CLASSES = [
    "unlabeled", "road", "road_marking", "natural", "building", "utility_line",
    "pole", "car", "fence", "traffic_sign", "traffic_light", "vegetation", 
    "terrain", "other_ground", "other_object", "bicycle", "motorcycle", 
    "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", 
    "parking", "sidewalk", "trunk"
]

# Map Toronto3D to canonical classes
TORONTO3D_TO_CANONICAL = {
    0: "unlabeled",
    1: "road",
    2: "road_marking", 
    3: "natural",
    4: "building",
    5: "utility_line",
    6: "pole",
    7: "car",
    8: "fence",
    9: "traffic_sign",
    10: "traffic_light",
    11: "vegetation",
    12: "terrain",
    13: "other_ground",
    14: "other_object"
}

# Map KITTI to canonical classes
KITTI_TO_CANONICAL = {
    0: "unlabeled",
    1: "car",
    2: "bicycle",
    3: "motorcycle", 
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign"
}

# Enhanced color palette for all classes
ENHANCED_COLORS = {
    "unlabeled": [0, 0, 0],
    "road": [128, 64, 128],
    "road_marking": [255, 255, 255],
    "natural": [70, 70, 70],
    "building": [70, 130, 180],
    "utility_line": [190, 153, 153],
    "pole": [153, 153, 153],
    "car": [0, 0, 142],
    "fence": [190, 153, 153],
    "traffic_sign": [220, 220, 0],
    "traffic_light": [250, 170, 30],
    "vegetation": [107, 142, 35],
    "terrain": [152, 251, 152],
    "other_ground": [70, 130, 180],
    "other_object": [102, 102, 156],
    "bicycle": [119, 11, 32],
    "motorcycle": [0, 0, 230],
    "truck": [0, 0, 70],
    "other-vehicle": [0, 60, 100],
    "person": [220, 20, 60],
    "bicyclist": [255, 0, 0],
    "motorcyclist": [0, 0, 142],
    "parking": [250, 170, 160],
    "sidewalk": [244, 35, 232],
    "trunk": [102, 102, 156]
}

# Model configurations with real pretrained models
MODEL_CONFIGS = {
    "pointnet2_toronto3d": {
        "url": "https://huggingface.co/datasets/PointCloudLibrary/pointnet2/resolve/main/pointnet2_toronto3d.pth",
        "sha256": "a1b2c3d4e5f6...",  # Placeholder - would need actual hash
        "input_size": [1024, 3],
        "num_classes": 15,
        "model_type": "pointnet2",
        "dataset": "toronto3d",
        "description": "PointNet++ trained on Toronto3D dataset"
    },
    "randla_net_toronto3d": {
        "url": "https://huggingface.co/datasets/PointCloudLibrary/randla_net/resolve/main/randla_net_toronto3d.pth",
        "sha256": "f6e5d4c3b2a1...",  # Placeholder
        "input_size": [8192, 3],
        "num_classes": 15,
        "model_type": "randla_net",
        "dataset": "toronto3d",
        "description": "RandLA-Net trained on Toronto3D dataset"
    },
    "pointnet2_semantickitti": {
        "url": "https://huggingface.co/datasets/PointCloudLibrary/pointnet2/resolve/main/pointnet2_semantickitti.pth",
        "sha256": "c3d4e5f6a1b2...",  # Placeholder
        "input_size": [1024, 3],
        "num_classes": 20,
        "model_type": "pointnet2",
        "dataset": "semantickitti",
        "description": "PointNet++ trained on SemanticKITTI dataset"
    },
    "minkunet_semantickitti": {
        "url": "https://github.com/isl-org/Open3D-ML/releases/download/v0.18.0/minkunet_semantickitti.pth",
        "sha256": "a1b2c3d4e5f6...",  # Placeholder
        "input_size": [48000, 3],
        "num_classes": 20,
        "model_type": "minkunet",
        "dataset": "semantickitti",
        "description": "MinkowskiNet trained on SemanticKITTI dataset"
    },
    "kpconv_semantickitti": {
        "url": "https://github.com/isl-org/Open3D-ML/releases/download/v0.18.0/kpconv_semantickitti.pth", 
        "sha256": "f6e5d4c3b2a1...",  # Placeholder
        "input_size": [48000, 3],
        "num_classes": 20,
        "model_type": "kpconv",
        "dataset": "semantickitti",
        "description": "KPConv trained on SemanticKITTI dataset"
    }
}

# Default model recommendations
DEFAULT_MODELS = {
    "toronto3d": "pointnet2_toronto3d",
    "semantickitti": "pointnet2_semantickitti",
    "urban": "randla_net_toronto3d",
    "general": "pointnet2_toronto3d"
}


def get_model_cache_dir() -> Path:
    """Get the model cache directory."""
    cache_dir = Path.home() / ".cache" / "pointroad" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_model(model_name: str, force_download: bool = False) -> Optional[Path]:
    """Download a pretrained model."""
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}")
        return None
    
    config = MODEL_CONFIGS[model_name]
    cache_dir = get_model_cache_dir()
    model_path = cache_dir / f"{model_name}.pth"
    
    if model_path.exists() and not force_download:
        logger.info(f"Model {model_name} already exists at {model_path}")
        return model_path
    
    logger.info(f"Downloading model {model_name} from {config['url']}")
    try:
        # Create a progress bar for download
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                logger.info(f"Download progress: {percent}%")
        
        urllib.request.urlretrieve(config['url'], model_path, show_progress)
        logger.info(f"Downloaded model to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None


def get_available_models() -> List[str]:
    """Get list of available model names."""
    return list(MODEL_CONFIGS.keys())


def get_model_info(model_name: str) -> Optional[Dict]:
    """Get information about a specific model."""
    if model_name not in MODEL_CONFIGS:
        return None
    return MODEL_CONFIGS[model_name].copy()


def get_recommended_model(dataset_type: str = "general") -> str:
    """Get recommended model for given dataset type."""
    return DEFAULT_MODELS.get(dataset_type, DEFAULT_MODELS["general"])


def verify_model_hash(model_path: Path, expected_hash: str) -> bool:
    """Verify model file hash."""
    if not model_path.exists():
        return False
    
    with open(model_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    return file_hash == expected_hash


def get_semantic_colors() -> Dict[str, np.ndarray]:
    """Get color palette for semantic classes."""
    colors = {}
    for class_name, rgb in ENHANCED_COLORS.items():
        colors[class_name] = np.array(rgb, dtype=np.float64) / 255.0
    return colors


def map_toronto3d_to_canonical(toronto3d_labels: np.ndarray) -> np.ndarray:
    """Map Toronto3D labels to canonical class indices."""
    canonical_labels = np.zeros_like(toronto3d_labels)
    for toronto3d_id, canonical_name in TORONTO3D_TO_CANONICAL.items():
        mask = toronto3d_labels == toronto3d_id
        if canonical_name in CANONICAL_CLASSES:
            canonical_labels[mask] = CANONICAL_CLASSES.index(canonical_name)
    return canonical_labels


def map_kitti_to_canonical(kitti_labels: np.ndarray) -> np.ndarray:
    """Map KITTI labels to canonical class indices."""
    canonical_labels = np.zeros_like(kitti_labels)
    for kitti_id, canonical_name in KITTI_TO_CANONICAL.items():
        mask = kitti_labels == kitti_id
        if canonical_name in CANONICAL_CLASSES:
            canonical_labels[mask] = CANONICAL_CLASSES.index(canonical_name)
    return canonical_labels


def get_class_names() -> list:
    """Get list of canonical class names."""
    return CANONICAL_CLASSES.copy()


def get_discrete_classes() -> set:
    """Get set of classes that represent discrete objects."""
    return {"car", "bicycle", "motorcycle", "truck", "other-vehicle", 
            "person", "bicyclist", "motorcyclist", "building", "fence", 
            "vegetation", "trunk", "pole", "traffic-sign", "traffic_light",
            "utility_line", "other_object"}


def get_ground_classes() -> set:
    """Get set of ground-level classes."""
    return {"road", "road_marking", "parking", "sidewalk", "other_ground", "terrain"}


def get_vegetation_classes() -> set:
    """Get set of vegetation classes."""
    return {"vegetation", "trunk", "natural"}


def get_vehicle_classes() -> set:
    """Get set of vehicle classes."""
    return {"car", "bicycle", "motorcycle", "truck", "other-vehicle"}


def get_person_classes() -> set:
    """Get set of person classes."""
    return {"person", "bicyclist", "motorcyclist"}


def get_infrastructure_classes() -> set:
    """Get set of infrastructure classes."""
    return {"building", "fence", "pole", "traffic-sign", "traffic_light", 
            "utility_line", "road_marking"}


def get_traffic_classes() -> set:
    """Get set of traffic-related classes."""
    return {"traffic-sign", "traffic_light", "road_marking"}


__all__ = [
    "download_model",
    "verify_model_hash", 
    "get_semantic_colors",
    "map_toronto3d_to_canonical",
    "map_kitti_to_canonical",
    "get_class_names",
    "get_discrete_classes",
    "get_ground_classes",
    "get_vegetation_classes", 
    "get_vehicle_classes",
    "get_person_classes",
    "get_infrastructure_classes",
    "get_traffic_classes",
    "get_available_models",
    "get_model_info",
    "get_recommended_model",
    "TORONTO3D_CLASSES",
    "SEMANTICKITTI_CLASSES",
    "CANONICAL_CLASSES",
    "TORONTO3D_TO_CANONICAL",
    "KITTI_TO_CANONICAL",
    "ENHANCED_COLORS",
    "MODEL_CONFIGS",
    "DEFAULT_MODELS"
]


