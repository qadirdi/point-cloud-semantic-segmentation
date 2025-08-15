#!/usr/bin/env python3
"""Model loader for pretrained semantic segmentation models."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import open3d as o3d
from loguru import logger

# SemanticKITTI class definitions
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

# Map to our canonical classes
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

CANONICAL_CLASSES = [
    "unlabeled", "car", "bicycle", "motorcycle", "truck", "other-vehicle",
    "person", "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
    "other-ground", "building", "fence", "vegetation", "trunk", "terrain",
    "pole", "traffic-sign"
]

# Color palette for SemanticKITTI classes
SEMANTICKITTI_COLORS = {
    "unlabeled": [0, 0, 0],
    "car": [245, 150, 100],
    "bicycle": [245, 230, 100],
    "motorcycle": [150, 60, 30],
    "truck": [180, 30, 80],
    "other-vehicle": [255, 0, 0],
    "person": [30, 30, 255],
    "bicyclist": [200, 40, 255],
    "motorcyclist": [90, 30, 150],
    "road": [255, 0, 255],
    "parking": [255, 150, 255],
    "sidewalk": [75, 0, 75],
    "other-ground": [75, 0, 175],
    "building": [0, 200, 255],
    "fence": [50, 120, 255],
    "vegetation": [0, 175, 0],
    "trunk": [0, 60, 135],
    "terrain": [80, 240, 150],
    "pole": [150, 240, 255],
    "traffic-sign": [0, 0, 255]
}

# Model configurations
MODEL_CONFIGS = {
    "minkunet_semantickitti": {
        "url": "https://github.com/isl-org/Open3D-ML/releases/download/v0.18.0/minkunet_semantickitti.pth",
        "sha256": "a1b2c3d4e5f6...",  # Placeholder - would need actual hash
        "input_size": [48000, 3],
        "num_classes": 20,
        "model_type": "minkunet"
    },
    "kpconv_semantickitti": {
        "url": "https://github.com/isl-org/Open3D-ML/releases/download/v0.18.0/kpconv_semantickitti.pth", 
        "sha256": "f6e5d4c3b2a1...",  # Placeholder
        "input_size": [48000, 3],
        "num_classes": 20,
        "model_type": "kpconv"
    }
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
        urllib.request.urlretrieve(config['url'], model_path)
        logger.info(f"Downloaded model to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None


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
    for class_name, rgb in SEMANTICKITTI_COLORS.items():
        colors[class_name] = np.array(rgb, dtype=np.float64) / 255.0
    return colors


def map_kitti_to_canonical(kitti_labels: np.ndarray) -> np.ndarray:
    """Map KITTI labels to canonical class indices."""
    canonical_labels = np.zeros_like(kitti_labels)
    for kitti_id, canonical_name in KITTI_TO_CANONICAL.items():
        mask = kitti_labels == kitti_id
        canonical_labels[mask] = CANONICAL_CLASSES.index(canonical_name)
    return canonical_labels


def get_class_names() -> list:
    """Get list of canonical class names."""
    return CANONICAL_CLASSES.copy()


def get_discrete_classes() -> set:
    """Get set of classes that represent discrete objects."""
    return {"car", "bicycle", "motorcycle", "truck", "other-vehicle", 
            "person", "bicyclist", "motorcyclist", "building", "fence", 
            "vegetation", "trunk", "pole", "traffic-sign"}


def get_ground_classes() -> set:
    """Get set of ground-level classes."""
    return {"road", "parking", "sidewalk", "other-ground", "terrain"}


def get_vegetation_classes() -> set:
    """Get set of vegetation classes."""
    return {"vegetation", "trunk"}


def get_vehicle_classes() -> set:
    """Get set of vehicle classes."""
    return {"car", "bicycle", "motorcycle", "truck", "other-vehicle"}


def get_person_classes() -> set:
    """Get set of person classes."""
    return {"person", "bicyclist", "motorcyclist"}


def get_infrastructure_classes() -> set:
    """Get set of infrastructure classes."""
    return {"building", "fence", "pole", "traffic-sign"}


__all__ = [
    "download_model",
    "verify_model_hash", 
    "get_semantic_colors",
    "map_kitti_to_canonical",
    "get_class_names",
    "get_discrete_classes",
    "get_ground_classes",
    "get_vegetation_classes", 
    "get_vehicle_classes",
    "get_person_classes",
    "get_infrastructure_classes",
    "SEMANTICKITTI_CLASSES",
    "CANONICAL_CLASSES",
    "KITTI_TO_CANONICAL"
]


