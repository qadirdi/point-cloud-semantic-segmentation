"""Enhanced inference module with pretrained deep learning models for point cloud semantic segmentation."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from loguru import logger

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using simplified features")

from .model_loader import (
    get_semantic_colors, get_class_names, get_available_models, 
    get_model_info, get_recommended_model, download_model,
    map_toronto3d_to_canonical, map_kitti_to_canonical,
    TORONTO3D_CLASSES, SEMANTICKITTI_CLASSES, CANONICAL_CLASSES
)


@dataclass
class EnhancedInferenceResult:
    """Enhanced inference result with detailed metadata."""
    labels: np.ndarray
    scores: np.ndarray
    class_names: List[str]
    colors: np.ndarray
    model_name: str
    model_confidence: float
    processing_time: float
    num_points_processed: int
    class_distribution: Dict[str, int]


class PointNet2Model(nn.Module):
    """PointNet++ model for point cloud segmentation."""
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Simplified PointNet++ architecture
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_points, input_channels)
        x = x.transpose(1, 2)  # (batch_size, input_channels, num_points)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.transpose(1, 2)  # (batch_size, num_points, num_classes)
        return x


class RandLANetModel(nn.Module):
    """RandLA-Net model for large-scale point cloud segmentation."""
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Simplified RandLA-Net architecture
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, num_classes, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_points, input_channels)
        x = x.transpose(1, 2)  # (batch_size, input_channels, num_points)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.transpose(1, 2)  # (batch_size, num_points, num_classes)
        return x


class PretrainedModelManager:
    """Manager for loading and running pretrained models."""
    
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def get_available_models(self) -> List[str]:
        """Expose available model keys from the loader without triggering downloads."""
        try:
            return list(get_available_models())
        except Exception:
            return []
    
    def load_model(self, model_name: str, force_download: bool = False) -> bool:
        """Load a pretrained model."""
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        model_info = get_model_info(model_name)
        if not model_info:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        # Download model if needed (with local/offline first strategy implemented in loader)
        model_path = download_model(model_name, force_download)
        if not model_path:
            logger.error(f"Failed to download model: {model_name}")
            return False
        
        try:
            # Create model architecture
            if model_info['model_type'] == 'pointnet2':
                model = PointNet2Model(
                    num_classes=model_info['num_classes'],
                    input_channels=3
                )
            elif model_info['model_type'] == 'randla_net':
                model = RandLANetModel(
                    num_classes=model_info['num_classes'],
                    input_channels=3
                )
            else:
                logger.error(f"Unsupported model type: {model_info['model_type']}")
                return False
            
            # Load pretrained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            self.models[model_name] = {
                'model': model,
                'info': model_info,
                'is_dummy_fallback': False
            }
            
            logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
            
            # If it's a dummy file or corrupted model, create a minimal working model
            if "PytorchStreamReader" in str(e) or "zip archive" in str(e):
                logger.info(f"Creating minimal fallback model for {model_name}")
                try:
                    # Initialize model with random weights (better than complete failure)
                    model.to(self.device)
                    model.eval()
                    
                    self.models[model_name] = {
                        'model': model,
                        'info': model_info,
                        'is_dummy_fallback': True
                    }
                    
                    logger.warning(f"Using minimal model for {model_name} - predictions will be random but system will work")
                    return True
                except Exception as e2:
                    logger.error(f"Failed to create fallback model: {e2}")
                    return False
            
            return False
    
    def preprocess_points(self, points: np.ndarray, model_name: str) -> torch.Tensor:
        """Preprocess points for model input."""
        model_info = self.models[model_name]['info']
        
        # Normalize points
        points_centered = points - points.mean(axis=0)
        points_normalized = points_centered / (points_centered.std(axis=0) + 1e-8)
        
        # Convert to tensor
        points_tensor = torch.from_numpy(points_normalized.astype(np.float32))
        points_tensor = points_tensor.unsqueeze(0)  # Add batch dimension
        
        return points_tensor.to(self.device)
    
    def run_inference(self, points: np.ndarray, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on points."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return None, None
        
        model = self.models[model_name]['model']
        model_info = self.models[model_name]['info']
        
        # Preprocess points
        points_tensor = self.preprocess_points(points, model_name)
        
        # Run inference
        with torch.no_grad():
            logits = model(points_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            confidence_scores = torch.max(probabilities, dim=-1)[0]
        
        # Convert to numpy
        labels = predictions.cpu().numpy()[0]
        scores = confidence_scores.cpu().numpy()[0]
        
        return labels, scores


def analyze_point_cloud_characteristics(points: np.ndarray) -> Dict[str, float]:
    """Analyze point cloud characteristics to recommend the best model."""
    characteristics = {}
    
    # Point density analysis
    if len(points) > 1000:
        # Calculate local density using nearest neighbors
        if SKLEARN_AVAILABLE:
            nbrs = NearestNeighbors(n_neighbors=min(10, len(points))).fit(points)
            distances, _ = nbrs.kneighbors(points)
            avg_distance = np.mean(distances[:, 1:])
            characteristics['point_density'] = 1.0 / (avg_distance + 1e-6)
        else:
            characteristics['point_density'] = len(points) / (np.prod(points.max(axis=0) - points.min(axis=0)) + 1e-6)
    else:
        characteristics['point_density'] = 0.0
    
    # Spatial extent
    bbox = points.max(axis=0) - points.min(axis=0)
    characteristics['spatial_extent'] = np.linalg.norm(bbox)
    
    # Height distribution (for urban vs rural classification)
    z_coords = points[:, 2]
    characteristics['height_variance'] = np.var(z_coords)
    characteristics['max_height'] = np.max(z_coords)
    
    # Point count
    characteristics['num_points'] = len(points)
    
    return characteristics


def recommend_model(points: np.ndarray) -> str:
    """Recommend the best model based on point cloud characteristics."""
    characteristics = analyze_point_cloud_characteristics(points)
    
    # Simple heuristics for model selection
    if characteristics['num_points'] > 100000:
        # Large point cloud - use RandLA-Net
        return get_recommended_model("urban")
    elif characteristics['height_variance'] > 10.0:
        # High variance in height - likely urban scene
        return get_recommended_model("toronto3d")
    else:
        # General case - use PointNet++
        return get_recommended_model("general")


def run_enhanced_segmentation(
    pcd: o3d.geometry.PointCloud,
    model_name: Optional[str] = None,
    force_download: bool = False
) -> EnhancedInferenceResult:
    """Run enhanced semantic segmentation using pretrained models."""
    import time
    
    start_time = time.time()
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        logger.error("Empty point cloud")
        return None
    
    # Recommend model if not specified
    if model_name is None:
        model_name = recommend_model(points)
        logger.info(f"Recommended model: {model_name}")
    
    # Initialize model manager
    model_manager = PretrainedModelManager()
    
    # Load model
    if not model_manager.load_model(model_name, force_download):
        logger.error(f"Failed to load model: {model_name}")
        return None
    
    # Check if this is a dummy/fallback model and use enhanced dummy segmentation instead
    if (model_name in model_manager.models and 
        model_manager.models[model_name].get('is_dummy_fallback', False)):
        logger.info("Using enhanced dummy segmentation instead of fallback model")
        from .infer import run_segmentation_dummy
        dummy_result = run_segmentation_dummy(pcd)
        
        # Convert to EnhancedInferenceResult format
        class_distribution = {}
        for i, class_name in enumerate(dummy_result.class_names):
            count = np.sum(dummy_result.labels == i)
            if count > 0:
                class_distribution[class_name] = int(count)
        
        processing_time = time.time() - start_time
        
        return EnhancedInferenceResult(
            labels=dummy_result.labels,
            scores=dummy_result.scores,
            class_names=dummy_result.class_names,
            colors=dummy_result.colors,
            model_name=f"{model_name}_enhanced_dummy",
            model_confidence=float(np.mean(dummy_result.scores)),
            processing_time=processing_time,
            num_points_processed=len(points),
            class_distribution=class_distribution
        )
    
    # Run inference with actual model
    labels, scores = model_manager.run_inference(points, model_name)
    
    if labels is None or scores is None:
        logger.error("Inference failed, falling back to enhanced dummy segmentation")
        from .infer import run_segmentation_dummy
        dummy_result = run_segmentation_dummy(pcd)
        
        # Convert to EnhancedInferenceResult format
        class_distribution = {}
        for i, class_name in enumerate(dummy_result.class_names):
            count = np.sum(dummy_result.labels == i)
            if count > 0:
                class_distribution[class_name] = int(count)
        
        processing_time = time.time() - start_time
        
        return EnhancedInferenceResult(
            labels=dummy_result.labels,
            scores=dummy_result.scores,
            class_names=dummy_result.class_names,
            colors=dummy_result.colors,
            model_name=f"{model_name}_enhanced_dummy_fallback",
            model_confidence=float(np.mean(dummy_result.scores)),
            processing_time=processing_time,
            num_points_processed=len(points),
            class_distribution=class_distribution
        )
    
    # Map labels to canonical classes
    model_info = model_manager.models[model_name]['info']
    if model_info['dataset'] == 'toronto3d':
        canonical_labels = map_toronto3d_to_canonical(labels)
    elif model_info['dataset'] == 'semantickitti':
        canonical_labels = map_kitti_to_canonical(labels)
    else:
        canonical_labels = labels
    
    # Generate colors
    colors = np.zeros((len(points), 3), dtype=np.float64)
    semantic_colors = get_semantic_colors()
    class_names = get_class_names()
    
    for i, class_name in enumerate(class_names):
        mask = canonical_labels == i
        if np.any(mask) and class_name in semantic_colors:
            colors[mask] = semantic_colors[class_name]
    
    # Calculate class distribution
    class_distribution = {}
    for i, class_name in enumerate(class_names):
        count = np.sum(canonical_labels == i)
        if count > 0:
            class_distribution[class_name] = int(count)
    
    # Calculate overall confidence
    model_confidence = float(np.mean(scores))
    
    processing_time = time.time() - start_time
    
    return EnhancedInferenceResult(
        labels=canonical_labels,
        scores=scores,
        class_names=class_names,
        colors=colors,
        model_name=model_name,
        model_confidence=model_confidence,
        processing_time=processing_time,
        num_points_processed=len(points),
        class_distribution=class_distribution
    )


def run_ensemble_segmentation(
    pcd: o3d.geometry.PointCloud,
    model_names: Optional[List[str]] = None,
    force_download: bool = False
) -> EnhancedInferenceResult:
    """Run ensemble segmentation using multiple models."""
    if model_names is None:
        model_names = [
            get_recommended_model("toronto3d"),
            get_recommended_model("semantickitti")
        ]
    
    results = []
    for model_name in model_names:
        try:
            result = run_enhanced_segmentation(pcd, model_name, force_download)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.warning(f"Failed to run model {model_name}: {e}")
    
    if not results:
        logger.error("All models failed")
        return None
    
    # Simple ensemble: average the predictions
    if len(results) == 1:
        return results[0]
    
    # Average labels and scores from all models
    avg_labels = np.zeros_like(results[0].labels)
    avg_scores = np.zeros_like(results[0].scores)
    
    for result in results:
        avg_labels += result.labels
        avg_scores += result.scores
    
    avg_labels = (avg_labels / len(results)).astype(np.int32)
    avg_scores = avg_scores / len(results)
    
    # Use the first result as template and update with ensemble results
    ensemble_result = results[0]
    ensemble_result.labels = avg_labels
    ensemble_result.scores = avg_scores
    ensemble_result.model_name = f"ensemble_{'_'.join([r.model_name for r in results])}"
    ensemble_result.model_confidence = float(np.mean([r.model_confidence for r in results]))
    
    return ensemble_result


# Backward compatibility functions
def analyze_geometric_features(points: np.ndarray) -> Dict[str, np.ndarray]:
    """Analyze geometric features (kept for backward compatibility)."""
    features = {}
    
    # Height analysis
    z_coords = points[:, 2]
    features['height'] = z_coords
    features['relative_height'] = z_coords - np.min(z_coords)
    
    # Density analysis
    if len(points) > 10 and SKLEARN_AVAILABLE:
        nbrs = NearestNeighbors(n_neighbors=min(10, len(points)), algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        features['local_density'] = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-6)
    else:
        features['local_density'] = np.ones(len(points))
    
    return features


def enhanced_car_classification(points: np.ndarray, geometric_features: Dict[str, np.ndarray]) -> np.ndarray:
    """Enhanced car classification (kept for backward compatibility)."""
    # This is now handled by the pretrained models
    # Return dummy scores for backward compatibility
    return np.ones(len(points)) * 0.8


__all__ = [
    "EnhancedInferenceResult",
    "PretrainedModelManager",
    "PointNet2Model",
    "RandLANetModel",
    "run_enhanced_segmentation",
    "run_ensemble_segmentation",
    "recommend_model",
    "analyze_point_cloud_characteristics",
    "analyze_geometric_features",
    "enhanced_car_classification"
]
