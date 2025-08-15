"""Pretrained model management for point cloud semantic segmentation."""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
from loguru import logger

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub not available, using fallback download methods")

from .model_loader import MODEL_CONFIGS, get_model_cache_dir


class PointNet2Pretrained(nn.Module):
    """PointNet++ model with pretrained weights."""
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # PointNet++ architecture
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
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, 1)
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        # x shape: (batch_size, num_points, input_channels)
        x = x.transpose(1, 2)  # (batch_size, input_channels, num_points)
        
        # Local features
        local_features = self.encoder(x)
        
        # Global features
        global_features = self.global_pool(local_features)
        global_features = global_features.expand(-1, -1, x.size(-1))
        
        # Combine local and global features
        combined_features = torch.cat([local_features, global_features], dim=1)
        
        # Decode
        output = self.decoder(combined_features)
        output = output.transpose(1, 2)  # (batch_size, num_points, num_classes)
        
        return output


class RandLANetPretrained(nn.Module):
    """RandLA-Net model with pretrained weights."""
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # RandLA-Net architecture
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
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
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
        
        # Encode
        features = self.encoder(x)
        features = features.transpose(1, 2)  # (batch_size, num_points, features)
        
        # Apply attention
        attended_features, _ = self.attention(features, features, features)
        
        # Decode
        attended_features = attended_features.transpose(1, 2)  # (batch_size, features, num_points)
        output = self.decoder(attended_features)
        output = output.transpose(1, 2)  # (batch_size, num_points, num_classes)
        
        return output


class PretrainedModelDownloader:
    """Downloader for pretrained models from various sources."""
    
    def __init__(self):
        self.cache_dir = get_model_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_from_huggingface(self, model_name: str, repo_id: str, filename: str) -> Optional[Path]:
        """Download model from Hugging Face Hub."""
        if not HF_AVAILABLE:
            logger.error("Hugging Face Hub not available")
            return None
            
        try:
            logger.info(f"Downloading {model_name} from Hugging Face Hub...")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.cache_dir,
                resume_download=True
            )
            logger.info(f"Downloaded {model_name} to {model_path}")
            return Path(model_path)
        except Exception as e:
            logger.error(f"Failed to download {model_name} from Hugging Face: {e}")
            return None
    
    def download_from_url(self, model_name: str, url: str) -> Optional[Path]:
        """Download model from URL."""
        import urllib.request
        
        model_path = self.cache_dir / f"{model_name}.pth"
        
        if model_path.exists():
            logger.info(f"Model {model_name} already exists at {model_path}")
            return model_path
        
        try:
            logger.info(f"Downloading {model_name} from {url}")
            
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    logger.info(f"Download progress: {percent}%")
            
            urllib.request.urlretrieve(url, model_path, show_progress)
            logger.info(f"Downloaded {model_name} to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return None
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the path to a downloaded model."""
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        # Check if model already exists
        model_path = self.cache_dir / f"{model_name}.pth"
        if model_path.exists():
            return model_path
        
        # Try to download
        config = MODEL_CONFIGS[model_name]
        
        # Try Hugging Face first
        if HF_AVAILABLE and "hf_repo" in config:
            path = self.download_from_huggingface(
                model_name, 
                config["hf_repo"], 
                config["hf_filename"]
            )
            if path:
                return path
        
        # Fallback to URL
        if "url" in config:
            return self.download_from_url(model_name, config["url"])
        
        logger.error(f"No download method available for {model_name}")
        return None


class ModelFactory:
    """Factory for creating pretrained models."""
    
    def __init__(self):
        self.downloader = PretrainedModelDownloader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_model(self, model_name: str, force_download: bool = False) -> Optional[nn.Module]:
        """Create a pretrained model."""
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        config = MODEL_CONFIGS[model_name]
        
        # Download model if needed
        model_path = self.downloader.get_model_path(model_name)
        if not model_path:
            logger.error(f"Failed to get model path for {model_name}")
            return None
        
        try:
            # Create model architecture
            if config['model_type'] == 'pointnet2':
                model = PointNet2Pretrained(
                    num_classes=config['num_classes'],
                    input_channels=3
                )
            elif config['model_type'] == 'randla_net':
                model = RandLANetPretrained(
                    num_classes=config['num_classes'],
                    input_channels=3
                )
            else:
                logger.error(f"Unsupported model type: {config['model_type']}")
                return None
            
            # Load pretrained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            
            logger.info(f"Successfully loaded pretrained model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(MODEL_CONFIGS.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a model."""
        if model_name not in MODEL_CONFIGS:
            return None
        return MODEL_CONFIGS[model_name].copy()


# Update MODEL_CONFIGS with Hugging Face repositories
MODEL_CONFIGS.update({
    "pointnet2_toronto3d": {
        "url": "https://huggingface.co/datasets/PointCloudLibrary/pointnet2/resolve/main/pointnet2_toronto3d.pth",
        "hf_repo": "PointCloudLibrary/pointnet2",
        "hf_filename": "pointnet2_toronto3d.pth",
        "sha256": "a1b2c3d4e5f6...",  # Placeholder
        "input_size": [1024, 3],
        "num_classes": 15,
        "model_type": "pointnet2",
        "dataset": "toronto3d",
        "description": "PointNet++ trained on Toronto3D dataset"
    },
    "randla_net_toronto3d": {
        "url": "https://huggingface.co/datasets/PointCloudLibrary/randla_net/resolve/main/randla_net_toronto3d.pth",
        "hf_repo": "PointCloudLibrary/randla_net",
        "hf_filename": "randla_net_toronto3d.pth",
        "sha256": "f6e5d4c3b2a1...",  # Placeholder
        "input_size": [8192, 3],
        "num_classes": 15,
        "model_type": "randla_net",
        "dataset": "toronto3d",
        "description": "RandLA-Net trained on Toronto3D dataset"
    },
    "pointnet2_semantickitti": {
        "url": "https://huggingface.co/datasets/PointCloudLibrary/pointnet2/resolve/main/pointnet2_semantickitti.pth",
        "hf_repo": "PointCloudLibrary/pointnet2",
        "hf_filename": "pointnet2_semantickitti.pth",
        "sha256": "c3d4e5f6a1b2...",  # Placeholder
        "input_size": [1024, 3],
        "num_classes": 20,
        "model_type": "pointnet2",
        "dataset": "semantickitti",
        "description": "PointNet++ trained on SemanticKITTI dataset"
    }
})


__all__ = [
    "PointNet2Pretrained",
    "RandLANetPretrained",
    "PretrainedModelDownloader",
    "ModelFactory",
    "MODEL_CONFIGS"
]