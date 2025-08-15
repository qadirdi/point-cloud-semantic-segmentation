#!/usr/bin/env python3
"""Test script for pretrained point cloud semantic segmentation models."""

import numpy as np
import open3d as o3d
from pathlib import Path
from loguru import logger

# Import our modules
from pointroad.pointroad.ml.infer import run_segmentation, get_available_methods, get_recommended_method
from pointroad.pointroad.ml.enhanced_infer import run_enhanced_segmentation, run_ensemble_segmentation
from pointroad.pointroad.ml.model_loader import get_available_models, get_model_info, get_recommended_model


def create_test_point_cloud() -> o3d.geometry.PointCloud:
    """Create a test point cloud with various objects."""
    # Create a simple urban scene
    points = []
    
    # Ground plane
    x_ground = np.linspace(-10, 10, 100)
    y_ground = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_ground, y_ground)
    Z = np.zeros_like(X)
    ground_points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    points.append(ground_points)
    
    # Buildings (rectangular prisms)
    for i in range(3):
        x = np.random.uniform(-8, 8)
        y = np.random.uniform(-8, 8)
        width = np.random.uniform(2, 4)
        length = np.random.uniform(2, 4)
        height = np.random.uniform(3, 8)
        
        x_building = np.linspace(x, x + width, 20)
        y_building = np.linspace(y, y + length, 20)
        z_building = np.linspace(0, height, 15)
        
        X, Y, Z = np.meshgrid(x_building, y_building, z_building)
        building_points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        points.append(building_points)
    
    # Cars (small rectangular prisms on ground)
    for i in range(5):
        x = np.random.uniform(-9, 9)
        y = np.random.uniform(-9, 9)
        width = np.random.uniform(1.5, 2.0)
        length = np.random.uniform(3, 5)
        height = np.random.uniform(1.2, 1.8)
        
        x_car = np.linspace(x, x + length, 15)
        y_car = np.linspace(y, y + width, 10)
        z_car = np.linspace(0, height, 8)
        
        X, Y, Z = np.meshgrid(x_car, y_car, z_car)
        car_points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        points.append(car_points)
    
    # Vegetation (random clusters)
    for i in range(8):
        center_x = np.random.uniform(-9, 9)
        center_y = np.random.uniform(-9, 9)
        center_z = np.random.uniform(0.5, 2)
        
        num_points = np.random.randint(50, 200)
        radius = np.random.uniform(0.5, 2)
        
        angles = np.random.uniform(0, 2*np.pi, num_points)
        radii = np.random.uniform(0, radius, num_points)
        heights = np.random.uniform(0, 3, num_points)
        
        x_veg = center_x + radii * np.cos(angles)
        y_veg = center_y + radii * np.sin(angles)
        z_veg = center_z + heights
        
        veg_points = np.column_stack([x_veg, y_veg, z_veg])
        points.append(veg_points)
    
    # Combine all points
    all_points = np.vstack(points)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    
    logger.info(f"Created test point cloud with {len(all_points):,} points")
    return pcd


def test_available_methods():
    """Test and display available segmentation methods."""
    logger.info("=== Testing Available Methods ===")
    
    methods = get_available_methods()
    logger.info(f"Available methods: {methods}")
    
    recommended = get_recommended_method()
    logger.info(f"Recommended method: {recommended}")
    
    return methods


def test_available_models():
    """Test and display available pretrained models."""
    logger.info("=== Testing Available Models ===")
    
    models = get_available_models()
    logger.info(f"Available models: {models}")
    
    for model_name in models:
        info = get_model_info(model_name)
        if info:
            logger.info(f"Model: {model_name}")
            logger.info(f"  Type: {info.get('model_type', 'unknown')}")
            logger.info(f"  Dataset: {info.get('dataset', 'unknown')}")
            logger.info(f"  Classes: {info.get('num_classes', 'unknown')}")
            logger.info(f"  Description: {info.get('description', 'No description')}")
    
    return models


def test_segmentation_methods(pcd: o3d.geometry.PointCloud):
    """Test different segmentation methods."""
    logger.info("=== Testing Segmentation Methods ===")
    
    # Test dummy segmentation
    logger.info("Testing dummy segmentation...")
    try:
        result_dummy = run_segmentation(pcd, method="dummy")
        stats_dummy = get_class_statistics(result_dummy)
        logger.info(f"Dummy segmentation completed. Classes found: {list(stats_dummy.keys())}")
    except Exception as e:
        logger.error(f"Dummy segmentation failed: {e}")
    
    # Test pretrained models
    logger.info("Testing pretrained models...")
    try:
        result_pretrained = run_segmentation(pcd, method="pretrained")
        stats_pretrained = get_class_statistics(result_pretrained)
        logger.info(f"Pretrained segmentation completed. Classes found: {list(stats_pretrained.keys())}")
    except Exception as e:
        logger.error(f"Pretrained segmentation failed: {e}")
    
    # Test auto method
    logger.info("Testing auto method...")
    try:
        result_auto = run_segmentation(pcd, method="auto")
        stats_auto = get_class_statistics(result_auto)
        logger.info(f"Auto segmentation completed. Classes found: {list(stats_auto.keys())}")
    except Exception as e:
        logger.error(f"Auto segmentation failed: {e}")


def test_enhanced_segmentation(pcd: o3d.geometry.PointCloud):
    """Test enhanced segmentation with specific models."""
    logger.info("=== Testing Enhanced Segmentation ===")
    
    # Test with recommended model
    recommended_model = get_recommended_model("general")
    logger.info(f"Testing with recommended model: {recommended_model}")
    
    try:
        result = run_enhanced_segmentation(pcd, model_name=recommended_model)
        if result:
            logger.info(f"Enhanced segmentation completed:")
            logger.info(f"  Model: {result.model_name}")
            logger.info(f"  Confidence: {result.model_confidence:.3f}")
            logger.info(f"  Processing time: {result.processing_time:.2f}s")
            logger.info(f"  Points processed: {result.num_points_processed:,}")
            logger.info(f"  Classes found: {list(result.class_distribution.keys())}")
        else:
            logger.warning("Enhanced segmentation returned None")
    except Exception as e:
        logger.error(f"Enhanced segmentation failed: {e}")


def test_ensemble_segmentation(pcd: o3d.geometry.PointCloud):
    """Test ensemble segmentation with multiple models."""
    logger.info("=== Testing Ensemble Segmentation ===")
    
    try:
        result = run_ensemble_segmentation(pcd)
        if result:
            logger.info(f"Ensemble segmentation completed:")
            logger.info(f"  Model: {result.model_name}")
            logger.info(f"  Confidence: {result.model_confidence:.3f}")
            logger.info(f"  Processing time: {result.processing_time:.2f}s")
            logger.info(f"  Points processed: {result.num_points_processed:,}")
            logger.info(f"  Classes found: {list(result.class_distribution.keys())}")
        else:
            logger.warning("Ensemble segmentation returned None")
    except Exception as e:
        logger.error(f"Ensemble segmentation failed: {e}")


def save_results(pcd: o3d.geometry.PointCloud, result, filename: str):
    """Save segmentation results to file."""
    try:
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save colored point cloud
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = pcd.points
        colored_pcd.colors = o3d.utility.Vector3dVector(result.colors)
        
        output_path = output_dir / f"{filename}.ply"
        o3d.io.write_point_cloud(str(output_path), colored_pcd)
        logger.info(f"Saved colored point cloud to {output_path}")
        
        # Save statistics
        stats = get_class_statistics(result)
        stats_path = output_dir / f"{filename}_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def main():
    """Main test function."""
    logger.info("Starting pretrained model tests...")
    
    # Create test point cloud
    pcd = create_test_point_cloud()
    
    # Test available methods and models
    methods = test_available_methods()
    models = test_available_models()
    
    # Test segmentation methods
    test_segmentation_methods(pcd)
    
    # Test enhanced segmentation
    test_enhanced_segmentation(pcd)
    
    # Test ensemble segmentation
    test_ensemble_segmentation(pcd)
    
    # Save results from auto method
    logger.info("=== Saving Results ===")
    try:
        result = run_segmentation(pcd, method="auto")
        save_results(pcd, result, "segmentation_result")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    logger.info("Tests completed!")


if __name__ == "__main__":
    main()