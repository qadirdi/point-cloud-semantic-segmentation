#!/usr/bin/env python3
"""
Console-based Interactive Point Cloud Semantic Segmentation Application

This application provides a complete user-friendly interface through the console,
completely bypassing all graphics driver issues while providing the same functionality.
"""

import sys
import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import open3d as o3d
from loguru import logger

# Import from the existing pointroad package
from pointroad.pointroad.io.loader import load_point_cloud
from pointroad.pointroad.io.exporter import export_colored_point_cloud, export_instances_json, export_summary_csv
from pointroad.pointroad.ml.infer import run_segmentation, get_class_statistics, InferenceResult, get_available_methods, get_recommended_method
from pointroad.pointroad.ml.enhanced_infer import run_enhanced_segmentation, run_ensemble_segmentation
from pointroad.pointroad.post.cluster import cluster_all_points
from pointroad.pointroad.post.enhanced_cluster import enhanced_clustering_all_classes
from pointroad.pointroad.post.color import generate_distinct_colors
from pointroad.pointroad.ml.model_loader import get_semantic_colors, get_discrete_classes, get_available_models, get_model_info, get_recommended_model


class ConsoleSemanticSegmentationGUI:
    """Console-based interactive GUI for point cloud semantic segmentation."""
    
    def __init__(self):
        self.current_pcd = None
        self.segmentation_result = None
        self.cluster_labels = None
        self.instances_data = []
        self.class_visibility = {}
        self.instance_visibility = {}
        self.original_colors = None
        self.processing_stats = {}
        self.selected_file_path = None
        
    def print_header(self):
        """Print application header."""
        print("\n" + "=" * 80)
        print("🎯 POINT CLOUD SEMANTIC SEGMENTATION - INTERACTIVE CONSOLE")
        print("=" * 80)
        print("🚀 Advanced semantic segmentation and instance detection")
        print("📊 Complete processing statistics and detailed analysis")
        print("💾 Automatic export to multiple formats (PLY, JSON, CSV)")
        print("🎮 Interactive controls for result exploration")
        print("=" * 80)
        
    def show_main_menu(self):
        """Show the main application menu."""
        while True:
            self.print_header()
            
            if self.selected_file_path:
                print(f"📁 Current file: {self.selected_file_path.name}")
                if self.processing_stats:
                    print(f"📊 Status: Processed ({self.processing_stats.get('downsampled_points', 0):,} points)")
                else:
                    print("📊 Status: Selected, not processed")
            else:
                print("📁 Current file: None selected")
            
            print("\n🎮 MAIN MENU:")
            print("  1. 📁 Select point cloud file")
            print("  2. ⚙️  Process selected file")
            print("  3. 📊 View processing statistics")
            print("  4. 🏷️  View semantic classification results")
            print("  5. 📦 View detected instances")
            print("  6. 🎨 Toggle class/instance visibility")
            print("  7. 💾 Export results")
            print("  8. 🔄 Process another file")
            print("  9. 🤖 Show available models and methods")
            print("  0. 🚪 Exit")
            
            try:
                choice = input("\n👉 Select option (0-9): ").strip()
                
                if choice == "1":
                    self.select_file()
                elif choice == "2":
                    self.process_file()
                elif choice == "3":
                    self.show_processing_stats()
                elif choice == "4":
                    self.show_classification_results()
                elif choice == "5":
                    self.show_instances()
                elif choice == "6":
                    self.toggle_visibility()
                elif choice == "7":
                    self.export_results()
                elif choice == "8":
                    self.select_file()
                elif choice == "9":
                    self.show_available_models()
                elif choice == "0":
                    print("\n👋 Thank you for using Point Cloud Semantic Segmentation!")
                    break
                else:
                    print("❌ Invalid option. Please select 0-9.")
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n👋 Application closed by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("\nPress Enter to continue...")
    
    def select_file(self):
        """Interactive file selection."""
        print("\n" + "─" * 60)
        print("📁 FILE SELECTION")
        print("─" * 60)
        
        # Show available files in current directory
        current_dir = Path(".")
        ply_files = list(current_dir.glob("*.ply"))
        pcd_files = list(current_dir.glob("*.pcd"))
        las_files = list(current_dir.glob("*.las"))
        
        all_files = ply_files + pcd_files + las_files
        
        if all_files:
            print("📋 Available point cloud files:")
            for i, file_path in enumerate(all_files, 1):
                size_mb = file_path.stat().st_size / (1024**2)
                print(f"  {i:2d}. {file_path.name:<30} ({size_mb:8.1f} MB)")
        else:
            print("⚠️  No point cloud files found in current directory")
        
        print(f"\n🎯 Options:")
        print(f"  • Enter number (1-{len(all_files)}) to select a file")
        print(f"  • Enter custom file path")
        print(f"  • Press Enter to cancel")
        
        try:
            choice = input(f"\n👉 Your choice: ").strip()
            
            if not choice:
                print("❌ Selection cancelled")
                return
                
            if choice.isdigit() and 1 <= int(choice) <= len(all_files):
                selected_file = all_files[int(choice) - 1]
            else:
                selected_file = Path(choice)
                if not selected_file.exists():
                    print(f"❌ File not found: {selected_file}")
                    input("\nPress Enter to continue...")
                    return
            
            self.selected_file_path = selected_file
            file_size = selected_file.stat().st_size / (1024**3)
            
            print(f"\n✅ Selected: {selected_file.name}")
            print(f"📊 Size: {file_size:.2f} GB")
            print(f"📄 Format: {selected_file.suffix.upper()}")
            
            # Ask if user wants to process immediately
            process_now = input("\n🚀 Process this file now? (y/N): ").strip().lower()
            if process_now in ['y', 'yes']:
                self.process_file()
            
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid selection or cancelled")
        
        input("\nPress Enter to continue...")
    
    def process_file(self):
        """Process the selected point cloud file."""
        if not self.selected_file_path:
            print("❌ No file selected. Please select a file first.")
            input("\nPress Enter to continue...")
            return
        
        print("\n" + "─" * 60)
        print("⚙️  PROCESSING POINT CLOUD")
        print("─" * 60)
        print(f"📁 File: {self.selected_file_path.name}")
        
        try:
            start_time = time.time()
            
            # Load point cloud
            print("\n🔄 Step 1/5: Loading point cloud...")
            pcd = load_point_cloud(self.selected_file_path)
            points_np = np.asarray(pcd.points)
            print(f"✅ Loaded {len(points_np):,} points")
            load_time = time.time() - start_time
            
            # Store initial stats
            self.processing_stats = {
                "original_points": len(points_np),
                "file_size_gb": self.selected_file_path.stat().st_size / (1024**3),
                "load_time": load_time
            }
            
            # Downsample for processing
            print("\n🔄 Step 2/5: Downsampling point cloud...")
            downsample_start = time.time()
            pcd_downsampled = self.downsample_for_processing(pcd, target_points=100000)
            points_downsampled = np.asarray(pcd_downsampled.points)
            downsample_time = time.time() - downsample_start
            
            self.processing_stats.update({
                "downsampled_points": len(points_downsampled),
                "downsample_time": downsample_time,
                "downsample_ratio": len(points_downsampled) / len(points_np)
            })
            
            print(f"✅ Downsampled to {len(points_downsampled):,} points ({self.processing_stats['downsample_ratio']:.1%})")
            
            # Run enhanced semantic segmentation with pretrained models
            print("\n🔄 Step 3/5: Running enhanced semantic segmentation with pretrained models...")
            segment_start = time.time()
            
            # Try to use pretrained models, fallback to dummy if needed
            try:
                self.segmentation_result = run_segmentation(pcd_downsampled, method="auto")
                if hasattr(self.segmentation_result, 'model_name'):
                    print(f"✅ Used model: {self.segmentation_result.model_name}")
                else:
                    print("✅ Used fallback segmentation method")
            except Exception as e:
                logger.warning(f"Pretrained models failed, using dummy: {e}")
                self.segmentation_result = run_segmentation(pcd_downsampled, method="dummy")
            
            segment_time = time.time() - segment_start
            self.processing_stats["segmentation_time"] = segment_time
            
            class_stats = get_class_statistics(self.segmentation_result)
            car_points = np.sum(self.segmentation_result.labels == self.segmentation_result.class_names.index("car"))
            print(f"✅ Identified {len(class_stats)} semantic classes ({car_points:,} car points)")
            
            # Run enhanced clustering
            print("\n🔄 Step 4/5: Running enhanced instance clustering...")
            cluster_start = time.time()
            
            # Create dummy car confidence scores for backward compatibility
            car_confidence_scores = np.ones(len(self.segmentation_result.labels)) * 0.8
            
            self.cluster_labels, self.enhanced_instances = enhanced_clustering_all_classes(
                pcd_downsampled, 
                self.segmentation_result.labels,
                self.segmentation_result.class_names,
                car_confidence_scores
            )
            cluster_time = time.time() - cluster_start
            self.processing_stats["clustering_time"] = cluster_time
            
            car_instances = [inst for inst in self.enhanced_instances if inst.class_name == "car"]
            print(f"✅ Found {len(self.enhanced_instances)} total instances ({len(car_instances)} cars)")
            
            # Generate colors and create instances
            print("\n🔄 Step 5/5: Generating visualization and instance data...")
            
            # Use colors from enhanced segmentation
            colors = self.segmentation_result.colors.copy()
            
            # Enhance car visualization with instance colors
            if len(self.enhanced_instances) > 0:
                instance_colors = generate_distinct_colors(len(self.enhanced_instances))
                for i, instance in enumerate(self.enhanced_instances):
                    # Get instance points
                    instance_mask = self.cluster_labels == instance.instance_id
                    if np.any(instance_mask):
                        if instance.class_name == "car":
                            # Highlight cars with brighter, more distinct colors
                            car_color = instance_colors[i] * 1.2  # Brighten
                            car_color = np.clip(car_color, 0, 1)
                            colors[instance_mask] = 0.3 * colors[instance_mask] + 0.7 * car_color
                        else:
                            # Blend other instances more subtly
                            colors[instance_mask] = 0.6 * colors[instance_mask] + 0.4 * instance_colors[i]
            
            # Set colors and store data
            pcd_downsampled.colors = o3d.utility.Vector3dVector(colors)
            self.current_pcd = pcd_downsampled
            self.original_colors = colors.copy()
            
            # Convert enhanced instances to standard format
            self.instances_data = []
            for instance in self.enhanced_instances:
                inst_data = {
                    "instance_id": instance.instance_id,
                    "class": instance.class_name,
                    "num_points": instance.num_points,
                    "centroid": instance.centroid.tolist(),
                    "aabb_min": instance.aabb_min.tolist(),
                    "aabb_max": instance.aabb_max.tolist(),
                    "extent": instance.extent.tolist(),
                    "mean_score": instance.confidence_score,
                    "point_indices": [],  # Will be filled if needed
                    # Enhanced car-specific data
                    "car_confidence": getattr(instance, 'car_confidence', 0.0),
                    "is_likely_car": getattr(instance, 'is_likely_car', False),
                    "car_dimensions": getattr(instance, 'car_dimensions', {}),
                    "geometric_features": getattr(instance, 'geometric_features', {})
                }
                self.instances_data.append(inst_data)
            
            # Initialize visibility states
            for class_name in class_stats.keys():
                self.class_visibility[class_name] = True
            for instance in self.instances_data:
                self.instance_visibility[instance["instance_id"]] = True
            
            # Update final stats
            total_time = time.time() - start_time
            car_instances = [inst for inst in self.instances_data if inst["class"] == "car"]
            self.processing_stats.update({
                "total_clusters": len(self.enhanced_instances),
                "total_instances": len(self.instances_data),
                "car_instances": len(car_instances),
                "total_time": total_time,
                "class_stats": class_stats
            })
            
            print(f"✅ Processing complete! ({total_time:.1f}s total)")
            
            # Auto-export results
            print("\n💾 Auto-exporting results...")
            self.auto_export_results()
            
            # Show summary
            self.show_processing_summary()
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            logger.error(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def downsample_for_processing(self, pcd, target_points: int = 100000):
        """Downsample point cloud with progress feedback."""
        import open3d as o3d
        
        current_points = len(pcd.points)
        if current_points <= target_points:
            return pcd
        
        print(f"   🎯 Target: {target_points:,} points")
        
        # Iterative refinement
        points_np = np.asarray(pcd.points)
        bounds = points_np.max(axis=0) - points_np.min(axis=0)
        volume = bounds[0] * bounds[1] * bounds[2]
        target_density = target_points / volume
        voxel_size = (1.0 / target_density) ** (1.0 / 3.0)
        
        max_iterations = 5
        tolerance = 0.1
        
        for iteration in range(max_iterations):
            print(f"   🔄 Iteration {iteration + 1}: voxel_size = {voxel_size:.4f}")
            
            downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
            actual_points = len(downsampled.points)
            
            print(f"      Result: {actual_points:,} points")
            
            if abs(actual_points - target_points) / target_points < tolerance:
                print(f"   ✅ Target achieved: {actual_points:,} points")
                return downsampled
            
            if actual_points > 0:
                ratio = actual_points / target_points
                voxel_size = voxel_size * (ratio ** (1.0 / 3.0))
            else:
                voxel_size = voxel_size * 0.5
        
        # Final attempt
        final_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        final_points = len(final_downsampled.points)
        print(f"   📊 Final result: {final_points:,} points")
        
        return final_downsampled
    
    def create_instance_data(self, points_downsampled, valid_clusters):
        """Create detailed instance data."""
        import open3d as o3d
        
        self.instances_data = []
        for i, cluster_id in enumerate(valid_clusters):
            idxs = np.where(self.cluster_labels == cluster_id)[0]
            if idxs.size == 0:
                continue
                
            cluster_points = points_downsampled[idxs]
            centroid = cluster_points.mean(axis=0)
            
            # Determine dominant class
            cluster_labels_semantic = self.segmentation_result.labels[idxs]
            unique_semantic, semantic_counts = np.unique(cluster_labels_semantic, return_counts=True)
            dominant_class_idx = unique_semantic[semantic_counts.argmax()]
            dominant_class = self.segmentation_result.class_names[dominant_class_idx]
            
            # Calculate bounding box
            sub = o3d.geometry.PointCloud()
            sub.points = o3d.utility.Vector3dVector(cluster_points)
            aabb = sub.get_axis_aligned_bounding_box()
            extent = aabb.get_extent()
            
            # Calculate mean score
            mean_score = float(self.segmentation_result.scores[idxs].mean())
            
            inst_data = {
                "instance_id": int(cluster_id),
                "class": dominant_class,
                "num_points": int(idxs.size),
                "centroid": centroid.tolist(),
                "aabb_min": aabb.get_min_bound().tolist(),
                "aabb_max": aabb.get_max_bound().tolist(),
                "extent": extent.tolist(),
                "mean_score": mean_score,
                "point_indices": idxs.tolist()
            }
            self.instances_data.append(inst_data)
    
    def show_processing_summary(self):
        """Show a quick processing summary."""
        if not self.processing_stats:
            return
            
        stats = self.processing_stats
        
        print("\n" + "─" * 60)
        print("📊 PROCESSING SUMMARY")
        print("─" * 60)
        print(f"⏱️  Total Time: {stats.get('total_time', 0):.2f}s")
        print(f"📊 Original Points: {stats.get('original_points', 0):,}")
        print(f"📊 Processed Points: {stats.get('downsampled_points', 0):,}")
        print(f"🎯 Clusters Found: {stats.get('total_clusters', 0)}")
        print(f"📦 Instances Detected: {stats.get('total_instances', 0)}")
        if stats.get('car_instances', 0) > 0:
            print(f"🚗 Cars Detected: {stats.get('car_instances', 0)}")
        
        class_stats = stats.get('class_stats', {})
        if class_stats:
            print(f"\n🏷️  Top 3 Classes:")
            sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
            for class_name, class_data in sorted_classes:
                extra_info = ""
                if class_name == "car" and 'high_confidence_points' in class_data:
                    extra_info = f" (🎯 {class_data['high_confidence_points']:,} high-confidence)"
                print(f"   {class_name}: {class_data['count']:,} points ({class_data['percentage']:.1f}%){extra_info}")
    
    def show_processing_stats(self):
        """Show detailed processing statistics."""
        if not self.processing_stats:
            print("❌ No processing completed yet. Please process a file first.")
            input("\nPress Enter to continue...")
            return
        
        stats = self.processing_stats
        
        print("\n" + "─" * 60)
        print("📊 DETAILED PROCESSING STATISTICS")
        print("─" * 60)
        
        print(f"📁 File Information:")
        print(f"   Name: {self.selected_file_path.name}")
        print(f"   Size: {stats.get('file_size_gb', 0):.2f} GB")
        print(f"   Format: {self.selected_file_path.suffix.upper()}")
        
        print(f"\n⏱️  Timing Breakdown:")
        print(f"   Load Time: {stats.get('load_time', 0):.2f}s")
        print(f"   Downsample Time: {stats.get('downsample_time', 0):.2f}s")
        print(f"   Segmentation Time: {stats.get('segmentation_time', 0):.2f}s")
        print(f"   Clustering Time: {stats.get('clustering_time', 0):.2f}s")
        print(f"   Total Time: {stats.get('total_time', 0):.2f}s")
        
        print(f"\n📊 Point Cloud Statistics:")
        print(f"   Original Points: {stats.get('original_points', 0):,}")
        print(f"   Processed Points: {stats.get('downsampled_points', 0):,}")
        print(f"   Downsample Ratio: {stats.get('downsample_ratio', 0):.1%}")
        
        print(f"\n🎯 Detection Results:")
        print(f"   Clusters Found: {stats.get('total_clusters', 0)}")
        print(f"   Instances Detected: {stats.get('total_instances', 0)}")
        if stats.get('car_instances', 0) > 0:
            print(f"   🚗 Cars Detected: {stats.get('car_instances', 0)}")
        
        input("\nPress Enter to continue...")
    
    def show_classification_results(self):
        """Show detailed semantic classification results."""
        if not self.processing_stats or not self.processing_stats.get('class_stats'):
            print("❌ No classification results available. Please process a file first.")
            input("\nPress Enter to continue...")
            return
        
        class_stats = self.processing_stats['class_stats']
        
        print("\n" + "─" * 60)
        print("🏷️  SEMANTIC CLASSIFICATION RESULTS")
        print("─" * 60)
        
        # Sort classes by point count
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        
        print(f"{'Class':<15} {'Points':<12} {'Percentage':<12} {'Confidence':<12} {'Visible'}")
        print("─" * 60)
        
        for class_name, stats in sorted_classes:
            percentage = stats['percentage']
            count = stats['count']
            score = stats.get('mean_score', 0)
            visible = "✅" if self.class_visibility.get(class_name, True) else "❌"
            
            print(f"{class_name:<15} {count:<12,} {percentage:<11.1f}% {score:<11.3f} {visible}")
        
        print(f"\n📊 Total Classes: {len(class_stats)}")
        print(f"📊 Total Points: {sum(stats['count'] for stats in class_stats.values()):,}")
        
        input("\nPress Enter to continue...")
    
    def show_instances(self):
        """Show detailed instance information."""
        if not self.instances_data:
            print("❌ No instances detected. Please process a file first.")
            input("\nPress Enter to continue...")
            return
        
        print("\n" + "─" * 60)
        print("📦 DETECTED INSTANCES")
        print("─" * 60)
        
        # Group instances by class
        instance_by_class = {}
        for instance in self.instances_data:
            class_name = instance['class']
            if class_name not in instance_by_class:
                instance_by_class[class_name] = []
            instance_by_class[class_name].append(instance)
        
        for class_name, instances in instance_by_class.items():
            print(f"\n🏷️  {class_name.upper()} ({len(instances)} instances):")
            
            if class_name == "car":
                # Enhanced display for cars
                print(f"{'ID':<4} {'Points':<8} {'Dimensions (L×W×H)':<20} {'Car Conf':<10} {'Likely Car':<12} {'Visible'}")
                print("─" * 80)
                
                for instance in instances:
                    inst_id = instance['instance_id']
                    points = instance['num_points']
                    car_conf = instance.get('car_confidence', 0.0)
                    is_car = instance.get('is_likely_car', False)
                    car_dims = instance.get('car_dimensions', {})
                    visible = "✅" if self.instance_visibility.get(inst_id, True) else "❌"
                    
                    if car_dims:
                        dims_str = f"{car_dims.get('length', 0):.1f}×{car_dims.get('width', 0):.1f}×{car_dims.get('height', 0):.1f}"
                    else:
                        extent = instance['extent']
                        dims_str = f"{extent[0]:.1f}×{extent[1]:.1f}×{extent[2]:.1f}"
                    
                    car_status = "🚗 YES" if is_car else "❌ NO"
                    
                    print(f"{inst_id:<4} {points:<8} {dims_str:<20} {car_conf:<9.3f} {car_status:<12} {visible}")
            else:
                # Standard display for other classes
                print(f"{'ID':<4} {'Points':<8} {'Centroid (X, Y, Z)':<25} {'Size (X×Y×Z)':<20} {'Visible'}")
                print("─" * 70)
                
                for instance in instances:
                    inst_id = instance['instance_id']
                    points = instance['num_points']
                    centroid = instance['centroid']
                    extent = instance['extent']
                    visible = "✅" if self.instance_visibility.get(inst_id, True) else "❌"
                    
                    centroid_str = f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"
                    extent_str = f"{extent[0]:.1f}×{extent[1]:.1f}×{extent[2]:.1f}"
                    
                    print(f"{inst_id:<4} {points:<8} {centroid_str:<25} {extent_str:<20} {visible}")
        
        print(f"\n📊 Total Instances: {len(self.instances_data)}")
        
        input("\nPress Enter to continue...")
    
    def toggle_visibility(self):
        """Interactive visibility toggling."""
        if not self.segmentation_result and not self.instances_data:
            print("❌ No results available. Please process a file first.")
            input("\nPress Enter to continue...")
            return
        
        while True:
            print("\n" + "─" * 60)
            print("🎨 VISIBILITY CONTROLS")
            print("─" * 60)
            print("1. 🏷️  Toggle semantic class visibility")
            print("2. 📦 Toggle instance visibility")
            print("3. 👁️  Show all")
            print("4. 🙈 Hide all")
            print("0. ⬅️  Back to main menu")
            
            try:
                choice = input("\n👉 Select option (0-4): ").strip()
                
                if choice == "1":
                    self.toggle_class_visibility()
                elif choice == "2":
                    self.toggle_instance_visibility()
                elif choice == "3":
                    self.show_all()
                elif choice == "4":
                    self.hide_all()
                elif choice == "0":
                    break
                else:
                    print("❌ Invalid option. Please select 0-4.")
                    
            except KeyboardInterrupt:
                break
    
    def toggle_class_visibility(self):
        """Toggle individual class visibility."""
        if not self.segmentation_result:
            print("❌ No classification results available.")
            return
        
        class_stats = self.processing_stats.get('class_stats', {})
        classes = list(class_stats.keys())
        
        print("\n🏷️  SEMANTIC CLASS VISIBILITY:")
        for i, class_name in enumerate(classes, 1):
            status = "✅" if self.class_visibility.get(class_name, True) else "❌"
            count = class_stats[class_name]['count']
            print(f"  {i}. {status} {class_name} ({count:,} points)")
        
        try:
            choice = input(f"\n👉 Toggle class (1-{len(classes)}) or Enter to cancel: ").strip()
            if choice and choice.isdigit() and 1 <= int(choice) <= len(classes):
                class_name = classes[int(choice) - 1]
                self.class_visibility[class_name] = not self.class_visibility.get(class_name, True)
                status = "visible" if self.class_visibility[class_name] else "hidden"
                print(f"✅ {class_name} is now {status}")
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    def toggle_instance_visibility(self):
        """Toggle individual instance visibility."""
        if not self.instances_data:
            print("❌ No instances available.")
            return
        
        print(f"\n📦 INSTANCE VISIBILITY ({len(self.instances_data)} instances):")
        for i, instance in enumerate(self.instances_data, 1):
            inst_id = instance['instance_id']
            status = "✅" if self.instance_visibility.get(inst_id, True) else "❌"
            class_name = instance['class']
            points = instance['num_points']
            print(f"  {i:2d}. {status} Instance {inst_id} ({class_name} - {points} points)")
        
        try:
            choice = input(f"\n👉 Toggle instance (1-{len(self.instances_data)}) or Enter to cancel: ").strip()
            if choice and choice.isdigit() and 1 <= int(choice) <= len(self.instances_data):
                instance = self.instances_data[int(choice) - 1]
                inst_id = instance['instance_id']
                self.instance_visibility[inst_id] = not self.instance_visibility.get(inst_id, True)
                status = "visible" if self.instance_visibility[inst_id] else "hidden"
                print(f"✅ Instance {inst_id} is now {status}")
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    def show_all(self):
        """Make all classes and instances visible."""
        for class_name in self.class_visibility:
            self.class_visibility[class_name] = True
        for inst_id in self.instance_visibility:
            self.instance_visibility[inst_id] = True
        print("✅ All classes and instances are now visible")
    
    def hide_all(self):
        """Hide all classes and instances."""
        for class_name in self.class_visibility:
            self.class_visibility[class_name] = False
        for inst_id in self.instance_visibility:
            self.instance_visibility[inst_id] = False
        print("✅ All classes and instances are now hidden")
    
    def export_results(self):
        """Interactive results export."""
        if not self.current_pcd:
            print("❌ No results to export. Please process a file first.")
            input("\nPress Enter to continue...")
            return
        
        print("\n" + "─" * 60)
        print("💾 EXPORT RESULTS")
        print("─" * 60)
        print("1. 🎨 Export colored point cloud (PLY)")
        print("2. 📋 Export instance metadata (JSON)")
        print("3. 📊 Export summary table (CSV)")
        print("4. 📦 Export all formats")
        print("0. ⬅️  Back to main menu")
        
        try:
            choice = input("\n👉 Select export option (0-4): ").strip()
            
            output_dir = Path("console_output")
            output_dir.mkdir(exist_ok=True)
            base_name = self.selected_file_path.stem
            
            if choice == "1" or choice == "4":
                ply_path = output_dir / f"{base_name}_processed.ply"
                export_colored_point_cloud(self.current_pcd, ply_path)
                print(f"✅ Exported PLY: {ply_path}")
            
            if choice == "2" or choice == "4":
                json_path = output_dir / f"{base_name}_instances.json"
                export_instances_json(self.instances_data, json_path)
                print(f"✅ Exported JSON: {json_path}")
            
            if choice == "3" or choice == "4":
                csv_path = output_dir / f"{base_name}_summary.csv"
                rows = []
                for instance in self.instances_data:
                    rows.append({
                        "instance_id": instance["instance_id"],
                        "class": instance["class"],
                        "num_points": instance["num_points"],
                        "centroid_x": instance["centroid"][0],
                        "centroid_y": instance["centroid"][1],
                        "centroid_z": instance["centroid"][2],
                        "extent_x": instance["extent"][0],
                        "extent_y": instance["extent"][1],
                        "extent_z": instance["extent"][2],
                        "mean_score": instance["mean_score"]
                    })
                export_summary_csv(rows, csv_path)
                print(f"✅ Exported CSV: {csv_path}")
            
            if choice in ["1", "2", "3", "4"]:
                print(f"\n📁 All exports saved to: {output_dir.absolute()}")
            elif choice != "0":
                print("❌ Invalid option")
                
        except Exception as e:
            print(f"❌ Export error: {e}")
        
        input("\nPress Enter to continue...")
    
    def auto_export_results(self):
        """Automatically export all results."""
        if not self.current_pcd or not self.instances_data:
            return
        
        try:
            output_dir = Path("console_output")
            output_dir.mkdir(exist_ok=True)
            base_name = self.selected_file_path.stem
            
            # Export colored point cloud
            ply_path = output_dir / f"{base_name}_processed.ply"
            export_colored_point_cloud(self.current_pcd, ply_path)
            print(f"   ✅ PLY: {ply_path.name}")
            
            # Export instances JSON
            json_path = output_dir / f"{base_name}_instances.json"
            export_instances_json(self.instances_data, json_path)
            print(f"   ✅ JSON: {json_path.name}")
            
            # Export summary CSV
            csv_path = output_dir / f"{base_name}_summary.csv"
            rows = []
            for instance in self.instances_data:
                rows.append({
                    "instance_id": instance["instance_id"],
                    "class": instance["class"],
                    "num_points": instance["num_points"],
                    "centroid_x": instance["centroid"][0],
                    "centroid_y": instance["centroid"][1],
                    "centroid_z": instance["centroid"][2],
                    "extent_x": instance["extent"][0],
                    "extent_y": instance["extent"][1],
                    "extent_z": instance["extent"][2],
                    "mean_score": instance["mean_score"]
                })
            export_summary_csv(rows, csv_path)
            print(f"   ✅ CSV: {csv_path.name}")
            
            print(f"\n📁 Results exported to: {output_dir.absolute()}")
            
        except Exception as e:
            print(f"❌ Auto-export error: {e}")
    
    def show_help(self):
        """Show help information."""
        print("\n" + "─" * 60)
        print("❓ HELP & INFORMATION")
        print("─" * 60)
        
        print("🎯 ABOUT THIS APPLICATION:")
        print("   This console-based application provides complete point cloud")
        print("   semantic segmentation and instance detection functionality.")
        print("   It bypasses all graphics driver issues by using text-based interface.")
        
        print("\n📋 WORKFLOW:")
        print("   1. Select a point cloud file (.ply, .pcd, .las)")
        print("   2. Process the file (automatic downsampling to ~100k points)")
        print("   3. View detailed results and statistics")
        print("   4. Toggle visibility of classes and instances")
        print("   5. Export results in multiple formats")
        
        print("\n🎨 FEATURES:")
        print("   • Semantic segmentation (road, building, car, etc.)")
        print("   • Instance detection and clustering")
        print("   • Detailed processing statistics")
        print("   • Interactive visibility controls")
        print("   • Multiple export formats (PLY, JSON, CSV)")
        print("   • Automatic result export")
        
        print("\n💾 OUTPUT FILES:")
        print("   • *_processed.ply: Colored point cloud with classifications")
        print("   • *_instances.json: Detailed instance metadata")
        print("   • *_summary.csv: Tabular summary of all instances")
        
        print("\n🔧 SUPPORTED FORMATS:")
        print("   Input: .ply, .pcd, .las files")
        print("   Output: PLY (colored), JSON (metadata), CSV (summary)")
        
        input("\nPress Enter to continue...")
    
    def show_available_models(self):
        """Show available models and methods."""
        print("\n" + "─" * 60)
        print("🤖 AVAILABLE MODELS & METHODS")
        print("─" * 60)
        
        # Show available methods
        print("🎯 SEGMENTATION METHODS:")
        methods = get_available_methods()
        for i, method in enumerate(methods, 1):
            print(f"   {i}. {method.upper()}")
        
        recommended_method = get_recommended_method()
        print(f"\n   💡 Recommended: {recommended_method.upper()}")
        
        # Show available models
        print("\n🧠 PRETRAINED MODELS:")
        models = get_available_models()
        if models:
            for model_name in models:
                info = get_model_info(model_name)
                if info:
                    print(f"   📦 {model_name}")
                    print(f"      Type: {info.get('model_type', 'unknown')}")
                    print(f"      Dataset: {info.get('dataset', 'unknown')}")
                    print(f"      Classes: {info.get('num_classes', 'unknown')}")
                    print(f"      Description: {info.get('description', 'No description')}")
                    print()
        else:
            print("   ⚠️  No pretrained models available")
            print("   💡 Models will be downloaded automatically when needed")
        
        # Show model recommendations
        print("🎯 MODEL RECOMMENDATIONS:")
        recommendations = {
            "toronto3d": "Urban scenes with buildings, roads, and infrastructure",
            "semantickitti": "Road scenes with vehicles and traffic elements",
            "urban": "Large-scale urban environments",
            "general": "General purpose (default)"
        }
        
        for dataset_type, description in recommendations.items():
            recommended_model = get_recommended_model(dataset_type)
            print(f"   🎯 {dataset_type.upper()}: {recommended_model}")
            print(f"      {description}")
        
        print("\n💡 TIPS:")
        print("   • Models are automatically downloaded when first used")
        print("   • Toronto3D models are best for urban environments")
        print("   • SemanticKITTI models are best for road scenes")
        print("   • The system automatically selects the best model")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Run the console application."""
        try:
            self.show_main_menu()
        except KeyboardInterrupt:
            print("\n👋 Application closed by user")
        except Exception as e:
            print(f"❌ Application error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    print("🚀 Starting Console Point Cloud Semantic Segmentation...")
    print("📟 Graphics-free interface for maximum compatibility")
    print("🎯 Complete semantic segmentation and instance detection")
    print("💾 Automatic export to multiple formats")
    print("-" * 70)
    
    try:
        app = ConsoleSemanticSegmentationGUI()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Application closed by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
