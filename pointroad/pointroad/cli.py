from __future__ import annotations

from pathlib import Path
import json
import click
import open3d as o3d
import numpy as np

from .utils.log import setup_logger
from .io.loader import load_point_cloud
from .io.exporter import export_colored_point_cloud, export_instances_json, export_summary_csv
from .ml.preprocess import DownsampleConfig, adaptive_voxel_downsample
from .ml.infer import run_segmentation
from .post.cluster import cluster_per_class, cluster_all_points
from .post.color import build_class_palette, colorize_by_class, vary_instance_brightness, generate_distinct_colors


@click.group()
def pcdseg():
    """PointRoad CLI."""


@pcdseg.command()
@click.option("--input", "input_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--preset", type=click.Choice(["fast", "balanced", "accurate"]), default="balanced")
@click.option("--export", "export_dir", type=click.Path(path_type=Path), required=False)
def run(input_path: Path, preset: str, export_dir: Path | None):
    setup_logger()
    pcd = load_point_cloud(input_path)
    presets = {
        "fast": dict(target_points_min=150_000, target_points_max=250_000, initial_voxel_size=0.08),
        "balanced": dict(target_points_min=200_000, target_points_max=400_000, initial_voxel_size=0.05),
        "accurate": dict(target_points_min=400_000, target_points_max=700_000, initial_voxel_size=0.03),
    }
    ds_cfg = DownsampleConfig(**presets[preset])
    ds, vx = adaptive_voxel_downsample(pcd, ds_cfg)
    result = run_segmentation(ds)
    # Temporary mapping: treat label 0 as 'road'
    class_names = np.array(["road"] * len(result.labels))
    palette = build_class_palette({
        "road": "#808080",
        "sidewalk": "#A9A9A9",
        "curb": "#FFD700",
        "pole": "#87CEFA",
        "traffic_sign": "#FFA500",
        "car": "#FF0000",
        "bicycle": "#00FF00",
        "pedestrian": "#800080",
        "vegetation": "#006400",
        "building": "#8B4513",
    })
    colors = colorize_by_class(class_names, palette)
    ds.colors = o3d.utility.Vector3dVector(colors)

    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)
        export_colored_point_cloud(ds, export_dir / "colored.ply")
        export_instances_json([], export_dir / "instances.json")
        export_summary_csv([], export_dir / "summary.csv")
    click.echo("Done")


@pcdseg.command()
@click.option("--input", "input_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--eps", type=float, default=0.5, help="DBSCAN eps in meters")
@click.option("--minpts", type=int, default=30, help="DBSCAN min points")
@click.option("--export", "export_dir", type=click.Path(path_type=Path), required=True)
def cluster(input_path: Path, eps: float, minpts: int, export_dir: Path):
    """Headless clustering and coloring without GUI: assigns unique colors per detected object and exports."""
    from loguru import logger
    
    setup_logger()
    logger.info(f"Starting clustering on {input_path}")
    logger.info(f"Parameters: eps={eps}, min_points={minpts}")
    
    # Load point cloud
    logger.info("Loading point cloud...")
    pcd = load_point_cloud(input_path)
    points_np = np.asarray(pcd.points)
    logger.info(f"Loaded {len(points_np)} points")
    logger.info(f"Point cloud bounds: X[{points_np[:, 0].min():.2f}, {points_np[:, 0].max():.2f}], "
                f"Y[{points_np[:, 1].min():.2f}, {points_np[:, 1].max():.2f}], "
                f"Z[{points_np[:, 2].min():.2f}, {points_np[:, 2].max():.2f}]")
    
    # Run clustering
    logger.info("Running DBSCAN clustering...")
    labels = cluster_all_points(pcd, eps=eps, min_points=minpts)
    
    # Analyze results
    unique_labels, counts = np.unique(labels, return_counts=True)
    noise_points = counts[unique_labels == -1][0] if -1 in unique_labels else 0
    valid_clusters = unique_labels[unique_labels >= 0]
    num_clusters = len(valid_clusters)
    
    logger.info(f"Clustering complete!")
    logger.info(f"  - Total clusters found: {num_clusters}")
    logger.info(f"  - Noise points: {noise_points}")
    logger.info(f"  - Points in clusters: {len(points_np) - noise_points}")
    
    if num_clusters > 0:
        logger.info(f"  - Cluster sizes: min={counts[valid_clusters].min()}, "
                   f"max={counts[valid_clusters].max()}, "
                   f"mean={counts[valid_clusters].mean():.1f}")
    
    # Generate colors
    logger.info("Generating distinct colors for clusters...")
    colors = np.zeros((len(points_np), 3), dtype=np.float64)
    
    if num_clusters > 0:
        palette = generate_distinct_colors(num_clusters)
        logger.info(f"Generated {len(palette)} distinct colors")
        
        for i, cluster_id in enumerate(valid_clusters):
            mask = labels == cluster_id
            colors[mask] = palette[i]
            logger.info(f"  Cluster {cluster_id}: {counts[unique_labels == cluster_id][0]} points, "
                       f"color RGB={palette[i]}")
    
    # Set colors
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Export
    logger.info(f"Exporting results to {export_dir}")
    export_dir.mkdir(parents=True, exist_ok=True)
    export_colored_point_cloud(pcd, export_dir / "colored_instances.ply")
    
    # Generate summaries
    logger.info("Generating instance summaries...")
    instances = []
    rows = []
    
    for i, cluster_id in enumerate(valid_clusters):
        idxs = np.where(labels == cluster_id)[0]
        if idxs.size == 0:
            continue
            
        cluster_points = points_np[idxs]
        centroid = cluster_points.mean(axis=0)
        
        # Create sub-pointcloud for bounding box calculation
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(cluster_points)
        aabb = sub.get_axis_aligned_bounding_box()
        
        # Calculate extent
        extent = aabb.get_extent()
        
        inst = {
            "instance_id": int(cluster_id),
            "class": "object",
            "num_points": int(idxs.size),
            "centroid": centroid.tolist(),
            "aabb_min": aabb.get_min_bound().tolist(),
            "aabb_max": aabb.get_max_bound().tolist(),
            "extent": extent.tolist(),
        }
        instances.append(inst)
        
        rows.append({
            "instance_id": int(cluster_id),
            "class": "object",
            "num_points": int(idxs.size),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "extent_x": float(extent[0]),
            "extent_y": float(extent[1]),
            "extent_z": float(extent[2]),
        })
        
        logger.info(f"  Instance {cluster_id}: {idxs.size} points, "
                   f"centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), "
                   f"extent=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})")
    
    export_instances_json(instances, export_dir / "instances.json")
    export_summary_csv(rows, export_dir / "summary.csv")
    
    logger.info(f"Export complete! Files saved to {export_dir}")
    logger.info(f"  - colored_instances.ply: Point cloud with cluster colors")
    logger.info(f"  - instances.json: Detailed instance metadata")
    logger.info(f"  - summary.csv: Tabular instance summary")
    
    click.echo(f"✅ Clustered {num_clusters} objects from {len(points_np)} points")
    click.echo(f"📁 Outputs saved to: {export_dir}")
    if num_clusters > 0:
        click.echo(f"🎨 Each cluster has a unique color in the PLY file")
    else:
        click.echo(f"⚠️  No clusters found - try adjusting eps/minpts parameters")


def main():
    pcdseg()


if __name__ == "__main__":
    main()


