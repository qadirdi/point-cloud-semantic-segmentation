from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import json
import pandas as pd
import open3d as o3d


def export_colored_point_cloud(pcd: o3d.geometry.PointCloud, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=True)


def export_instances_json(instances: List[Dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(instances, f, indent=2)


def export_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


__all__ = [
    "export_colored_point_cloud",
    "export_instances_json",
    "export_summary_csv",
]



