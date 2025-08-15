from __future__ import annotations

from typing import Dict
import numpy as np
import open3d as o3d


def class_counts(class_names: np.ndarray) -> Dict[str, int]:
    unique, counts = np.unique(class_names, return_counts=True)
    return {str(k): int(v) for k, v in zip(unique, counts)}


__all__ = ["class_counts"]



