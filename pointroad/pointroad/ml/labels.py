from __future__ import annotations

from typing import Dict, List


SEMANTIC_KITTI_LABELS: Dict[int, str] = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
}


# Mapping to our GUI classes
CANONICAL_CLASSES: List[str] = [
    "road",
    "sidewalk",
    "curb",
    "pole",
    "traffic_sign",
    "car",
    "bicycle",
    "pedestrian",
    "vegetation",
    "building",
]


KITTI_TO_CANONICAL: Dict[int, str] = {
    40: "road",
    48: "sidewalk",
    80: "pole",
    81: "traffic_sign",
    10: "car",
    11: "bicycle",
    30: "pedestrian",
    70: "vegetation",
    50: "building",
}


__all__ = [
    "SEMANTIC_KITTI_LABELS",
    "CANONICAL_CLASSES",
    "KITTI_TO_CANONICAL",
]



