from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b


def build_class_palette(hex_map: Dict[str, str]) -> Dict[str, np.ndarray]:
    return {k: np.array(hex_to_rgb01(v), dtype=np.float64) for k, v in hex_map.items()}


def colorize_by_class(class_names: np.ndarray, palette: Dict[str, np.ndarray]) -> np.ndarray:
    colors = np.zeros((len(class_names), 3), dtype=np.float64)
    for cname, rgb in palette.items():
        mask = class_names == cname
        colors[mask] = rgb
    return colors


def vary_instance_brightness(colors: np.ndarray, instance_ids: np.ndarray) -> np.ndarray:
    out = colors.copy()
    unique = np.unique(instance_ids)
    for uid in unique:
        if uid < 0:
            continue
        factor = 0.85 + 0.25 * ((uid * 0.37) % 1.0)
        mask = instance_ids == uid
        out[mask] = np.clip(out[mask] * factor, 0.0, 1.0)
    return out


__all__ = [
    "hex_to_rgb01",
    "build_class_palette",
    "colorize_by_class",
    "vary_instance_brightness",
]

def hsv_to_rgb01(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return r, g, b


def generate_distinct_colors(num: int, seed: int = 0) -> np.ndarray:
    """Generate distinct colors using golden ratio spacing in HSV space."""
    if num <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    
    colors = np.zeros((num, 3), dtype=np.float64)
    golden_ratio = 0.61803398875
    h = (seed * golden_ratio) % 1.0
    
    for i in range(num):
        h = (h + golden_ratio) % 1.0
        s = 0.7 + 0.2 * (i % 3)  # Vary saturation
        v = 0.8 + 0.15 * (i % 2)  # Vary brightness
        colors[i] = np.array(hsv_to_rgb01(h, s, v), dtype=np.float64)
    
    return colors



