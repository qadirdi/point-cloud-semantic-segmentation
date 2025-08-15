from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Optional
import psutil


@dataclass
class SystemInfo:
    os: str
    python_version: str
    total_ram_gb: float
    cpu: str
    cuda_available: bool


def detect_system() -> SystemInfo:
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    # CUDA detection deferred to torch if installed. Import lazily to avoid import cost.
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    return SystemInfo(
        os=platform.platform(),
        python_version=platform.python_version(),
        total_ram_gb=total_ram_gb,
        cpu=platform.processor() or platform.machine(),
        cuda_available=cuda_available,
    )


__all__ = ["SystemInfo", "detect_system"]



