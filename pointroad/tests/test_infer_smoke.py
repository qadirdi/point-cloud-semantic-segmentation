from __future__ import annotations

import numpy as np
import open3d as o3d
from pointroad.ml.infer import run_segmentation


def test_smoke_infer_dummy():
    pts = np.random.rand(1000, 3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    res = run_segmentation(pcd, backend="onnx", use_gpu_if_available=False)
    assert res.labels.shape[0] == pts.shape[0]
    assert res.scores.min() >= 0



