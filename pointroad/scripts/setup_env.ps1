Param(
    [switch]$NoRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[PointRoad] Creating venv with Python 3.10..."
if (-Not (Test-Path ".\.venv")) {
    try {
        py -3.10 -m venv .\.venv
    } catch {
        Write-Warning "Python 3.10 launcher not found; falling back to default python"
        python -m venv .\.venv
    }
}

Write-Host "[PointRoad] Activating venv..."
& .\.venv\Scripts\Activate.ps1

Write-Host "[PointRoad] Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "[PointRoad] Installing pinned deps... (this may take a while)"
pip install -r .\requirements.txt -c .\constraints.txt

Write-Host "[PointRoad] Installing PointRoad package (editable)"
pip install -e .

Write-Host "[PointRoad] Verifying Open3D import..."
python -c "import open3d as o3d; print('Open3D', o3d.__version__)"

Write-Host "[PointRoad] Smoke test: launch GUI..."
if (-Not $NoRun) {
    python -m pointroad.gui.app
}


