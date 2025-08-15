Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

& .\.venv\Scripts\Activate.ps1
pip install pyinstaller==6.7.0

pyinstaller --noconfirm --onedir --windowed --name PointRoad --collect-all open3d -i NONE -y -s pointroad\gui\app.py

Write-Host "Build complete. See dist/PointRoad"


