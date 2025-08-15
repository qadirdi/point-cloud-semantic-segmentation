### Quick start (Windows 11, CPU-only)

1) Open PowerShell in the repository root.
2) Run:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

This creates `.venv`, installs pinned dependencies, verifies Open3D, and launches the GUI.

### Troubleshooting
- White or blank window: update Windows graphics drivers; ensure hardware acceleration is enabled. Open3D uses ANGLE/DirectX on Windows.
- If `open3d-ml` install fails, re-run the script. For very slow CPUs, expect longer install times.
- To skip launching GUI: `PowerShell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -NoRun`



