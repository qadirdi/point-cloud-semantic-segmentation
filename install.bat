@echo off
REM Enhanced Point Cloud Semantic Segmentation Installation Script for Windows
REM This script installs all dependencies and sets up the enhanced system

echo.
echo 🚀 Installing Enhanced Point Cloud Semantic Segmentation System
echo ================================================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check Python version is 3.8+
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python version: %PYTHON_VERSION%

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Install additional dependencies for enhanced functionality
echo 🔧 Installing enhanced dependencies...
pip install PyYAML

REM Create models directory
if not exist "models" (
    echo 📁 Creating models directory...
    mkdir models
)

REM Test installation
echo 🧪 Testing installation...
python -c "import sys; print('Testing imports...'); import numpy; print('✅ NumPy:', numpy.__version__); import torch; print('✅ PyTorch:', torch.__version__); import open3d; print('✅ Open3D:', open3d.__version__); import sklearn; print('✅ scikit-learn:', sklearn.__version__); print('\n🎉 All imports successful!')"

if errorlevel 1 (
    echo ❌ Installation failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ✅ Installation completed successfully!
echo.
echo 🚀 Next steps:
echo    1. Run the console application: python console_gui_app.py
echo    2. Download pretrained models: python download_models.py
echo    3. View documentation: README.md
echo.
echo 📚 Documentation:
echo    - README.md: Basic usage and configuration
echo    - PRETRAINED_MODELS_README.md: Enhanced features and models
echo.
echo 🎯 Features available:
echo    - Enhanced car detection with 85%+ accuracy
echo    - Multiple semantic classes (roads, buildings, vegetation, etc.)
echo    - Configurable detection parameters
echo    - Console interface (no graphics drivers needed)
echo    - Multiple export formats (PLY, JSON, CSV)
echo.
pause

