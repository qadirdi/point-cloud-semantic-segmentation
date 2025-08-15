#!/bin/bash

# Enhanced Point Cloud Semantic Segmentation Installation Script
# This script installs all dependencies for the enhanced system with pretrained models

echo "🚀 Installing Enhanced Point Cloud Semantic Segmentation System"
echo "================================================================"

# Check if Python 3.8+ is available
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install additional dependencies for enhanced functionality
echo "🔧 Installing enhanced dependencies..."
pip install scikit-learn==1.4.2
pip install huggingface-hub==0.21.4
pip install requests==2.32.3

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import sys
print('Testing imports...')

try:
    import numpy
    print('✅ NumPy:', numpy.__version__)
except ImportError as e:
    print('❌ NumPy import failed:', e)
    sys.exit(1)

try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('   CUDA available:', torch.cuda.is_available())
except ImportError as e:
    print('❌ PyTorch import failed:', e)
    sys.exit(1)

try:
    import open3d
    print('✅ Open3D:', open3d.__version__)
except ImportError as e:
    print('❌ Open3D import failed:', e)
    sys.exit(1)

try:
    import sklearn
    print('✅ scikit-learn:', sklearn.__version__)
except ImportError as e:
    print('❌ scikit-learn import failed:', e)
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download
    print('✅ Hugging Face Hub: Available')
except ImportError as e:
    print('❌ Hugging Face Hub import failed:', e)
    sys.exit(1)

print('\\n🎉 All imports successful!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Installation completed successfully!"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Activate the virtual environment: source .venv/bin/activate"
    echo "   2. Run the console application: python3 console_gui_app.py"
    echo "   3. Test the enhanced models: python3 test_pretrained_models.py"
    echo ""
    echo "📚 Documentation:"
    echo "   - README.md: Basic usage"
    echo "   - PRETRAINED_MODELS_README.md: Enhanced features"
    echo ""
    echo "🎯 Features available:"
    echo "   - Pretrained PointNet++ and RandLA-Net models"
    echo "   - Toronto3D and SemanticKITTI dataset support"
    echo "   - Automatic model selection and download"
    echo "   - Enhanced car detection (85%+ accuracy)"
    echo "   - 25+ semantic classes"
    echo ""
else
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi