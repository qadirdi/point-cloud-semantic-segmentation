# Point Cloud Semantic Segmentation

A production-ready Python application for semantic segmentation and instance detection of large road-scene point clouds with enhanced car detection accuracy.

## 🚀 Features

- **Enhanced Car Detection**: Sophisticated algorithms specifically optimized for precise vehicle identification
- **Semantic Segmentation**: Identifies roads, buildings, cars, vegetation, sidewalks, poles, and more
- **Instance Detection**: Groups points into individual object instances with detailed metadata
- **Console Interface**: User-friendly text-based GUI that works on any system
- **Multiple Formats**: Supports .ply, .pcd, and .las input files
- **Adaptive Processing**: Intelligent downsampling targeting ~100k points for optimal performance
- **Comprehensive Export**: Colored PLY, detailed JSON metadata, and CSV summaries

## 📋 Requirements

- Python 3.8+
- Windows 11 (tested)
- ~4GB RAM for large point clouds
- No NVIDIA CUDA required (CPU-only processing)

## 🛠️ Installation

1. **Clone and setup environment:**
   ```bash
   git clone <repository>
   cd ClassificationC4
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import open3d; print('✅ Open3D installed successfully')"
   ```

## 🎯 Usage

### Console Application (Recommended)

Launch the interactive console GUI:

```bash
python console_gui_app.py
```

**Features:**
- 📁 **File Selection**: Choose from available point cloud files
- ⚙️ **Processing**: Automatic enhanced semantic segmentation and car detection
- 📊 **Statistics**: Detailed processing timing and point cloud information
- 🏷️ **Classification**: View semantic class results with confidence scores
- 📦 **Instances**: Examine detected objects with car-specific analysis
- 🎨 **Visibility**: Toggle display of classes and individual instances
- 💾 **Export**: Save results in multiple formats

### Command Line Interface

For headless processing:

```bash
python -m pointroad.cli run <input.ply>
```

## 📊 Enhanced Car Detection

The application includes sophisticated car detection algorithms:

### Multi-Feature Analysis
- **Height-based classification**: Cars typically 1-2.5m above ground
- **Geometric analysis**: Validates car-like dimensions (3-6m length, 1.5-2.5m width)
- **Density scoring**: Analyzes point density patterns
- **Spatial clustering**: Groups coherent point clusters
- **Surface complexity**: Examines normal variations for mixed surfaces

### Car-Specific Metrics
- **Confidence scores**: Multiple validation layers
- **Dimensional analysis**: Length×Width×Height validation
- **Aspect ratios**: Geometric plausibility checks
- **Volume analysis**: Realistic size constraints

## 📁 Project Structure

```
ClassificationC4/
├── console_gui_app.py          # Main console application
├── console_output/             # Processing results
├── pointroad/                  # Core package
│   ├── pointroad/
│   │   ├── io/                # File I/O operations
│   │   ├── ml/                # Machine learning modules
│   │   │   ├── enhanced_infer.py    # Enhanced car detection
│   │   │   └── model_loader.py      # Model management
│   │   ├── post/              # Post-processing
│   │   │   ├── enhanced_cluster.py  # Car-optimized clustering
│   │   │   └── color.py            # Visualization colors
│   │   └── utils/             # Utilities
│   └── scripts/               # Setup and build scripts
├── requirements.txt           # Dependencies
├── constraints.txt           # Version constraints
└── pyproject.toml           # Project configuration
```

## 🎨 Output Files

### Colored Point Cloud (PLY)
- Semantic colors for all classes
- Enhanced highlighting for detected cars
- Compatible with any 3D viewer

### Instance Metadata (JSON)
```json
{
  "instance_id": 1,
  "class": "car",
  "num_points": 1247,
  "car_confidence": 0.874,
  "is_likely_car": true,
  "car_dimensions": {
    "length": 4.2,
    "width": 1.8,
    "height": 1.5
  }
}
```

### Summary Table (CSV)
Tabular data for analysis and reporting with car-specific metrics.

## 🔧 Configuration

Default settings in `pointroad/config/defaults.yaml`:
- Target processing points: 100,000
- Car clustering epsilon: 0.3
- Minimum car points: 30
- Car dimension validation ranges

## 🚗 Car Detection Accuracy

The enhanced algorithms provide:
- **Precision**: Validates geometric constraints
- **Recall**: Multi-feature scoring prevents missed detections
- **Confidence**: Detailed scoring for result reliability
- **Dimensions**: Real-world size validation

## 📈 Performance

**Typical Processing Times:**
- Loading: ~9s (2.4GB file)
- Downsampling: ~6s (52M → 90K points)
- Enhanced Segmentation: ~0.4s
- Car-Optimized Clustering: ~20s
- **Total**: ~35s for large road scenes

## 🐛 Troubleshooting

**Graphics Issues**: The console application bypasses all graphics driver problems by using text-based interface.

**Memory Issues**: Automatic adaptive downsampling handles large files efficiently.

**Missing Dependencies**: All required packages are pinned in requirements.txt.

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

**Enhanced Point Cloud Processing with Precision Car Detection** 🚗✨
