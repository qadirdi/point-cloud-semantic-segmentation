# Dummy Model Files

These files are placeholders for the actual pretrained models. They contain random data and **will not work** for actual inference.

## Why Dummy Files?

These dummy files allow the system to find model files during startup, which helps with testing the system's file loading and configuration.

## How to Get Real Models

For actual inference, you need real pretrained models. You can:

1. Get a Hugging Face token and run:
   ```
   python download_models.py --token YOUR_TOKEN_HERE
   ```

2. Manually download the models from the Hugging Face repositories and place them in this directory with the correct names:
   - pointnet2_toronto3d.pth
   - randla_net_toronto3d.pth
   - pointnet2_semantickitti.pth

## Important Note

When using the system with these dummy files, it will fall back to the dummy segmentation method, which uses simple heuristics instead of deep learning.
