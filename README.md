# Nexus Vision: Autonomous Terrain Segmentation & Agentic Fusion

## Overview
This repository contains the Nexus Vision pipeline for the BigRock Exchange Hackathon. It features a robust semantic segmentation model to classify offroad terrain, enhanced with our novel **Agentic Fusion** architecture that integrates YOLOv8n for real-time dynamic object detection.

## Dual-Stream Architecture
- **Stream A (The "Where")**: SegFormer/DINOv2 Semantic Segmentation for drivable terrain analysis.
- **Stream B (The "What")**: YOLOv8n Object Detection for dynamic obstacles and people.
- **Fusion**: YOLO bounding boxes overlaid on the high-contrast segmentation mask for a comprehensive scene understanding.

## Requirements
Ensure you have the required dependencies installed (using the provided `ENV_SETUP` scripts or manually):
```bash
pip install torch torchvision ultralytics opencv-python numpy
```

## Running the Fusion Pipeline
The `fusion_vision.py` script combines the segmentation mask and YOLO object detection.

```bash
python fusion_vision.py --img <path_to_image> --mask <path_to_segmentation_mask> --out <output_image_name>
```

**Example:**
```bash
python fusion_vision.py --img TestImages/Offroad_Segmentation_testImages/Color_Images/0000060.png --mask TestImages/Offroad_Segmentation_testImages/Segmentation/0000060.png --out fusion_demo.png
```

## Expected Output
The script will output a merged image (e.g., `fusion_demo.png`) where:
1. The background terrain is colorized according to the hackathon's distinct color palette.
2. YOLOv8 bounding boxes (in red) are overlaid on top to identify objects with high confidence.

## Evaluation
Our training pipeline (`train_segmentation.py`) evaluates the model using Mean Intersection over Union (mIoU) and handles rare classes (like Logs and Flowers) through logarithmic class weighting.
