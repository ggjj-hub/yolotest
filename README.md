# YOLO11 Class Test

A Python-based testing and development repository for YOLOv11 (You Only Look Once version 11) object detection and pose estimation models.

## 📋 Description

This repository contains testing scripts and utilities for YOLOv11, a state-of-the-art real-time object detection and pose estimation framework. The project focuses on validating YOLOv11's capabilities through various test cases and automation workflows.

## 🛠️ Technologies

- **Language:** Python
- **Primary Framework:** YOLOv11
- **Pre-trained Models:** 
  - `yolo11n.pt` - YOLOv11 nano model (object detection)
  - `yolo11n-pose.pt` - YOLOv11 nano model (pose estimation)

## 📁 Project Structure

### Core Testing Scripts
- **`test.py`** - Main test script for YOLOv11 object detection
- **`test-pose.py`** - Pose estimation testing
- **`jiance.py`** - Detection/measurement utilities
- **`gpu.py`** - GPU-related testing and diagnostics

### Automation & Labeling
- **`auto_label.py`** - Automatic image labeling
- **`fast_auto_label.py`** - Optimized labeling for faster processing
- **`cont.py`** - Continuous processing utilities

### Training & Deployment
- **`train_action.py`** - Training workflow automation
- **`test_action.py`** - Testing workflow automation
- **`worker.py`** - Background worker processes
- **`poseggjj.py`** - Pose estimation specific implementations

### Utilities
- **`video_cut.py`** - Video processing and segmentation
- **`seed.yaml`** - Configuration/seed data
- **`seed_data.cache`** - Cached seed data

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision
pip install ultralytics  # YOLOv11 framework
```

### Running Tests

**Basic Object Detection Test:**
```bash
python test.py
```

**Pose Estimation Test:**
```bash
python test-pose.py
```

**GPU Diagnostics:**
```bash
python gpu.py
```

### Auto Labeling

**Automatic Label Generation:**
```bash
python auto_label.py
```

**Fast Auto Labeling (Optimized):**
```bash
python fast_auto_label.py
```

## 📊 Features

- **Object Detection** - YOLOv11 object detection capabilities
- **Pose Estimation** - Human pose keypoint detection
- **Automated Labeling** - Batch image annotation
- **GPU Support** - Optimized for GPU acceleration
- **Training Workflows** - Automated model training pipelines
- **Video Processing** - Video analysis and frame extraction

## 🔧 Configuration

Configuration settings are defined in `seed.yaml`. Modify this file to adjust:
- Model parameters
- Input/output paths
- Detection thresholds
- Pose estimation settings

## ⚙️ Workflow Automation

The repository includes GitHub Actions-compatible scripts:
- **`train_action.py`** - Training automation
- **`test_action.py`** - Testing automation

These enable CI/CD integration for continuous model validation and training.

## 📝 Notes

- Pre-trained models (`yolo11n.pt`, `yolo11n-pose.pt`) are included
- GPU support recommended for optimal performance
- Supports both CPU and GPU inference

## 📄 License

This project does not specify a license. Please check with the repository owner for licensing information.

## 👤 Author

[ggjj-hub](https://github.com/ggjj-hub)

---

**Last Updated:** 2026-03-05
