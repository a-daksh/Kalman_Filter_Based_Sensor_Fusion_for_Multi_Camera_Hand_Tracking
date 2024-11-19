# Kalman_Filter_Based_Sensor_Fusion_for_Multi_Camera_Hand_Tracking

This repository contains the implementation of a system for improving joint angle estimation in prosthetic hands by fusing dual camera sensors using Kalman filters. The project leverages **MediaPipe** for hand tracking and **YOLO** for real-time hand-object interaction tracking, with an evaluation based on the Grasp Quality Index.

## Features
- **Kalman Filter Implementation**: 
  - Combines data from dual camera sensors to enhance the accuracy of joint angle estimation for prosthetic hands.
  - Improves reliability of joint estimation without requiring explicit camera calibration.

- **Hand Tracking with MediaPipe**: 
  - Utilizes MediaPipe for accurate joint angle estimation.

- **Object Interaction Tracking with YOLO**:
  - Real-time detection and tracking of hand-object interactions.
  - Evaluation of the fused model based on the Grasp Quality Index.

## File Structure

| File Name               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `cam_mixed.csv`         | Sample dataset combining readings from dual cameras for sensor fusion.     |
| `graph.py`              | Script for visualizing the results, such as joint estimation accuracy.     |
| `Hand_Graph.png`        | Example output graph showing joint estimation improvement.                 |
| `kalman.py`             | Kalman filter implementation for fusing dual camera sensor data.          |
| `threaded_Camera.py`    | Multi-threaded implementation for real-time camera data acquisition.       |
| `threaded_Camera_2.py`  | Extended version with enhancements for real-time processing.               |

## Prerequisites

1. **Python 3.8+**
2. Install required libraries:
   ```bash
   pip install numpy opencv-python mediapipe torch torchvision
