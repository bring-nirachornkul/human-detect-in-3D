# Human 3D Point Cloud Detection using Intel RealSense D435i Camera

Detect human keypoints in 3D space using the Intel RealSense D435i camera. This project leverages the Point Cloud Generator from the Intel RealSense OpenCV Python SDK and integrates it with MediaPipe for real-time human keypoint detection.


## Videos

### RGB Camera and Depth Camera
[![RGB Camera and Depth Camera](http://img.youtube.com/vi/qY6Oh9aakHA/0.jpg)](https://www.youtube.com/watch?v=qY6Oh9aakHA)

### RGBD: Measure the Depth of Human's Landmarks
[![RGBD: Measure the Depth of Human's Landmarks](http://img.youtube.com/vi/VCi-aCqc92I/0.jpg)](https://youtu.be/VCi-aCqc92I)


## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites
- Intel RealSense D435i Camera
- Python environment

## Installation

### Intel RealSense OpenCV Python SDK
Follow the installation guide provided by Intel [here](https://github.com/IntelRealSense/librealsense).


### MediaPipe
Install MediaPipe using pip:
\```bash
pip install mediapipe
\```

## Usage
1. Connect your Intel RealSense D435i Camera to your computer.
2. Run the Python script to initiate the point cloud generation and human keypoint detection.
3. The program will display the video feed with detected human keypoints. Each keypoint will show its X, Y, and Z coordinates.

## Key Features
- **3D Point Cloud Generation**: Generate a 3D point cloud of the environment using the Intel RealSense SDK.
- **Human Keypoint Detection**: Detect human keypoints in real-time with the help of MediaPipe.
- **Z-axis Coordinate Display**: Display the Z-axis coordinate for each detected human keypoint, offering a depth perspective.

## Contributing
Pull requests are welcome. For significant changes, please open an issue first to discuss your proposed changes.

## License
[MIT](LICENSE)
