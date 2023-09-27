Human 3D Point Cloud Detection using Intel RealSense D435i Camera
This project aims to detect human keypoints in 3D space using the Intel RealSense D435i camera. We utilize the Point Cloud Generator based on the Intel RealSense OpenCV Python SDK and integrate it with MediaPipe for human keypoint detection.

Table of Contents
Prerequisites
Installation
Usage
Key Features
Contributing
License
Prerequisites
Intel RealSense D435i Camera
Python environment
Installation
Intel RealSense OpenCV Python SDK

Follow the installation guide provided by Intel here.

MediaPipe

Install MediaPipe using pip:

bash
Copy code
pip install mediapipe
Usage
Connect your Intel RealSense D435i Camera to your computer.
Run the Python script to start the point cloud generation and human keypoint detection.
The program will display the video feed with human keypoints detected. Each keypoint will have its X, Y, and Z coordinates displayed.
Key Features
3D Point Cloud Generation: Using the Intel RealSense SDK, we generate a 3D point cloud of the environment.
Human Keypoint Detection: With the integration of MediaPipe, the program can detect human keypoints in real-time.
Z-axis Coordinate Display: For each detected human keypoint, the program displays the Z-axis coordinate, giving a depth perspective.
Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

License
MIT
