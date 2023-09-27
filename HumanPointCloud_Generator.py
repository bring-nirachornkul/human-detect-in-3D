#Point Cloud Generator Based on the Intel RealSense OpenCV Python SDK
# which can be found here: https://dev.intelrealsense.com/docs/python2

"""
    [e]     Export points to ply (./out.ply)
    [v]     View most recent .ply fle
    [m]     Change the camera
    [q\ESC] Quit
"""

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
from open3d.cpu.pybind.io import read_point_cloud
from open3d.cpu.pybind.visualization import draw_geometries

#how to install mediapipe : https://github.com/google/mediapipe/blob/master/docs/getting_started/install.md
import mediapipe as mp
import numpy as np

from mediapipe.framework.formats import landmark_pb2
import time
import csv
import os
import time
# from realsense_depth import *


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize frame count and other variables
frame_count = 0
frame_interval = 6  # Adjust as needed
output_dir = "./output"  # Adjust as needed

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def add_frame_number(image, count):
    cv2.putText(image, f"Frame: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

colorizer = rs.colorizer()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

def filter_connections(landmark_indices):
    # Create a mapping from original index to subset index
    index_map = {original_index: subset_index for subset_index, original_index in enumerate(landmark_indices)}
    
    # Filter the connections
    filtered_connections = [(index_map[start], index_map[end]) for start, end in mp_pose.POSE_CONNECTIONS if start in index_map and end in index_map]
    
    return filtered_connections

def landmark_sub(results_landmark):
    # Draw only specific landmarks: body + arms + legs
    landmarks = [results_landmark[0]] + results_landmark[11:17] + results_landmark[23:29]
    return landmark_pb2.NormalizedLandmarkList(landmark=landmarks)

# Get the indices of the landmarks in your subset
landmark_indices = [0] + list(range(11, 17)) + list(range(23, 29))

# Get the filtered connections
filtered_connections = filter_connections(landmark_indices)

# Streaming loop
frame_count = 0
frame_interval = 6  # Assuming you want to save every 6 frames
output_dir = "output_frames"  # Define your output directory here


# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 5 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

display_mode = "RGB"


def display_image_mode(mode, image, depth_colormap):
    if mode == "RGB":
        return image
    elif mode == "Depth":
        return depth_colormap

print("Starting program...")

try:
    print("Entering main loop...")
    while True:
        print("Capturing frames...")
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            print("Invalid frames detected. Skipping...")
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        print("Processing MediaPipe Pose Detection...")
        results = pose.process(color_image)
        # Convert depth_image to colormap for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
        if results.pose_landmarks:
            results_landmark = results.pose_landmarks.landmark
            landmark_subset = landmark_sub(results_landmark)
            
            # Draw landmarks on color_image
            mp_draw.draw_landmarks(color_image, landmark_subset, filtered_connections)
            
            # Draw landmarks on depth_colormap
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            mp_draw.draw_landmarks(depth_colormap, landmark_subset, filtered_connections)
            
            for landmark in landmark_subset.landmark:
                x, y = int(landmark.x * color_image.shape[1]), int(landmark.y * color_image.shape[0])
                if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
                    distance = depth_image[y, x]
                    cv2.putText(color_image, "{}cm".format(int(distance/10)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(depth_colormap, "{}cm".format(int(distance/10)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    print(f"Skipping out-of-bounds landmark at ({x}, {y})")

            if frame_count % frame_interval == 0:
                filename = f'{output_dir}/frame_{frame_count}.png'
                add_frame_number(color_image, frame_count)
                cv2.imwrite(filename, color_image)
        frame_count += 1

        displayed_image = display_image_mode(display_mode, color_image, depth_colormap)
        cv2.imshow('Align Example', displayed_image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
        if key == ord("e"):
            frames = pipeline.wait_for_frames()
            colorized = colorizer.process(frames)
            ply = rs.save_to_ply("out.ply")
            ply.set_option(rs.save_to_ply.option_ply_binary, False)
            ply.set_option(rs.save_to_ply.option_ply_normals, True)
            ply.process(colorized)
        if key == ord("v"):
            cloud = read_point_cloud("out.ply")
            draw_geometries([cloud])
        if key == ord("m"):
            display_mode = "Depth" if display_mode == "RGB" else "RGB"

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Stopping pipeline...")
    pipeline.stop()