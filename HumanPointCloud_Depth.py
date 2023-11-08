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
import datetime

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize frame count and other variables
frame_count = 0
frame_interval = 6  # Adjust as needed
output_dir = './output'  # Make sure this directory exists
os.makedirs(output_dir, exist_ok=True)  # This will create the directory if it doesn't exist

# Define directories for output
base_dir = 'coordinates'
csv_dir = os.path.join(base_dir, 'csv')
image_dir = os.path.join(base_dir, 'images')

# Create directories if they do not exist
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)


def landmark_sub(results_landmark):
    # Draw only specific landmarks: body + arms + legs
    landmarks = [results_landmark[0]] + results_landmark[11:17] + results_landmark[23:29]
    return landmark_pb2.NormalizedLandmarkList(landmark=landmarks)

# Define a function to add overlay information to the image
def add_overlay_info(image, count, x, y, z):
    cv2.putText(image, f"Frame: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Head X: {x:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Head Y: {y:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Head Z: {z:.2f}m", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Define a function to display the image based on the mode
def display_image_mode(mode, image, depth_colormap):
    if mode == "RGB":
        return image
    elif mode == "Depth":
        return depth_colormap

def filter_connections(landmark_indices):
    # Create a mapping from original index to subset index
    index_map = {original_index: subset_index for subset_index, original_index in enumerate(landmark_indices)}
    # Filter the connections
    filtered_connections = [(index_map[start], index_map[end]) for start, end in mp_pose.POSE_CONNECTIONS if start in index_map and end in index_map]
    return filtered_connections

def save_image(image, base_path, count, head_x=None, head_y=None, head_z=None, suffix=''):
    # If head coordinates are provided, add overlay info to the image
    if head_x is not None and head_y is not None and head_z is not None:
        add_overlay_info(image, count, head_x, head_y, head_z)
    # Construct the filename using the base path, count, and an optional suffix
    filename = f"{base_path}_frame_{count:04d}{suffix}.png"
    # Save the image
    cv2.imwrite(filename, image)



# Get the current date and time for file naming
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Define the CSV and image file paths with the current date and time
csv_filename = os.path.join(csv_dir, f'body_coordinates_{current_time}.csv')
image_file_base = os.path.join(image_dir, f'image_{current_time}')

# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()
colorizer = rs.colorizer()

# Configure the pipeline to stream
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# Initialize video writer
color_frame = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
frame_width = color_frame.width
frame_height = color_frame.height
fps = 30  # The RealSense camera's configured FPS

# Define video writer with mp4 format
video_filename = os.path.join(output_dir, f'video_{current_time}.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4 format
video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

# Check if video writer is opened
if not video_writer.isOpened():
    raise Exception("Video writer could not be opened.")

# Define display mode
display_mode = "Depth"  # Can be "RGB" or "Depth"

# Before the main loop, initialize the CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header
    csv_writer.writerow(['frame', 'part', 'x', 'y', 'z'])

# Define an align object
align_to = rs.stream.color
align = rs.align(align_to)

# Get the indices of the landmarks in your subset
landmark_indices = [0] + list(range(11, 17)) + list(range(23, 29))

# Get the filtered connections
filtered_connections = filter_connections(landmark_indices)


print("Program will start after a 2-second delay.")
time.sleep(2)

print("Starting program...")

try:
    print("Entering main loop...")
    while True:
        #Capture Frame
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            print("Invalid frames detected. Skipping...")
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # print("Processing MediaPipe Pose Detection...")
        results = pose.process(color_image)
        # Convert depth_image to colormap for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)



        if results.pose_landmarks:
            results_landmark = results.pose_landmarks.landmark
            landmark_subset = landmark_sub(results_landmark)
            # Draw landmarks on the displayed image
            mp_draw.draw_landmarks(color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw landmarks on depth_colormap
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            mp_draw.draw_landmarks(depth_colormap, landmark_subset, filtered_connections)

            # Get the coordinates of the first landmark (head)
            head_landmark = results.pose_landmarks.landmark[0]
            head_x = head_landmark.x * frame_width
            head_y = head_landmark.y * frame_height
            head_z = depth_image[int(head_y), int(head_x)] * depth_scale
            add_overlay_info(color_image, frame_count, head_x, head_y, head_z)


            for landmark in landmark_subset.landmark:
                x, y = int(landmark.x * color_image.shape[1]), int(landmark.y * color_image.shape[0])
                # Ensure x is within the width bounds and y is within the height bounds
                if 0 <= x < depth_image.shape[1] - 1 and 0 <= y < depth_image.shape[0] - 1:
                    distance = depth_image[y, x]
                    # cv2.putText(color_image, "{}cm".format(int(distance/10)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(depth_colormap, "{}cm".format(int(distance/10)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    print(f"Skipping out-of-bounds landmark at ({x}, {y})")

            # Save the image color image and depth colormap
            # save_image(color_image, image_file_base, frame_count)
            save_image(depth_colormap, image_file_base, frame_count, '_depth')
        # Check if it's the right frame interval to save the image
        if frame_count % frame_interval == 0:
            # Save the color image with overlay info
            # save_image(color_image, image_file_base, frame_count, head_x, head_y, head_z)
            # Save the depth image with overlay info
            save_image(depth_colormap, image_file_base, frame_count, suffix='_depth')
        frame_count += 1

        # Choose which image to write to the video file based on the displayed image
        displayed_image = display_image_mode(display_mode, color_image, depth_colormap)
        video_writer.write(displayed_image)

        cv2.imshow('Align Example', displayed_image)

        # Handle key events
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
        elif key == ord("m"):
            display_mode = "Depth" if display_mode == "RGB" else "RGB"

        frame_count += 1

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Releasing resources...")
    video_writer.release()
    pipeline.stop()
    cv2.destroyAllWindows()
