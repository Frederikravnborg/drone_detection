import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time

# ================================
# Configuration Variables
# ================================

# Path to your video file
VIDEO_PATH = 'run.MP4'  # Change this to your video file path

# Maximum number of frames to process (set to None to process the entire video)
MAX_FRAMES = 100  # Set to None for no limit

# Enable verbose logging (set to True for detailed logs, False to suppress)
VERBOSE = False  # Set to True if you want detailed logs

# ================================
# Device Configuration
# ================================

# Determine the device to use: CUDA, MPS, or CPU
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
    if VERBOSE:
        print("Using MPS (Metal Performance Shaders) backend.")
else:
    DEVICE = 'cpu'
    if VERBOSE:
        print("Using CPU.")

# Initialize YOLOv8 model (YOLOv8n for nano - faster performance)
MODEL = YOLO('yolov8n.pt').to(DEVICE)  # Use 'yolov8n.pt' for faster performance

# ================================
# Define Target Classes and Colors
# ================================

# Define target classes based on your requirements
TARGET_CLASSES = [
    "person", "bicycle", "motorcycle", "car", "aeroplane",
    "bus", "boat", "stop sign", "umbrella", "sports ball",
    "baseball bat", "bed", "tennis racket", "suitcase", "skis"
]

# Define colors for classes (optional)
COLORS = {
    "person": (0, 255, 0),
    "car": (255, 0, 0),
    "motorcycle": (0, 0, 255),
    # Add more classes and their colors as needed
}

# ================================
# Video Capture Setup
# ================================

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open video file {VIDEO_PATH}")
    exit()

# Get total number of frames in the video
original_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Determine the number of frames to process
if MAX_FRAMES is not None:
    total_frames = min(original_total_frames, MAX_FRAMES)
else:
    total_frames = original_total_frames

if VERBOSE:
    print(f"Total frames in video: {original_total_frames}")
    if MAX_FRAMES is not None:
        print(f"Processing up to {total_frames} frames for testing.")

# ================================
# Create Display Window and Trackbar
# ================================

# Create a window
cv2.namedWindow('Object Recognition', cv2.WINDOW_NORMAL)

# Initialize variables
paused = False
current_frame = 0
processed_frames = 0  # Counter for processed frames

def on_trackbar(val):
    global current_frame, paused
    current_frame = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    paused = True  # Pause the video when trackbar is used
    if VERBOSE:
        print(f"Trackbar moved to frame {current_frame}.")

# Create a trackbar with the updated total_frames
if total_frames > 0:
    cv2.createTrackbar('Position', 'Object Recognition', 0, total_frames - 1, on_trackbar)
else:
    print("Warning: Total frames to process is 0.")

# ================================
# Frame Processing Loop
# ================================

while True:
    # Check if the current frame exceeds MAX_FRAMES
    if MAX_FRAMES is not None and processed_frames >= MAX_FRAMES:
        if VERBOSE:
            print(f"Reached the maximum of {MAX_FRAMES} frames. Exiting.")
        break

    if not paused:
        ret, frame = cap.read()
        if not ret:
            if VERBOSE:
                print("End of video reached or cannot fetch the frame.")
            break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos('Position', 'Object Recognition', current_frame)
        processed_frames += 1
        if VERBOSE:
            print(f"Processing frame {processed_frames}/{total_frames}")
    else:
        # When paused, seek to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            if VERBOSE:
                print("Cannot fetch the frame while paused.")
            break

    # Start time for FPS calculation
    start_time = time.time()

    # Perform inference with verbose=False to suppress model printouts
    results = MODEL(frame, verbose=False)

    # Extract detections
    detections = results[0].boxes  # Assuming batch size of 1

    # Loop through detections
    for box in detections:
        cls_id = int(box.cls)
        cls_name = MODEL.names[cls_id]
        confidence = box.conf.item()  # Correctly convert tensor to scalar

        if cls_name in TARGET_CLASSES and confidence > 0.2:
            # Properly convert tensor to list before mapping to int
            try:
                # Access the first (and only) list inside the outer list
                coords = box.xyxy.tolist()
                if isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], list):
                    x1, y1, x2, y2 = [int(coord) for coord in coords[0]]
                else:
                    raise ValueError("Invalid coordinates format.")
            except Exception as e:
                if VERBOSE:
                    print(f"Error converting box coordinates: {e}")
                continue  # Skip this box if there's an error

            # Optionally filter based on box size for scale
            frame_height, frame_width = frame.shape[:2]
            box_width = x2 - x1
            box_height = y2 - y1

            # Example condition for "Car (>1:8 Scale Model)"
            if cls_name == "car" and box_width < (frame_width / 8):
                continue  # Skip if car is smaller than 1:8 scale

            # Get color for the class
            color = COLORS.get(cls_name, (0, 255, 0))  # Default to green

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Put label
            label = f"{cls_name}: {confidence*100:.2f}%"
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.putText(frame, label, (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Put FPS on frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow('Object Recognition', frame)

    key = cv2.waitKey(1) & 0xFF  # Reduced wait time for smoother FPS

    if key == ord('q'):
        if VERBOSE:
            print("Quit key pressed. Exiting.")
        break
    elif key == ord('p'):
        # Pause or play
        paused = not paused
        if VERBOSE:
            state = "paused" if paused else "playing"
            print(f"Video {state}.")
    elif key == ord('n'):
        # Next frame
        paused = True
        current_frame += 1
        if current_frame >= total_frames:
            current_frame = total_frames - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        if VERBOSE:
            print(f"Moved to frame {current_frame}.")
    elif key == ord('b'):
        # Previous frame
        paused = True
        current_frame -= 1
        if current_frame < 0:
            current_frame = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        if VERBOSE:
            print(f"Moved back to frame {current_frame}.")

# ================================
# Cleanup
# ================================

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
if VERBOSE:
    print("Released video capture and destroyed all windows.")
