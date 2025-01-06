import cv2
import numpy as np

# Paths to the model files
prototxt = "/Users/fredmac/Documents/DTU-FredMac/5Drone/models/MobileNetSSD_deploy.prototxt.txt"
model = "/Users/fredmac/Documents/DTU-FredMac/5Drone/models/MobileNetSSD_deploy.caffemodel"

# List of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Generate random colors for each class label
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Open the video file
cap = cv2.VideoCapture('/Users/fredmac/Documents/DTU-FredMac/5Drone/run1.MP4')

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a window
cv2.namedWindow('Object Recognition', cv2.WINDOW_NORMAL)

# Initialize variables
paused = False
current_frame = 0

def on_trackbar(val):
    global current_frame, paused
    current_frame = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    paused = True  # Pause the video when trackbar is used

# Create a trackbar
cv2.createTrackbar('Position', 'Object Recognition', 0, total_frames - 1, on_trackbar)

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos('Position', 'Object Recognition', current_frame)
    else:
        # When paused, seek to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

    # Get frame dimensions
    (h, w) = frame.shape[:2]

    # Preprocess the frame: resize and normalize
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # Set the input blob for the neural network
    net.setInput(blob)

    # Perform forward pass (object detection)
    detections = net.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # Extract the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a threshold
        if confidence > 0.2:
            # Extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])

            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # Draw the prediction on the frame
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx], 2)

    # Display the output frame
    cv2.imshow('Object Recognition', frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('p'):
        # Pause or play
        paused = not paused
    elif key == ord('n'):
        # Next frame
        paused = True
        current_frame += 1
        if current_frame >= total_frames:
            current_frame = total_frames - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    elif key == ord('b'):
        # Previous frame
        paused = True
        current_frame -= 1
        if current_frame < 0:
            current_frame = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()