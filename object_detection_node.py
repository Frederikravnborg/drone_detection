#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import torch
from ultralytics import YOLO
import json

# Define the set of COCO class IDs that approximate the desired categories
TARGET_CLASS_IDS = {
    0,   # person (for person/mannequin)
    2,   # car
    3,   # motorcycle
    4,   # airplane
    5,   # bus
    8,   # boat
    11,  # stop sign
    25,  # umbrella
    28,  # suitcase
    30,  # skis
    31,  # snowboard
    32,  # sports ball
    34,  # baseball bat
    38,  # tennis racket
    59,  # bed (for mattress)
}

CUSTOM_LABELS = {
    0:  "Person / Mannequin",
    2:  "Car (>1:8 Scale Model)",
    3:  "Motorcycle (>1:8 Scale Model)",
    4:  "Airplane (>3m Wing Span Scale Model)",
    5:  "Bus (>1:8 Scale Model)",
    8:  "Boat (>1:8 Scale Model)",
    11: "Stop Sign (Flat, Upwards Facing)",
    25: "Umbrella",
    28: "Suitcase",
    30: "Skis",
    31: "Snowboard",
    32: "Sports Ball (Regulation Size)",
    34: "Baseball Bat",
    38: "Tennis Racket",
    59: "Bed / Mattress (> Twin Size)",
}

# Tracking hyperparameters
MAX_MISSES = 5
ALPHA = 0.7

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # ROS Sub/Pub Setup
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.listener_callback,
            10
        )
        self.publisher_ = self.create_publisher(String, '/vision/object_spotted', 10)
        self.bridge = CvBridge()

        # Device Selection
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.get_logger().info("Using Apple MPS device.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.get_logger().info("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            self.get_logger().info("Using CPU device.")

        # Load YOLOv8 model
        # Use a larger model as in Script 2, if desired; here we continue using "yolov8n.pt" for consistency
        self.model = YOLO("yolov8n.pt").to(self.device)

        # Initialize track history
        self.track_history = {}

        self.get_logger().info("Object Detection Node started. Subscribed to /camera/image.")

    def listener_callback(self, ros_image):
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to OpenCV image: {e}")
            return

        stamp = ros_image.header.stamp.sec + ros_image.header.stamp.nanosec * 1e-9
        timestamp_ms = int(stamp * 1000)

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use the tracking API on the single frame
        # Note: We wrap the frame in a list to treat it as a source sequence of one frame.
        tracking_results = self.model.track(
            source=[rgb_frame],
            device=self.device,
            conf=0.01,     # Lower confidence threshold as in Script 2
            iou=0.50,      # Slightly higher IoU threshold
            tracker="config/bytetrack.yaml",
            stream=True,
            show=False
        )

        # Process tracking results for the current frame
        current_detections = []
        for result in tracking_results:
            # Check if detections exist for this frame
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id not in TARGET_CLASS_IDS:
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # Retrieve track_id from the tracker; if not available, use -1 as a placeholder
                    track_id = int(box.id[0]) if box.id is not None else -1

                    default_label = self.model.names.get(class_id, f"class_{class_id}")
                    display_label = CUSTOM_LABELS.get(class_id, default_label)

                    # Apply smoothing if this track has history
                    if track_id in self.track_history:
                        old_x1, old_y1, old_x2, old_y2 = self.track_history[track_id]["bbox"]
                        x1 = int(ALPHA * x1 + (1 - ALPHA) * old_x1)
                        y1 = int(ALPHA * y1 + (1 - ALPHA) * old_y1)
                        x2 = int(ALPHA * x2 + (1 - ALPHA) * old_x2)
                        y2 = int(ALPHA * y2 + (1 - ALPHA) * old_y2)

                    # Update track history
                    self.track_history[track_id] = {
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "label": display_label,
                        "miss_count": 0
                    }
                    current_detections.append(track_id)

        # Increase miss_count for tracks not detected in this frame
        for t_id in list(self.track_history.keys()):
            if t_id not in current_detections:
                self.track_history[t_id]["miss_count"] += 1
                if self.track_history[t_id]["miss_count"] > MAX_MISSES:
                    del self.track_history[t_id]

        # Prepare detections for publishing
        results_to_publish = []
        for track_id, info in self.track_history.items():
            x1, y1, x2, y2 = info["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1

            detection_dict = {
                "object_id": info.get("label"),  # Using label instead of class_id for more informative output
                "position": [cx, cy, width, height],
                "confidence": info["conf"],
                "timestamp": timestamp_ms,
                "track_id": track_id
            }
            results_to_publish.append(detection_dict)

        msg = String()
        msg.data = json.dumps({"detections": results_to_publish})
        self.publisher_.publish(msg)

        self.get_logger().info(f"Published {len(results_to_publish)} detections at t={timestamp_ms} ms.")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Object Detection node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
