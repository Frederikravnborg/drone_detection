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

# Optionally, define a custom label dictionary if you want user-friendly names.
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

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # ---------------------------
        # ROS Sub/Pub Setup
        # ---------------------------
        # Subscribe to the /camera/image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.listener_callback,
            10
        )

        # Publisher for the detection results (JSON-encoded string on /vision/object_spotted)
        self.publisher_ = self.create_publisher(String, '/vision/object_spotted', 10)

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # ---------------------------
        # YOLO & Torch Setup
        # ---------------------------
        # Pick device: 'mps' if available (Apple Silicon), else GPU, else CPU.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.get_logger().info("Using Apple MPS device.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.get_logger().info("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            self.get_logger().info("Using CPU device.")

        # Load YOLOv8 model (pretrained on COCO) to the chosen device
        self.model = YOLO("yolov8n.pt").to(self.device)

        self.get_logger().info("Object Detection Node started. Subscribed to /camera/image.")

    def listener_callback(self, ros_image):
        """Callback that runs each time we receive an Image on /camera/image."""
        # Convert ROS Image to OpenCV format (BGR)
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to OpenCV image: {e}")
            return

        # Retrieve the timestamp (from the image header).
        # Convert ROS2 time to float in seconds or to an int in milliseconds.
        stamp = ros_image.header.stamp.sec + ros_image.header.stamp.nanosec * 1e-9
        timestamp_ms = int(stamp * 1000)  # example in milliseconds

        # --------------------------------------------------------
        # Perform object detection using your YOLO approach
        # --------------------------------------------------------
        # Convert BGR -> RGB if desired (YOLOv8 can handle BGR, but here's the canonical approach):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        results = self.model.predict(rgb_frame, device=self.device, verbose=False)
        boxes = results[0].boxes

        # Filter boxes to keep only TARGET_CLASS_IDS
        detections = []
        for box in boxes:
            class_id = int(box.cls[0])
            if class_id in TARGET_CLASS_IDS:
                conf = float(box.conf[0])
                # Convert xyxy floats to integers for bounding boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Calculate center coordinates, width, and height
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1

                # Use YOLO's default label or a custom label
                default_label = self.model.names[class_id]
                display_label = CUSTOM_LABELS.get(class_id, default_label)

                # This detection includes bounding box corners, label, confidence, etc.
                detection_info = {
                    "class_id": class_id,
                    "label": display_label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "width": width,
                    "height": height
                }
                detections.append(detection_info)

        # --------------------------------------------------------
        # Publish detection results
        # --------------------------------------------------------
        # Prepare the data to publish with [cx, cy, width, height]
        results_to_publish = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            # Calculate center, width, and height
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1

            # Construct a detection dictionary consistent with your specs
            detection_dict = {
                "object_id": detection["class_id"],            # or some unique ID if needed
                "position": [cx, cy, width, height],
                "label": detection["label"],
                "confidence": detection["confidence"],
                "timestamp": timestamp_ms,
            }
            results_to_publish.append(detection_dict)

        # Publish the entire list of detections as a JSON string
        msg = String()
        msg.data = json.dumps({"detections": results_to_publish})
        self.publisher_.publish(msg)

        # Optional debug log
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
