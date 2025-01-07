import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Create a publisher that publishes sensor_msgs/Image on the /camera/image topic.
        self.publisher_ = self.create_publisher(Image, '/camera/image', 10)

        # Create a timer to periodically read frames from the camera.
        # Here we set 0.1 seconds => 10 Hz frame rate. Adjust as needed.
        timer_period = 0.1  
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize the camera (assuming device index 0).
        self.cap = cv2.VideoCapture(0)  
        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera.")
            # Optionally raise an exception or handle error

        # Initialize CvBridge
        self.bridge = CvBridge()
        self.get_logger().info("Camera node started, publishing to /camera/image...")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to read frame from camera.")
            return

        # Convert OpenCV image (BGR) to ROS Image message
        # encoding can be 'bgr8' (typical for OpenCV’s default BGR) or 'rgb8'
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        # Attach the current ROS2 time to the header
        ros_image.header.stamp = self.get_clock().now().to_msg()

        # Publish the image
        self.publisher_.publish(ros_image)

    def destroy_node(self):
        # Close camera when node is destroyed
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        camera_node.get_logger().info("Camera node stopped by user.")
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
