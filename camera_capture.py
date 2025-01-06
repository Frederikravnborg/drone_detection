import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import base64

class CameraModule:
    def __init__(self, frame_rate=30, encode_format="base64"):
        """
        Initializes the camera module.
        :param frame_rate: Frame rate at which the images are captured and published.
        :param encode_format: The encoding format for images ('raw' or 'base64').
        """
        self.cap = cv2.VideoCapture(0)  # Open the default camera
        if not self.cap.isOpened():
            rospy.logerr("Failed to open camera")
            raise RuntimeError("Could not access the camera.")
        
        self.frame_rate = frame_rate
        self.encode_format = encode_format
        self.publisher = rospy.Publisher('/camera/image', String if encode_format == "base64" else Image, queue_size=10)
        self.bridge = CvBridge()
        rospy.init_node('camera_module', anonymous=True)
        self.rate = rospy.Rate(self.frame_rate)

    def capture_and_publish(self):
        """Captures images from the webcam and publishes them to the ROS topic."""
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Failed to capture frame. Skipping...")
                continue
            
            if self.encode_format == "base64":
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                self.publisher.publish(image_base64)
            elif self.encode_format == "raw":
                image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.publisher.publish(image_message)
            else:
                rospy.logerr(f"Unsupported encode format: {self.encode_format}")
            
            self.rate.sleep()

    def release_camera(self):
        """Releases the camera resource."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        camera_module = CameraModule(frame_rate=30, encode_format="base64")  # Change encode_format to "raw" if needed
        camera_module.capture_and_publish()
    except rospy.ROSInterruptException:
        pass
    except RuntimeError as e:
        rospy.logerr(str(e))
    finally:
        camera_module.release_camera()
