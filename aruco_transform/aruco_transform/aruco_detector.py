import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        
        # Subscribe to the camera image
        self.subscription = self.create_subscription(
            CompressedImage, '/D415/color/image_raw/compressed', self.image_callback, 10)
        
        self.get_logger().info("Aruco Detector Node Initialized")

        # ArUco detection setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()

        # Camera calibration parameters (example values)
        self.marker_length = 0.045  # Marker size in meters
        self.camera_matrix = np.array([[306.805847, 0.0, 214.441849],
                                       [0.0, 306.642456, 124.910301],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])

    def image_callback(self, msg):
        """Process the image to detect ArUco markers."""
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            self.get_logger().info(f"Detected {len(ids)} markers: {ids.flatten().tolist()}")
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Estimate pose for each detected marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            for i in range(len(ids)):
                self.get_logger().info(
                    f"Marker {ids[i][0]}: Position (x={tvecs[i][0][0]:.3f}, y={tvecs[i][0][1]:.3f}, z={tvecs[i][0][2]:.3f})")
        else:
            self.get_logger().warn("No ArUco markers detected!")
        
        cv2.imshow("Aruco Marker Detection", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()






