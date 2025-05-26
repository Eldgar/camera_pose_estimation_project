import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf_transformations

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        
        # Subscribe to the camera image
        self.subscription = self.create_subscription(
            Image, '/wrist_rgbd_depth_sensor/image_raw', self.image_callback, 10)
        
        # Subscribe to the known marker transform (rg2_gripper_aruco_link) published by the C++ node
        self.transform_subscription = self.create_subscription(
            TransformStamped, "/ur_transform", self.transform_callback, 10)

        # TF broadcaster to publish the computed camera pose
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Latest known marker transform (T_base_marker)
        self.latest_transform = None
        self.get_logger().info("Aruco Detector Node Initialized (computing camera pose)")

        # ArUco detection setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()

        # Camera calibration parameters (example values)
        self.marker_length = 0.045  # Marker size in meters
        self.camera_matrix = np.array([[520.78138, 0.0, 320.5],
                                       [0.0, 520.78138, 240.5],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.zeros((5, 1))

        self.processed_image_pub = self.create_publisher(Image, '/aruco_detector/image', 10)

    def transform_callback(self, msg):
        """
        Store the latest known marker transform (T_base_marker).
        This transform is from base_link to rg2_gripper_aruco_link.
        """
        self.latest_transform = msg
        self.get_logger().info("Received known marker transform (rg2_gripper_aruco_link)!")

    def image_callback(self, msg):
        """
        Process the image to detect ArUco markers and compute the camera's pose.
        """
        self.get_logger().info("Received image message!")  # Debug log

        if self.latest_transform is None:
            self.get_logger().warn("Waiting for known marker transform (rg2_gripper_aruco_link)...")
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.get_logger().info(f"Processing image... (size: {frame.shape[0]}x{frame.shape[1]})")
        

        # Detect ArUco markers
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            self.get_logger().info(f"Detected {len(ids)} markers: {ids.flatten().tolist()}")
        else:
            self.get_logger().warn("No ArUco markers detected in the image!")
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            # Process each detected marker
            for i in range(len(ids)):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]
                aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)
                self.publish_camera_pose(rvec, tvec, ids[i][0])

        cv2.imshow("Aruco Marker Detection", frame)
        cv2.waitKey(1)
        processed_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.processed_image_pub.publish(processed_msg)

    def publish_camera_pose(self, rvec, tvec, marker_id):
        """
        Compute the camera pose (T_base_camera) from:
          - T_base_marker: Known marker pose from /ur_transform (base_link -> rg2_gripper_aruco_link)
          - T_camera_marker: From ArUco detection (camera -> marker)
        
        We invert T_camera_marker to get T_marker_camera, then compose:
             T_base_camera = T_base_marker * T_marker_camera
        """
        # 1. Get T_camera_marker from ArUco detection
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        t = np.array([tvec[0], tvec[1], tvec[2]])
        
        # Build 4x4 homogeneous matrix for T_camera_marker
        T_camera_marker = np.eye(4)
        T_camera_marker[:3, :3] = R
        T_camera_marker[:3, 3] = t
        
        # Invert to get T_marker_camera (marker -> camera)
        T_marker_camera = np.linalg.inv(T_camera_marker)
        
        # 2. Get T_base_marker from the known marker transform
        base_marker = self.latest_transform.transform
        t_base_marker = np.array([base_marker.translation.x,
                                  base_marker.translation.y,
                                  base_marker.translation.z])
        q_base_marker = [base_marker.rotation.x,
                         base_marker.rotation.y,
                         base_marker.rotation.z,
                         base_marker.rotation.w]
        # Create homogeneous matrix from quaternion
        T_base_marker = tf_transformations.quaternion_matrix(q_base_marker)
        T_base_marker[:3, 3] = t_base_marker
        
        # 3. Compose to get T_base_camera = T_base_marker * T_marker_camera
        T_base_camera = np.dot(T_base_marker, T_marker_camera)
        
        # Extract translation and rotation (quaternion) from T_base_camera
        t_base_camera = T_base_camera[:3, 3]
        quat_base_camera = tf_transformations.quaternion_from_matrix(T_base_camera)
        
        # Prepare and publish the transform message
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        # Parent frame is base_link
        transform.header.frame_id = "base_link"
        # Child frame is camera_link (computed camera pose)
        transform.child_frame_id = "camera_link"
        transform.transform.translation.x = t_base_camera[0]
        transform.transform.translation.y = t_base_camera[1]
        transform.transform.translation.z = t_base_camera[2]
        transform.transform.rotation.x = quat_base_camera[0]
        transform.transform.rotation.y = quat_base_camera[1]
        transform.transform.rotation.z = quat_base_camera[2]
        transform.transform.rotation.w = quat_base_camera[3]

        self.tf_broadcaster.sendTransform(transform)
        self.get_logger().info(
            f"Published camera_link transform based on marker {marker_id}: "
            f"translation=({t_base_camera[0]:.3f}, {t_base_camera[1]:.3f}, {t_base_camera[2]:.3f})"
        )

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

