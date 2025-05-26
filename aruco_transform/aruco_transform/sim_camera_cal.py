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
import tf2_ros
from tf2_ros import TransformException
import time
from scipy.spatial.transform import Rotation as R

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

        # Marker size in meters
        self.marker_length = 0.045

        # Adjusted camera calibration (2x resolution)
        # self.camera_matrix = np.array([[1041.56276, 0.0, 641.0],
        #                                [0.0, 1041.56276, 481.0],
        #                                [0.0, 0.0, 1.0]])
        self.camera_matrix = np.array([[1040.0, 0.0, 640.0],
                                       [0.0, 1040.0, 480.0],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.zeros((5, 1))

        self.processed_image_pub = self.create_publisher(Image, '/aruco_detector/image', 10)

    def transform_callback(self, msg):
        self.latest_transform = msg
        self.get_logger().info("Received known marker transform (rg2_gripper_aruco_link)!")

    def image_callback(self, msg):
        self.get_logger().info("Received image message!")

        if self.latest_transform is None:
            self.get_logger().warn("Waiting for known marker transform (rg2_gripper_aruco_link)...")
            return

        # Convert and resize image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
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
            
            for i in range(len(ids)):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # Draw axes
                aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

                self.publish_camera_pose(rvec, tvec, ids[i][0])

        # Show image
        cv2.imshow("Aruco Marker Detection", frame)
        cv2.waitKey(1)

        # Publish processed image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_msg = self.bridge.cv2_to_imgmsg(rgb_frame, encoding="rgb8")
        processed_msg.header.stamp = msg.header.stamp
        self.processed_image_pub.publish(processed_msg)
    def compare_transforms(self):
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer, self)

        # Give it time to populate
        rclpy.spin_once(self, timeout_sec=0.2)

        try:
            # Estimated camera pose
            t_est = tf_buffer.lookup_transform(
                'base_link', 'camera_link', rclpy.time.Time())

            # Ground truth target camera frame (choose one)
            t_true = tf_buffer.lookup_transform(
                'base_link', 'wrist_rgbd_camera_depth_optical_frame', rclpy.time.Time())

            # Convert to matrices
            T_est = self.transform_to_matrix(t_est.transform)
            T_true = self.transform_to_matrix(t_true.transform)

            # Compute error transform
            T_err = np.linalg.inv(T_true) @ T_est
            trans_err = np.linalg.norm(T_err[:3, 3])

            rot_err = R.from_matrix(T_err[:3, :3]).magnitude()

            self.get_logger().info(
                f"[TF ERROR] Translation error: {trans_err:.4f} m | Rotation error: {np.degrees(rot_err):.2f} deg")

        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}")

    def transform_to_matrix(self, transform):
        t = np.array([transform.translation.x,
                    transform.translation.y,
                    transform.translation.z])
        q = np.array([transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w])
        T = tf_transformations.quaternion_matrix(q)
        T[:3, 3] = t
        return T


    def publish_camera_pose(self, rvec, tvec, marker_id):
        R, _ = cv2.Rodrigues(rvec)
        t = np.array([tvec[0], tvec[1], tvec[2]])

        T_camera_marker = np.eye(4)
        T_camera_marker[:3, :3] = R
        T_camera_marker[:3, 3] = t

        T_marker_camera = np.linalg.inv(T_camera_marker)

        base_marker = self.latest_transform.transform
        t_base_marker = np.array([base_marker.translation.x,
                                  base_marker.translation.y,
                                  base_marker.translation.z])
        q_base_marker = [base_marker.rotation.x,
                         base_marker.rotation.y,
                         base_marker.rotation.z,
                         base_marker.rotation.w]

        T_base_marker = tf_transformations.quaternion_matrix(q_base_marker)
        T_base_marker[:3, 3] = t_base_marker

        T_base_camera = np.dot(T_base_marker, T_marker_camera)

        t_base_camera = T_base_camera[:3, 3]
        quat_base_camera = tf_transformations.quaternion_from_matrix(T_base_camera)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "camera_link"
        transform.transform.translation.x = t_base_camera[0]
        transform.transform.translation.y = t_base_camera[1]
        transform.transform.translation.z = t_base_camera[2]
        transform.transform.rotation.x = quat_base_camera[0]
        transform.transform.rotation.y = quat_base_camera[1]
        transform.transform.rotation.z = quat_base_camera[2]
        transform.transform.rotation.w = quat_base_camera[3]

        self.tf_broadcaster.sendTransform(transform)
        self.compare_transforms()
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
