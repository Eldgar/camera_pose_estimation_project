#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <opencv2/opencv.hpp>

class CupholderDetector : public rclcpp::Node
{
public:
    CupholderDetector()
    : Node("cupholder_detector")
    {
        RCLCPP_INFO(this->get_logger(), "Cupholder Detector Node Initialized");

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/wrist_rgbd_depth_sensor/points", 10,
            std::bind(&CupholderDetector::pointcloud_callback, this, std::placeholders::_1));

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert PointCloud2 to PCL format
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // Create a grayscale depth image from the point cloud
        // We'll assume a frontal view and project X, Y into image plane and use Z for brightness

        int width = 640;
        int height = 480;
        cv::Mat depth_image = cv::Mat::zeros(height, width, CV_8UC1);

        for (const auto& point : cloud->points)
        {
            if (!std::isfinite(point.z) || point.z <= 0.1 || point.z > 2.0) continue;

            // Map point.x and point.y to image coordinates (you'll likely need calibration here)
            int u = static_cast<int>((point.x + 0.3) * 640);  // adjust scale/offset as needed
            int v = static_cast<int>((-point.y + 0.2) * 480); // flip y-axis if needed

            if (u >= 0 && u < width && v >= 0 && v < height)
            {
                uint8_t depth = static_cast<uint8_t>(255.0 * (1.0 - std::min(point.z / 2.0, 1.0)));
                depth_image.at<uchar>(v, u) = depth;
            }
        }

        // Detect circles
        std::vector<cv::Vec3f> circles;
        cv::GaussianBlur(depth_image, depth_image, cv::Size(9, 9), 2, 2);
        cv::HoughCircles(depth_image, circles, cv::HOUGH_GRADIENT, 1, 40, 100, 20, 5, 50);

        for (size_t i = 0; i < circles.size(); ++i)
        {
            float u = circles[i][0];
            float v = circles[i][1];

            // Reverse projection from image to approximate point cloud
            float x = (u / 640.0f) - 0.3f;
            float y = -((v / 480.0f) - 0.2f);

            // Find nearest Z in region
            float z = 0.5f; // fallback
            for (const auto& pt : cloud->points)
            {
                if (std::abs(pt.x - x) < 0.02 && std::abs(pt.y - y) < 0.02)
                {
                    z = pt.z;
                    break;
                }
            }

            RCLCPP_INFO(this->get_logger(), "Detected cupholder at [%.2f, %.2f, %.2f]", x, y, z);

            geometry_msgs::msg::TransformStamped tf;
            tf.header.stamp = this->get_clock()->now();
            tf.header.frame_id = msg->header.frame_id;
            tf.child_frame_id = "cupholder_" + std::to_string(i);

            tf.transform.translation.x = x;
            tf.transform.translation.y = y;
            tf.transform.translation.z = z;

            tf.transform.rotation.w = 1.0;
            tf.transform.rotation.x = 0.0;
            tf.transform.rotation.y = 0.0;
            tf.transform.rotation.z = 0.0;

            tf_broadcaster_->sendTransform(tf);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CupholderDetector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
