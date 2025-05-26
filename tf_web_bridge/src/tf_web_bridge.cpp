/********************************************************************
 * tf2_web_republisher.cpp
 *
 * Listens to /tf and republishes every transform on
 * /tf2_web_republisher/feedback so a web front-end (e.g. ros2-web-bridge
 * TFClient) can consume it.
 ********************************************************************/

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_msgs/msg/tf_message.hpp>
#include <map>
#include <string>

using namespace std::chrono_literals;

class TFWebBridge : public rclcpp::Node
{
public:
  TFWebBridge()
  : Node("tf2_web_republisher"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // subscribe to /tf
    tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf", 10,
        std::bind(&TFWebBridge::tf_cb, this, std::placeholders::_1));

    // publish to web-friendly topic
    web_pub_ = create_publisher<tf2_msgs::msg::TFMessage>(
        "/tf2_web_republisher/feedback", 10);

    // timer to republish at 10 Hz
    timer_ = create_wall_timer(
        100ms, std::bind(&TFWebBridge::republish, this));

    RCLCPP_INFO(get_logger(), "✅ TF2 Web Republisher started");
  }

private:
  /* store the last transform for each child_frame_id */
  void tf_cb(const tf2_msgs::msg::TFMessage::SharedPtr msg)
  {
    for (const auto & tr : msg->transforms)
      cache_[tr.child_frame_id] = tr;
  }

  void republish()
  {
    if (cache_.empty()) return;

    tf2_msgs::msg::TFMessage out;
    for (const auto & kv : cache_)
      out.transforms.push_back(kv.second);

    web_pub_->publish(out);
  }

  /* members */
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr    web_pub_;
  tf2_ros::Buffer                                           tf_buffer_;
  tf2_ros::TransformListener                                tf_listener_;
  rclcpp::TimerBase::SharedPtr                              timer_;
  std::map<std::string, geometry_msgs::msg::TransformStamped> cache_;
};

/* --------------------------- main ---------------------------- */
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TFWebBridge>());
  rclcpp::shutdown();
  return 0;
}

