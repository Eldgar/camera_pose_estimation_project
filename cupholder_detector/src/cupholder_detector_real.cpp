#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <deque>
#include <algorithm>

class CupholderDetector : public rclcpp::Node
{
public:
  CupholderDetector()
  : Node("cupholder_detector")
  {
    RCLCPP_INFO(get_logger(), "CupholderDetector node started");

    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/D415/aligned_depth_to_color/image_raw", 10,
        std::bind(&CupholderDetector::depthCallback, this, std::placeholders::_1));

    color_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        "/D415/color/image_raw/compressed", rclcpp::SensorDataQoS(),
        std::bind(&CupholderDetector::colorCallback, this, std::placeholders::_1));

    debug_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "cupholder_detection", 1);
    depth_dbg_pub_  = create_publisher<sensor_msgs::msg::Image>(
        "cupholder_depth_dbg", 1);
    roi_eq_pub_     = create_publisher<sensor_msgs::msg::Image>(
        "cupholder_roi_eq", 1);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer_      = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  }

private:
  /* ─────────── camera intrinsics ─────────── */
  const float fx_ = 306.806f, fy_ = 306.806f;
  const float cx_d_ = 214.4f, cy_d_ = 124.9f;

  /* ─────────── runtime state ─────────── */
  float roi_u_ = -1.f, roi_v_ = -1.f, roi_radius_ = -1.f;
  cv::Mat last_depth_raw_;

  bool cupholders_initialized_ = false;
  std::vector<Eigen::Vector3f> cached_cupholder_positions_;
  static constexpr std::size_t MAX_HISTORY = 5;
  std::vector<std::deque<Eigen::Vector3f>> cupholder_history_{4};

  /* stop detection after a certain number of successful frames */
  int  successful_detections_ = 0;
  static constexpr int DETECTION_LIMIT = 40;
  bool detection_stopped_ = false;

  float barista_center_height_ = -1.f;  // height in camera frame
  Eigen::Vector3f barista_center_cam_{0,0,-1};

  /* ─────────── ROS I/O ─────────── */
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr         depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr color_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr            debug_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr            depth_dbg_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr            roi_eq_pub_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::Buffer>               tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener>    tf_listener_;

  /* ───────────────────────────────────────────── */

  /* depthCallback ── just finds the big ROI circle */
  void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat depth16;
    try {
      depth16 = cv_bridge::toCvCopy(
          msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
      return;
    }
    if (depth16.empty()) return;

    cv::Mat d8, norm;
    cv::normalize(depth16, norm, 0, 255, cv::NORM_MINMAX);
    norm.convertTo(d8, CV_8UC1);
    cv::equalizeHist(d8, d8);
    cv::GaussianBlur(d8, d8, {3,3}, 0);

    /* ───────── iterate over threshold values to find ROI circle ───────── */
    std::vector<cv::Vec3f> best_circles;
    int best_threshold = -1;

    for (int thr = 50; thr <= 120; thr += 2) {
      cv::Mat thresh;
      cv::threshold(d8, thresh, thr, 255, cv::THRESH_BINARY);

      std::vector<cv::Vec3f> circles;
      cv::HoughCircles(thresh, circles, cv::HOUGH_GRADIENT,
                       1,   // dp
                       50,  // minDist
                       24,  // param1 (Canny high t)
                       12,  // param2 (centre threshold)
                       53,  // minRadius
                       74); // maxRadius

    //   RCLCPP_DEBUG(get_logger(), "thr %d → %zu circles", thr, circles.size());

      if (!circles.empty()) {
        best_circles = circles;
        best_threshold = thr;
        break;                      // stop as soon as we have a hit
      }

      // otherwise keep the threshold that got closest (largest count)
      if (circles.size() > best_circles.size()) {
        best_circles = circles;
        best_threshold = thr;
      }
    }

    if (!best_circles.empty()) {
      roi_u_      = best_circles[0][0];
      roi_v_      = best_circles[0][1];
      roi_radius_ = best_circles[0][2];
    }

    // RCLCPP_INFO(get_logger(), "Depth ROI: thr=%d  circles=%zu  (u=%.1f v=%.1f r=%.1f)",
    //             best_threshold, best_circles.size(), roi_u_, roi_v_, roi_radius_);

    // store depth image for use in color callback
    last_depth_raw_ = depth16.clone();

    /* ─── Visualisation of the detected ROI circle ─── */
    cv::Mat depth_vis; cv::cvtColor(d8, depth_vis, cv::COLOR_GRAY2BGR);
    if (!best_circles.empty()) {
      cv::Point center(cvRound(roi_u_), cvRound(roi_v_));
      int radius = cvRound(roi_radius_);
      cv::circle(depth_vis, center, 3, {0,0,255}, -1);          // red center dot
      cv::circle(depth_vis, center, radius, {0,255,0}, 2);      // green circle
    }

    // show in local window for quick troubleshooting
    // cv::imshow("Depth ROI", depth_vis);
    // Also show the binary mask that yielded the best detection (recreate)
    if (best_threshold >= 0) {
      cv::Mat best_mask;
      cv::threshold(d8, best_mask, best_threshold, 255, cv::THRESH_BINARY);
    //   cv::imshow("Depth mask", best_mask);
    }
    cv::waitKey(1);

    // publish the annotated image
    publishMat(depth_vis, depth_dbg_pub_);
  }

  /* colorCallback ── finds the four rims, publishes TFs */
  void colorCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    if (roi_u_ < 0 || last_depth_raw_.empty()) return;

    cv::Mat color_bgr = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (color_bgr.empty()) return;

    cv::Mat color_dbg = color_bgr.clone();
    cv::Mat depth8;  cv::normalize(last_depth_raw_, depth8, 0,255, cv::NORM_MINMAX, CV_8U);
    cv::Mat depth_dbg; cv::cvtColor(depth8, depth_dbg, cv::COLOR_GRAY2BGR);

    RCLCPP_DEBUG(get_logger(), "Color frame received; ROI valid=%d history=%zu", (roi_u_>=0), cupholder_history_[0].size());

    /* crop ROI */
    const int BUF = 5;
    int x = std::max(0,  int(roi_u_ - roi_radius_ - BUF));
    int y = std::max(0,  int(roi_v_ - roi_radius_ - BUF));
    int w = std::min(color_bgr.cols - x, int(2*roi_radius_ + 2*BUF));
    int h = std::min(color_bgr.rows - y, int(2*roi_radius_ + 2*BUF));
    if (w<=0 || h<=0) return;

    cv::Mat gray; cv::cvtColor(color_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat roi_eq; cv::equalizeHist(gray(cv::Rect(x,y,w,h)), roi_eq);

    // Show the cropped and equalized ROI
    // cv::imshow("Cropped ROI", roi_eq);
    cv::waitKey(1);

    // Apply gamma correction
    cv::Mat roi_gamma;
    double gamma = 0.5;  // Adjust this value: <1 makes image brighter, >1 makes it darker
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for(int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    cv::LUT(roi_eq, lookUpTable, roi_gamma);

    // Show the gamma-corrected image
    // cv::imshow("Gamma Corrected", roi_gamma);
    cv::waitKey(1);

    cv::Mat roi_bgr; cv::cvtColor(roi_gamma, roi_bgr, cv::COLOR_GRAY2BGR);
    publishMat(roi_bgr, roi_eq_pub_);

    // Try different threshold values until we find exactly 4 circles
    std::vector<cv::Vec3f> best_circles;
    int best_threshold = -1;
    
    for (int threshold = 40; threshold <= 170; threshold += 5) {
        // Create a mask for darker regions
        cv::Mat mask;
        cv::threshold(roi_gamma, mask, threshold, 255, cv::THRESH_BINARY_INV);
        
        // Apply morphological operations to clean up the mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        // Try to find circles
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT,
                         1,  // dp: Inverse ratio of accumulator resolution
                         32, // minDist: Minimum distance between centers
                         32, // param1: Upper threshold for Canny edge detector
                         12,  // param2: Threshold for center detection
                         8,  // minRadius: Minimum circle radius
                         15  // maxRadius: Maximum circle radius
                        );

        // RCLCPP_INFO(get_logger(), "Threshold %d: Found %zu circles", threshold, circles.size());

        // If we found exactly 4 circles, we're done
        if (circles.size() == 4) {
            best_circles = circles;
            best_threshold = threshold;
            break;
        }
        // Otherwise, keep track of the best result (closest to 4 circles)
        else if (best_circles.empty() || 
                std::abs(static_cast<int>(circles.size()) - 4) < 
                std::abs(static_cast<int>(best_circles.size()) - 4)) {
            best_circles = circles;
            best_threshold = threshold;
        }
    }

    // RCLCPP_INFO(get_logger(), "Using threshold %d, found %zu circles", best_threshold, best_circles.size());

    // Visualize the detected circles
    cv::Mat roi_vis = roi_bgr.clone();
    for (const auto &c : best_circles) {
        cv::Point center(c[0], c[1]);
        int radius = c[2];
        cv::circle(roi_vis, center, 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(roi_vis, center, radius, cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("Detected Circles", roi_vis);
    cv::waitKey(1);

    /* small rims */
    std::vector<Eigen::Vector3f> new_positions;
    if (!best_circles.empty())
    {
        for (const auto &c: best_circles)
        {
            float u = x + c[0];
            float v = y + c[1];

            cv::circle(color_dbg, {int(u), int(v)}, 3, {0,0,255}, -1);
            cv::circle(depth_dbg, {int(u), int(v)}, 3, {0,255,0}, -1);

            if (u<0||v<0||u>=last_depth_raw_.cols||v>=last_depth_raw_.rows) continue;
            uint16_t d = last_depth_raw_.at<uint16_t>(int(v), int(u));
            if (d==0) continue;

            float z = d*0.001f;
            float x_m = (u - cx_d_) * z / fx_;
            float y_m = (v - cy_d_) * z / fy_;
            new_positions.emplace_back(x_m, y_m, z);
        }
    }

    /* publish barista_center */
    if (best_circles.size() == 4)
    {
      // compute average pixel coordinates of the detected circles
      double sum_u = 0.0, sum_v = 0.0;
      for (const auto &c : best_circles) {
        double u = x + c[0];
        double v = y + c[1];
        sum_u += u;
        sum_v += v;
      }
      double u_avg = sum_u / 4.0;
      double v_avg = sum_v / 4.0;

      // get depth at averaged pixel
      uint16_t d = last_depth_raw_.at<uint16_t>(int(v_avg), int(u_avg));
      if (d > 0) {
        float z = d * 0.001f;
        float x_m = (u_avg - cx_d_) * z / fx_;
        float y_m = (v_avg - cy_d_) * z / fy_;
        Eigen::Vector3f centre(x_m, y_m, z);
        barista_center_cam_   = centre;
        barista_center_height_ = z;  // still store camera-frame depth for reference

        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp    = now();
        tf.header.frame_id = "D415_color_optical_frame";
        tf.child_frame_id  = "barista_center";
        tf.transform.translation.x = centre.x();
        tf.transform.translation.y = centre.y();
        tf.transform.translation.z = centre.z();
        tf.transform.rotation.w    = 1.0;
        tf_broadcaster_->sendTransform(tf);
      }
    }

    /* history / TFs */
    if (new_positions.size() == 4) updateHistory(new_positions);
    if (cupholders_initialized_)   publishCupholderTFs(color_dbg, depth_dbg);

    /* increment counter & possibly disable further detection */
    if (new_positions.size() == 4) {
      ++successful_detections_;
      if (!detection_stopped_ && successful_detections_ >= DETECTION_LIMIT) {
        detection_stopped_ = true;
        RCLCPP_INFO(get_logger(),
          "Cupholder detection succeeded %d times switching to publish‑only mode.",
          successful_detections_);
      }
    }

    publishMat(color_dbg, debug_image_pub_);
  }

  /* history maintenance */
  void updateHistory(const std::vector<Eigen::Vector3f>& new_positions)
  {
    if (!cupholders_initialized_) {
      cached_cupholder_positions_ = new_positions;
      cupholders_initialized_ = true;
    } else {
      std::vector<bool> matched(4,false);
      auto updated = cached_cupholder_positions_;
      for (std::size_t i=0;i<4;++i)
      {
        float best = std::numeric_limits<float>::max(); int best_j=-1;
        for (std::size_t j=0;j<4;++j) {
          if (matched[j]) continue;
          float d = (cached_cupholder_positions_[i]-new_positions[j]).norm();
          if (d<best) {best=d; best_j=j;}
        }
        if (best_j>=0 && best<0.04f) {updated[i]=new_positions[best_j]; matched[best_j]=true;}
      }
      cached_cupholder_positions_ = updated;
      for (std::size_t i=0;i<4;++i) {
        cupholder_history_[i].push_back(updated[i]);
        if (cupholder_history_[i].size()>MAX_HISTORY) cupholder_history_[i].pop_front();
      }
    }
  }

  /* publish individual cupholder and pickup TFs */
  void publishCupholderTFs(const cv::Mat& color_dbg, const cv::Mat& depth_dbg)
  {
    geometry_msgs::msg::TransformStamped cam2base;
    try {
      cam2base = tf_buffer_->lookupTransform("base_link",
                                             "D415_color_optical_frame",
                                             tf2::TimePointZero);
    } catch (const tf2::TransformException &e) {
      RCLCPP_WARN(get_logger(), "TF lookup failed: %s", e.what());
      return;
    }

    tf2::Quaternion q_tf; tf2::fromMsg(cam2base.transform.rotation, q_tf);
    Eigen::Quaterniond q_eig(q_tf.w(), q_tf.x(), q_tf.y(), q_tf.z());
    Eigen::Matrix3d R = q_eig.toRotationMatrix();
    Eigen::Vector3d t(cam2base.transform.translation.x,
                      cam2base.transform.translation.y,
                      cam2base.transform.translation.z);

    std::vector<Eigen::Vector3d> cupholders_base;
    for (std::size_t i=0;i<4;++i)
    {
      if (cupholder_history_[i].empty()) continue;

      Eigen::Vector3f avg(0,0,0);
      for (const auto &p : cupholder_history_[i]) avg += p;
      avg /= static_cast<float>(cupholder_history_[i].size());

      geometry_msgs::msg::TransformStamped tf_cam;
      tf_cam.header.stamp    = now();
      tf_cam.header.frame_id = "D415_color_optical_frame";
      tf_cam.child_frame_id  = "cupholder_" + std::to_string(i);
      tf_cam.transform.translation.x = avg.x();
      tf_cam.transform.translation.y = avg.y();
      tf_cam.transform.translation.z = avg.z();
      tf_cam.transform.rotation      = cam2base.transform.rotation;
      tf_broadcaster_->sendTransform(tf_cam);

      Eigen::Vector3d c_cam(avg.x(), avg.y(), avg.z());
      cupholders_base.push_back(R * c_cam + t);
    }

    if (cupholders_base.empty()) return;
    // Compute barista height in base frame using the same cam2base transform
    double barista_z_base;
    if (barista_center_cam_.z() > 0) {
      Eigen::Vector3d b_cam(barista_center_cam_.x(), barista_center_cam_.y(), barista_center_cam_.z());
      barista_z_base = (R * b_cam + t).z();
    } else {
      // fallback to mean of cupholder heights if not yet available
      double tmp = 0.0; for (auto &p: cupholders_base) tmp += p.z();
      barista_z_base = tmp / cupholders_base.size();
    }

    double z_mean = barista_z_base; // now base-frame barista height

    for (std::size_t i=0;i<cupholders_base.size(); ++i)
    {
      geometry_msgs::msg::TransformStamped tf_base;
      tf_base.header.stamp    = now();
      tf_base.header.frame_id = "base_link";
      tf_base.child_frame_id  = "pickup_target_" + std::to_string(i) + "_top";
      tf_base.transform.translation.x = cupholders_base[i].x();
      tf_base.transform.translation.y = cupholders_base[i].y();
      tf_base.transform.translation.z = z_mean + 0.30;
      tf_base.transform.rotation.w    = 1.0;
      tf_broadcaster_->sendTransform(tf_base);
    }

    if (debug_image_pub_->get_subscription_count() > 0) {
      std_msgs::msg::Header h;  h.stamp = now();
      auto img_msg = cv_bridge::CvImage(h,
                      sensor_msgs::image_encodings::BGR8,
                      color_dbg).toImageMsg();
      debug_image_pub_->publish(*img_msg);
    }

    if (std::getenv("DISPLAY")) {
      cv::imshow("Color", color_dbg);
      cv::imshow("Depth", depth_dbg);
      cv::waitKey(1);
    }
  }

  void publishMat(const cv::Mat& img, const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& pub)
  {
    if (pub->get_subscription_count()==0) return;
    std_msgs::msg::Header h; h.stamp = now();
    auto msg = cv_bridge::CvImage(h, sensor_msgs::image_encodings::BGR8, img).toImageMsg();
    pub->publish(*msg);
  }
};


/* ------------------------- main --------------------------- */
int main(int argc,char** argv)
{
  rclcpp::init(argc,argv);
  rclcpp::spin(std::make_shared<CupholderDetector>());
  rclcpp::shutdown();
  return 0;
}
