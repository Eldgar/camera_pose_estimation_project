/*********************************************************************
 * web_action_bridge.cpp – camera-calibrate, arm-control, RG2 move
 *********************************************************************/

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <std_msgs/msg/string.hpp>
#include <nlohmann/json.hpp>

#include "ur_action_servers/action/camera_calibrate.hpp"
#include "ur_action_servers/action/arm_control.hpp"
#include "ur_action_servers/action/rg2_relative_move.hpp"

using namespace std::chrono_literals;
using json = nlohmann::json;

/* ───────── helper ───────── */
inline json parse_or_wrap(const std::string& raw)
{
  if (json::accept(raw)) return json::parse(raw);
  return json{{"command", raw}};
}
inline json strict_parse(const std::string& raw)
{
  if (!json::accept(raw))
    throw std::runtime_error("payload is not valid JSON");
  return json::parse(raw);
}

/* ───────── WebActionBridge ───────── */
class WebActionBridge : public rclcpp::Node
{
public:
  using CamAct  = ur_action_servers::action::CameraCalibrate;
  using ArmAct  = ur_action_servers::action::ArmControl;
  using MoveAct = ur_action_servers::action::Rg2RelativeMove;

  WebActionBridge() : Node("web_action_bridge")
  {
    //* inbound topics */
    cam_goal_sub_ = create_subscription<std_msgs::msg::String>(
    "/camera_calibrate/goal", 10,
    [this](const std_msgs::msg::String::SharedPtr msg)
    { cam_goal_cb(msg); });

    arm_goal_sub_ = create_subscription<std_msgs::msg::String>(
    "/arm_control/goal", 10,
    [this](const std_msgs::msg::String::SharedPtr msg)
    { arm_goal_cb(msg); });

    move_goal_sub_ = create_subscription<std_msgs::msg::String>(
    "/rg2_relative_move/goal", 10,
    [this](const std_msgs::msg::String::SharedPtr msg)
    { move_goal_cb(msg); });


    /* publishers back to UI */
    cam_fb_pub_  = create_publisher<std_msgs::msg::String>("/camera_calibrate/feedback", 10);
    cam_res_pub_ = create_publisher<std_msgs::msg::String>("/camera_calibrate/result", 10);
    arm_fb_pub_  = create_publisher<std_msgs::msg::String>("/arm_control/feedback", 10);
    arm_res_pub_ = create_publisher<std_msgs::msg::String>("/arm_control/result", 10);
    move_fb_pub_ = create_publisher<std_msgs::msg::String>("/rg2_relative_move/feedback", 10);
    move_res_pub_= create_publisher<std_msgs::msg::String>("/rg2_relative_move/result", 10);

    /* action clients */
    cam_cli_  = rclcpp_action::create_client<CamAct >(this,"/camera_calibrate");
    arm_cli_  = rclcpp_action::create_client<ArmAct >(this,"/arm_control");
    move_cli_ = rclcpp_action::create_client<MoveAct>(this,"/rg2_relative_move");

    RCLCPP_INFO(get_logger(),"✅ Web Action Bridge initialised");
  }

/* ── camera-calibrate path ───────────────────────────────────────── */
private:
  void cam_goal_cb(const std_msgs::msg::String::SharedPtr msg)
  {
    try{
      if(!cam_cli_->wait_for_action_server(2s))
        throw std::runtime_error("action server unavailable");

      CamAct::Goal goal;
      goal.command = parse_or_wrap(msg->data).value("command","calibrate");

      auto opts = rclcpp_action::Client<CamAct>::SendGoalOptions();
      opts.goal_response_callback = [this](auto gh){
        if(!gh) send_simple(cam_res_pub_,false,"Goal rejected"); };
      opts.feedback_callback = [this](auto, auto fb){
        send_json(cam_fb_pub_, json{{"status", fb->status}}); };
      opts.result_callback = [this](auto res){
        json j; j["success"]= (res.code==rclcpp_action::ResultCode::SUCCEEDED);
        j["message"] = j["success"]? res.result->message : "failed/aborted";
        send_json(cam_res_pub_, j); };
      cam_cli_->async_send_goal(goal, opts);
    }catch(const std::exception& e){
      send_simple(cam_res_pub_,false,e.what());
    }
  }

/* ── arm-control path ────────────────────────────────────────────── */
  void arm_goal_cb(const std_msgs::msg::String::SharedPtr msg)
  {
    try{
      if(!arm_cli_->wait_for_action_server(2s))
        throw std::runtime_error("action server unavailable");

      ArmAct::Goal goal;
      goal.command = parse_or_wrap(msg->data).value("command","home");

      auto opts = rclcpp_action::Client<ArmAct>::SendGoalOptions();
      opts.goal_response_callback=[this](auto gh){
        if(!gh) send_simple(arm_res_pub_,false,"Goal rejected"); };
      opts.feedback_callback=[this](auto,auto fb){
        send_json(arm_fb_pub_, json{{"status", fb->status}}); };
      opts.result_callback=[this](auto res){
        json j; j["success"]= (res.code==rclcpp_action::ResultCode::SUCCEEDED);
        j["message"]= j["success"]? res.result->message : "failed/aborted";
        send_json(arm_res_pub_, j); };
      arm_cli_->async_send_goal(goal, opts);
    }catch(const std::exception& e){
      send_simple(arm_res_pub_,false,e.what());
    }
  }

/* ── RG2 relative-move path ──────────────────────────────────────── */
  void move_goal_cb(const std_msgs::msg::String::SharedPtr msg)
  {
    try{
      if(!move_cli_->wait_for_action_server(2s))
        throw std::runtime_error("action server unavailable");

      json j = strict_parse(msg->data);
      if(!j.contains("offset")||!j["offset"].is_object())
        throw std::runtime_error("offset object missing");

      MoveAct::Goal goal;
      goal.x = j["offset"].value("x",0.0);
      goal.y = j["offset"].value("y",0.0);
      goal.z = j["offset"].value("z",0.0);

      auto opts = rclcpp_action::Client<MoveAct>::SendGoalOptions();
      opts.goal_response_callback=[this](auto gh){
        if(!gh) send_simple(move_res_pub_,false,"Goal rejected"); };
      opts.feedback_callback=[this](auto,auto fb){
        send_json(move_fb_pub_, json{{"progress", fb->progress}}); };
      opts.result_callback=[this](auto res){
        json jr; jr["success"]=(res.code==rclcpp_action::ResultCode::SUCCEEDED);
        jr["message"]=jr["success"]? res.result->message : "failed/aborted";
        send_json(move_res_pub_, jr); };
      move_cli_->async_send_goal(goal, opts);
    }catch(const std::exception& e){
      send_simple(move_res_pub_,false,e.what());
    }
  }

/* ── helpers ─────────────────────────────────────────────────────── */
  void send_json(const rclcpp::Publisher<std_msgs::msg::String>::SharedPtr& pub,
                 const json& j)
  {
    std_msgs::msg::String m; m.data = j.dump(); pub->publish(m);
  }
  void send_simple(const rclcpp::Publisher<std_msgs::msg::String>::SharedPtr& pub,
                   bool ok, const std::string& msg)
  {
    send_json(pub, json{{"success",ok},{"message",msg}});
  }

/* ── member fields ──────────────────────────────────────────────── */
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr cam_goal_sub_,arm_goal_sub_,move_goal_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cam_fb_pub_,cam_res_pub_,arm_fb_pub_,arm_res_pub_,move_fb_pub_,move_res_pub_;
  rclcpp_action::Client<CamAct>::SharedPtr  cam_cli_;
  rclcpp_action::Client<ArmAct>::SharedPtr  arm_cli_;
  rclcpp_action::Client<MoveAct>::SharedPtr move_cli_;
};

/* ───────── main ───────── */
int main(int argc,char** argv)
{
  rclcpp::init(argc,argv);
  auto bridge = std::make_shared<WebActionBridge>();
  rclcpp::spin(bridge);
  rclcpp::shutdown();
  return 0;
}
