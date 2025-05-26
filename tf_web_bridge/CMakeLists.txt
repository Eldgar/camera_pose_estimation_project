cmake_minimum_required(VERSION 3.8)
project(tf_web_bridge)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ur_action_servers REQUIRED)
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Declare the executable
add_executable(tf_web_bridge src/tf_web_bridge.cpp)
ament_target_dependencies(tf_web_bridge
  rclcpp
  tf2_ros
  geometry_msgs
  tf2_msgs
  rclcpp_action
  std_msgs
  ur_action_servers
)
target_link_libraries(tf_web_bridge
  nlohmann_json::nlohmann_json
)

add_executable(web_action_bridge src/web_action_bridge.cpp)
ament_target_dependencies(web_action_bridge
  rclcpp
  rclcpp_action
  std_msgs
  ur_action_servers
  geometry_msgs
)
target_link_libraries(web_action_bridge
  nlohmann_json::nlohmann_json
)

# Install the executable
install(TARGETS
  tf_web_bridge
  web_action_bridge
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)


ament_package()


