cmake_minimum_required(VERSION 3.8)
project(cupholder_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(tf2_geometry_msgs REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(cupholder_detector src/cupholder_detector.cpp)
ament_target_dependencies(cupholder_detector
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_geometry_msgs
  tf2_ros
  cv_bridge
  image_transport
  pcl_conversions

)
target_link_libraries(cupholder_detector ${PCL_LIBRARIES})
include_directories(${PCL_INCLUDE_DIRS})

target_link_libraries(cupholder_detector ${OpenCV_LIBS})

install(TARGETS
  cupholder_detector
  DESTINATION lib/${PROJECT_NAME})


add_executable(cupholder_detector_real src/cupholder_detector_real.cpp)
ament_target_dependencies(cupholder_detector_real
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_geometry_msgs
  tf2_ros
  cv_bridge
  image_transport
  pcl_conversions
)
target_link_libraries(cupholder_detector_real ${PCL_LIBRARIES})
include_directories(${PCL_INCLUDE_DIRS})

target_link_libraries(cupholder_detector_real ${OpenCV_LIBS})

install(TARGETS
  cupholder_detector_real
  DESTINATION lib/${PROJECT_NAME})


ament_package()
