cmake_minimum_required(VERSION 3.8)
project(auv_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/AddWaypoint.srv"
  "srv/ClearWaypoints.srv"
  "srv/SetGoal.srv"
  DEPENDENCIES geometry_msgs
)

ament_package()
