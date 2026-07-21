cmake_minimum_required(VERSION 3.5)
project(humanoid_arm_description)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_library(humanoid_arm_gazebo_joint_driver SHARED
  src/gazebo_joint_driver.cpp
)
target_include_directories(humanoid_arm_gazebo_joint_driver PRIVATE
  ${GAZEBO_INCLUDE_DIRS}
)
target_link_libraries(humanoid_arm_gazebo_joint_driver
  ${GAZEBO_LIBRARIES}
)
ament_target_dependencies(humanoid_arm_gazebo_joint_driver
  gazebo_ros
  rclcpp
  sensor_msgs
  trajectory_msgs
)

install(TARGETS humanoid_arm_gazebo_joint_driver
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY config launch urdf worlds
  DESTINATION share/${PROJECT_NAME}
)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

ament_package()
