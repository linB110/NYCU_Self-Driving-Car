cmake_minimum_required(VERSION 3.0.2)
project(radar_relative_speed)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/radar_relative_speed.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
