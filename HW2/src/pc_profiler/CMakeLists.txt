cmake_minimum_required(VERSION 3.0.2)
project(pc_profiler)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/pc_profiler.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
