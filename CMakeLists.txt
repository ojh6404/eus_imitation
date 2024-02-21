cmake_minimum_required(VERSION 3.0.2)
project(eus_imitation)

find_package(catkin
  REQUIRED
  message_generation
  std_msgs
  sensor_msgs
)

add_message_files(
  FILES
  Float32MultiArrayStamped.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime 
  std_msgs
)


