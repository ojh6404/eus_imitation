cmake_minimum_required(VERSION 3.0.2)
project(eus_imitation_core)

find_package(catkin
  REQUIRED
  std_msgs
  sensor_msgs
  eus_imitation_msgs
)


catkin_package(
  CATKIN_DEPENDS
  message_runtime 
  std_msgs
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)

# if(CATKIN_ENABLE_TESTING)
#   add_custom_target(${PROJECT_NAME}_download_image ALL
#     COMMAND ${PROJECT_SOURCE_DIR}/test/prepare_test_data.sh)
#   find_package(catkin REQUIRED COMPONENTS rostest)
#   add_rostest(test/test_node.test)
# endif()
