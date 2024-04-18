cmake_minimum_required(VERSION 3.0.2)
project(eus_imitation)

find_package(catkin
  REQUIRED
  message_generation
  std_msgs
  sensor_msgs
  catkin_virtualenv
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


catkin_generate_virtualenv(
  INPUT_REQUIREMENTS requirements.txt
  PYTHON_INTERPRETER python3.9
  USE_SYSTEM_PACKAGES TRUE
  ISOLATE_REQUIREMENTS FALSE
  CHECK_VENV FALSE
)

file(GLOB SCRIPT_PROGRAMS scripts/*.py)
catkin_install_python(
  PROGRAMS ${SCRIPT_PROGRAMS}
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts/)
file(GLOB SCRIPTS_FILES node_scripts/*.py)
catkin_install_python(
  PROGRAMS ${SCRIPTS_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(FILES
  requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
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
