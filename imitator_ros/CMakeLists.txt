cmake_minimum_required(VERSION 3.0.2)
project(imitator_ros)

find_package(catkin
  REQUIRED
  std_msgs
  sensor_msgs
  eus_imitation_msgs
  catkin_virtualenv
)


catkin_package(
  CATKIN_DEPENDS
  message_runtime 
  std_msgs
)


catkin_generate_virtualenv(
  INPUT_REQUIREMENTS requirements.txt
  PYTHON_INTERPRETER python3
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
