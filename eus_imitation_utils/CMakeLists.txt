cmake_minimum_required(VERSION 3.0.2)
project(eus_imitation_utils)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  eus_imitation_msgs
  # catkin_virtualenv
)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Set up the Python virtual environment with isolated requirements
# catkin_generate_virtualenv(
#   INPUT_REQUIREMENTS requirements.txt
#   PYTHON_INTERPRETER python3
#   USE_SYSTEM_PACKAGES FALSE
#   CHECK_VENV FALSE
# )

# # Install Python scripts
# file(GLOB SCRIPT_PROGRAMS scripts/*.py)
# catkin_install_python(
#   PROGRAMS ${SCRIPT_PROGRAMS}
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts/)
# file(GLOB SCRIPTS_FILES node_scripts/*.py)
# catkin_install_python(
#   PROGRAMS ${SCRIPTS_FILES}
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# # Install additional files if needed
# install(FILES requirements.txt
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

# Add your other installation or build rules here
