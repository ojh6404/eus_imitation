cmake_minimum_required(VERSION 2.8.3)
project(eus_imitation)

find_package(catkin
  REQUIRED
  catkin_virtualenv
  message_generation
  std_msgs
)

add_message_files(
  FILES
  Float32MultiArrayStamped.msg
)


catkin_python_setup()

catkin_generate_virtualenv(
  INPUT_REQUIREMENTS requirements.txt
  PYTHON_INTERPRETER python3
  USE_SYSTEM_PACKAGES TRUE
  ISOLATE_REQUIREMENTS FALSE
  CHECK_VENV FALSE
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


