#!/usr/bin/env bash

# run docker eus_imitation


# set path variables


docker run --rm --net=host -it --gpus 1 \
    -v /home/leus/ros/test_ws/src/eus_imitation/node_scripts:/home/user/catkin_ws/src/eus_imitation/node_scripts \
    eus_imitation:latest \
    /bin/bash
