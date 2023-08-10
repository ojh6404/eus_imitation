# eus_imitation

ROS Package for multi robot imitation learning including data collection and execution.
it works with imitation learning framework [imitator](https://github.com/ojh6404/imitator.git).

## Note
README will be updated soon.

#### Tested : Ubuntu 20.04 + ROS Noetic

## Installation
just build with catkin after cloning this repo.

### for rlds
please refer to scripts/rosbag2npy.py
assume that rosbags folder contains rosbag of each episode like rosbag-0.bag, rosbag-1.bag...
```bash
python3 rosbag2npy.py -d dir/to/rosbags
```
it will convert rosbag-0.bag to episode_0.npy
tfds dataset can be build with this [repo](https://github.com/ojh6404/rlds_dataset_builder.git) 
