#!/usr/bin/env python3


import rospy
import rosbag
import os
import time

from eus_imitation.utils.file_utils import sort_names_by_number


class PatchTimer(rospy.Time):
    # PatchTimer Time.now so we don't need to call rospy.init_node (so we don't need a roscore)
    # Also solves problems with /use_sim_time (simulated time)
    def __init__(self, secs=0, nsecs=0):
        super(rospy.Time, self).__init__(secs, nsecs)

    @staticmethod
    def now():
        # initialize with wallclock
        float_secs = time.time()
        secs = int(float_secs)
        nsecs = int((float_secs - secs) * 1000000000)
        return PatchTimer(secs, nsecs)


class RosbagUtils(object):
    @classmethod
    def get_rosbag_files(cls, record_dir):
        rosbag_files = []
        for file in os.listdir(record_dir):
            if file.endswith(".bag"):
                rosbag_files.append(file)
        rosbag_files = sort_names_by_number(rosbag_files)
        return rosbag_files

    @classmethod
    def get_rosbag_full_paths(cls, record_dir, rosbag_files):
        rosbag_full_paths = []
        for rosbag_file in rosbag_files:
            rosbag_full_paths.append(os.path.join(record_dir, rosbag_file))
        return rosbag_full_paths

    @classmethod
    def get_rosbag_abs_paths(cls, record_dir):
        rosbag_files = cls.get_rosbag_files(record_dir)
        rosbag_abs_paths = []
        for rosbag_file in rosbag_files:
            rosbag_abs_paths.append(
                os.path.abspath(os.path.join(record_dir, rosbag_file))
            )
        return rosbag_abs_paths
