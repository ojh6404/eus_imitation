#!/usr/bin/env python3
import sys
import time
import rospy
import rosbag
import message_filters
import h5py

import numpy as np
import argparse


from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from eus_imitation.msg import Float32MultiArrayStamped
from eus_imitation.utils.rosbag_utils import RosbagUtils, PatchTimer


rospy.Time = PatchTimer


class Rosbags2Dataset(object):
    def __init__(self, config_file, rosbag_file):
        self.config = config
        self.rosbag_files = rosbag_files
        self.record_dir = record_dir

    def callback(self, *msgs):
        for topic_name, msg in zip(topic_names, msgs):
            if topic_name == "/pr2_imitation/robot_action":
                print("robot_action", msg)


def callback(*msgs):
    for topic_name, msg in zip(topic_names, msgs):
        if topic_name == "/pr2_imitation/robot_action":
            print("robot_action", msg)


def main(config, rosbag_dir):
    # Get an input rosbag

    rosbag_file = "/home/oh/ros/pr2_ws/src/eus_imitation/data/rosbag-0.bag"  # for test

    data_file = h5py.File("/tmp/" + str(0) + ".hdf5", mode="w")

    data_file.create_dataset(
        "joint_states", (0, 7), maxshape=(None, 7), dtype=np.float32
    )
    data_file.flush()
    data_file.close()

    # Create a rosbag with only synchronized messages
    joint_states_sub = message_filters.Subscriber("/joint_states", JointState)
    compressed_img_sub = message_filters.Subscriber(
        "/kinect_head/rgb/image_rect_color/compressed", CompressedImage
    )
    robot_action_sub = message_filters.Subscriber(
        "/pr2_imitation/robot_action", Float32MultiArrayStamped
    )

    # ApproximateTimeSynchronizer wants a list of subscribers
    subscriber_list = [joint_states_sub, compressed_img_sub, robot_action_sub]
    # Customize ApproximateTimeSynchronizer parameters
    ats_queue_size = 1000  # Max messages in any queue
    ats_slop = 0.1  # Max delay to allow between messages
    rosbag_reader_skip_index = (
        False  # Makes opening the bag faster, but if the bag is unindexed it will fail
    )
    # -------------------------------------------------------------------

    # We want a dictionary view of the list for efficiency when dispatching messages
    subscriber_dict = {}
    for subscriber in subscriber_list:
        subscriber_dict[subscriber.topic] = subscriber
    # We want a list with the topic names in the same order to correctly dispatch messages

    topic_names = [subscriber.topic for subscriber in subscriber_list]
    ts = message_filters.ApproximateTimeSynchronizer(
        subscriber_list,
        queue_size=ats_queue_size,
        slop=ats_slop,
        allow_headerless=False,
    )

    ts.registerCallback(callback)

    print("Opening bag... (Only reading topics {})".format(topic_names))
    bag_reader = rosbag.Bag(rosbag_file, skip_index=True)
    print("Synchronizing...")
    ini_t = time.time()
    for message_idx, (topic, msg, t) in enumerate(
        bag_reader.read_messages(topics=topic_names)
    ):
        # Send the message to the correct message_filters.Subscriber
        subscriber = subscriber_dict.get(topic)
        if subscriber:
            # Show some output to show we are alive
            if message_idx % 1000 == 0:
                print(
                    "Message #{}, Topic: {}, message stamp: {}".format(
                        message_idx, topic, msg.header.stamp
                    )
                )
            subscriber.signalMessage(msg)
    fin_t = time.time()
    total_t = fin_t - ini_t

    print("Done. (Parsed {} messages in {}s)".format(message_idx, total_t))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="config.yaml")
    args.add_argument("--rosbag_dir", type=str, default="rosbag.bag")
    args.parse_args()

    config = args.config
    rosbag_dir = args.rosbag_dir
    main(config, rosbag_dir)
