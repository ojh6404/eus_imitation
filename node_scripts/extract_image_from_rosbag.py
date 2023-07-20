#!/usr/bin/env python3

from attrdict import AttrDict
import sys
import os
import time
import rospy
import rosbag
import rospkg
import message_filters
import h5py
import yaml

import cv2
from cv_bridge import CvBridge

import numpy as np
import argparse


from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from eus_imitation.msg import Float32MultiArrayStamped
from eus_imitation.utils.rosbag_utils import RosbagUtils, PatchTimer


rospy.Time = PatchTimer

def get_file_name_without_ext(file_path):
    file_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    return file_name_without_ext

class Rosbag2Dataset(object):
    def __init__(self, config, rosbag_file, image_only=False, out_bag=False):
        self.config = AttrDict(config)
        self.rosbag_file = rosbag_file
        self.data = self.config.dataset.data
        self.data_names = list(self.data.keys())

        self.slop = self.config.dataset.rosbag.message_filters.slop
        self.queue_size = self.config.dataset.rosbag.message_filters.queue_size
        self.image_only = image_only
        self.out_bag = out_bag



        if self.image_only:
            self.subs = [
                message_filters.Subscriber(
                    self.data[data_name]["topic_name"],
                    eval(self.data[data_name]["msg_type"]),
                )
                for data_name in self.data if self.data[data_name]["msg_type"] == "CompressedImage"]

            self.data_names = ["RGBImage"]
        else:
            self.subs = [
                message_filters.Subscriber(
                    self.data[data_name]["topic_name"],
                    eval(self.data[data_name]["msg_type"]),
                )
                for data_name in self.data
            ]



        self.sub_dict = AttrDict()
        for sub in self.subs:
            self.sub_dict[sub.topic] = sub

        self.topic_names = [sub.topic for sub in self.subs]
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.subs,
            queue_size=self.queue_size,
            slop=self.slop,
            allow_headerless=False,
        )
        self.ts.registerCallback(self.callback)

        self.data_file = h5py.File(
            get_file_name_without_ext(self.rosbag_file) + ".hdf5",
            mode="w",
        )

        self.outbag_file = get_file_name_without_ext(self.rosbag_file) + "_out.bag"

        self.cv_bridge = CvBridge()

        self.dataset = AttrDict({data_name: [] for data_name in self.data_names})

    def callback(self, *msgs):
        for topic_name, msg in zip(self.topic_names, msgs):
            # NOTE just for checking
            if self.out_bag:
                self.outbag.write(topic_name, msg, t=msg.header.stamp)

            # TODO creatq handler for conversion rules
            if topic_name == "/joint_states":
                pass
            #     self.dataset["JointStates"].append(
            #         np.array(msg.position, dtype=np.float32)
            #     )
            elif topic_name == "/pr2_imitation/robot_action":
                pass
                # self.dataset["RobotAction"].append(np.array(msg.data, dtype=np.float32))
            elif topic_name == "/kinect_head/rgb/image_rect_color/compressed":
                data = np.fromstring(msg.data, np.uint8)
                # data = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
                self.dataset["RGBImage"].append(np.array(data, dtype=np.uint8))

    # def processing(self):
    #     with rosbag.Bag(self.outbag_file, "w") as self.outbag:
    #         # print("Opening bag... (Only reading topics {})".format(self.topic_names))
    #         bag_reader = rosbag.Bag(self.rosbag_file, skip_index=True)
    #         # print("Synchronizing...")
    #         ini_t = time.time()
    #         for message_idx, (topic, msg, t) in enumerate(
    #             bag_reader.read_messages(topics=self.topic_names)
    #         ):
    #             # Send the message to the correct message_filters.Subscriber
    #             subscriber = self.sub_dict.get(topic)
    #             if subscriber:
    #                 subscriber.signalMessage(msg)
    #         fin_t = time.time()
    #         total_t = fin_t - ini_t

    #         for dataset_name in self.data_names:
    #             self.dataset[dataset_name] = np.array(self.dataset[dataset_name])
    #             self.data_file.create_dataset(
    #                 dataset_name, data=self.dataset[dataset_name]
    #             )

    #         self.data_file.flush()
    #         self.data_file.close()
    #         print("Done. (Parsed {} messages in {}s)".format(message_idx, total_t))


    def processing(self):
        with rosbag.Bag(self.outbag_file, "w") as self.outbag:
            # print("Opening bag... (Only reading topics {})".format(self.topic_names))
            bag_reader = rosbag.Bag(self.rosbag_file, skip_index=True)
            # print("Synchronizing...")
            ini_t = time.time()
            for message_idx, (topic, msg, t) in enumerate(
                bag_reader.read_messages(topics=self.topic_names)
            ):
                # Send the message to the correct message_filters.Subscriber
                subscriber = self.sub_dict.get(topic)
                if subscriber:
                    subscriber.signalMessage(msg)
            fin_t = time.time()
            total_t = fin_t - ini_t

            for dataset_name in self.data_names:
                self.dataset[dataset_name] = np.array(self.dataset[dataset_name])
                self.data_file.create_dataset(
                    dataset_name, data=self.dataset[dataset_name]
                )

            del self.dataset

            self.data_file.flush()
            self.data_file.close()
            bag_reader.close()
            print("Done. (Parsed {} messages in {}s)".format(message_idx, total_t))


def main(config, rosbag_dir,image_only=False):
    # Get an input rosbag
    rosbag_dir = '/home/oh/.mohou/pr2_bowl/rosbag'
    dir_list = RosbagUtils.get_rosbag_files(rosbag_dir)
    full_paths = RosbagUtils.get_rosbag_full_paths(rosbag_dir, dir_list)


    package_path = rospkg.RosPack().get_path("eus_imitation")
    rosbag_file = os.path.join(package_path, "data", "rosbag-0.bag")
    config_path = os.path.join(package_path, "config", "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)



    for rosbag_file in full_paths:
        print(rosbag_file)
        rosbag2dataset = Rosbag2Dataset(config, rosbag_file, image_only=image_only)
        rosbag2dataset.processing()
        del rosbag2dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--rosbag_dir", type=str, default="rosbag.bag")
    parser.add_argument("--rosbag_out", action="store_true", default="output rosbag")
    args = parser.parse_args()

    config = args.config
    rosbag_dir = args.rosbag_dir
    main(config, rosbag_dir, image_only=True)
