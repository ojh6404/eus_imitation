#!/usr/bin/env python3
import os
import re
import argparse
import pprint
from tqdm import tqdm

import numpy as np
import cv2
from moviepy.editor import ImageSequenceClip
from tunable_filter.composite_zoo import HSVBlurCropResolFilter

import rospy
import rosbag
import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage, Image, JointState
from eus_imitation.msg import Float32MultiArrayStamped
import imitator.utils.ros_utils as RosUtils
import imitator.utils.file_utils as FileUtils

# for no roscore
rospy.Time = RosUtils.PatchTimer

def extract_number(name):
    match = re.search(r"\d+", name)
    if match:
        return int(match.group())
    else:
        return 0


def sort_names_by_number(names):
    sorted_names = sorted(names, key=extract_number)
    return sorted_names

def main(args):
    # get rosbags list of absolute path
    # rosbags are rosbag-0.bag, ..., rosbag-9.bag

    rosbags = []
    for file in os.listdir(args.rosbag_dir):
        if file.endswith(".bag"):
            rosbags.append(file)
    rosbags = sort_names_by_number(rosbags)
    print(rosbags)
    rosbag_abs_paths = []
    for file in rosbags:
        rosbag_abs_paths.append(
            os.path.abspath(os.path.join(args.rosbag_dir, file))
        )
    rosbags = rosbag_abs_paths


    print("Found {} rosbags".format(len(rosbags)))

    print("rosbags ", rosbags)

    queue_size = 1000
    slop = 0.1

    obs_keys = ["image", "state"]
    obs_topics = ["/kinect_head/rgb/image_rect_color/compressed", "/eus_imitation/robot_state"]
    obs_msg_types = [CompressedImage, Float32MultiArrayStamped]
    action_topic = "/eus_imitation/robot_action"
    action_msg_type = Float32MultiArrayStamped

    print("obs_keys: {}".format(obs_keys))
    print("obs_topics: {}".format(obs_topics))
    print("obs_msg_types: {}".format(obs_msg_types))
    print("action_topic: {}".format(action_topic))
    print("action_msg_type: {}".format(action_msg_type))

    topics_to_keys = dict(zip(obs_topics + [action_topic], obs_keys + ["action"]))
    topics = obs_topics + [action_topic]
    msg_types = obs_msg_types + [action_msg_type]

    subscribers = dict()
    for topic, msg_type in zip(topics, msg_types):
        subscribers[topic] = message_filters.Subscriber(topic, msg_type)

    img_bridge = CvBridge()
    ts = message_filters.ApproximateTimeSynchronizer(subscribers.values(), queue_size=queue_size, slop=slop, allow_headerless=False)


    print("Observation and Action keys: {}".format(obs_keys + ["action"]))
    print("Subscribing to topics: {}".format(topics))


    def callback(*msgs):
        data_buffer = dict()
        for topic, msg in zip(topics, msgs):
            if topic == action_topic: # action
                data = np.array(msg.data).astype(np.float32)
                data_buffer["action"] = data
            else:
                if "Image" in msg._type:
                    if "Compressed" in msg._type:
                        data = img_bridge.compressed_imgmsg_to_cv2(msg, "rgb8").astype(np.uint8)
                    else:
                        data = img_bridge.imgmsg_to_cv2(msg, "rgb8").astype(np.uint8)
                    if args.image_tune: # tuning image with first data
                        tunable = HSVBlurCropResolFilter.from_image(data)
                        print("Press q to finish tuning")
                        tunable.launch_window()
                        tunable.start_tuning(data)
                        pprint.pprint(tunable.export_dict())
                        tunable.dump_yaml(
                            os.path.join(
                                FileUtils.get_config_folder(args.project_name),
                                "image_filter.yaml",
                            )
                        )
                        exit(0)
                    else: # load from yaml and proess image
                        assert os.path.exists(
                            os.path.join(
                                FileUtils.get_config_folder(args.project_name),
                                "image_filter.yaml",
                            )
                        ), "image_filter.yaml not found. Please run with --image_tune first."
                        tunable = HSVBlurCropResolFilter.from_yaml(
                            os.path.join(
                                FileUtils.get_config_folder(args.project_name),
                                "image_filter.yaml",
                            )
                        )
                        data = tunable(data)
                else:
                    data = np.array(msg.data).astype(np.float32)
                data_buffer[topics_to_keys[topic]] = data

        assert len(data_buffer.keys()) == len(topics), "Some topics are missing"
        episode_buffer.append(data_buffer)

    ts.registerCallback(callback)

    # create directory if not exist
    output_dir = os.path.join(FileUtils.get_project_folder(args.project_name), "data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Processing rosbags...")
    for i, bag in enumerate(tqdm(rosbags)):
        bag_reader = rosbag.Bag(bag, skip_index=True)

        episode_buffer= []

        for message_idx, (topic, msg, t) in enumerate(
            bag_reader.read_messages(topics=topics)
        ):
            subscriber = subscribers.get(topic)
            if subscriber:
                subscriber.signalMessage(msg)


        os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
        np.save(os.path.join(output_dir, "train", "episode_{}.npy".format(i)), episode_buffer)

        os.makedirs(os.path.join(output_dir, "gif"), exist_ok=True)
        if i % 1 == 0 and args.gif:
            # episode_buffer is list of dict
            test_images = []
            for data_buffer in episode_buffer:
                test_images.append(data_buffer["image"])
            clip = ImageSequenceClip(test_images, fps=10)
            clip.write_gif(
                os.path.join(output_dir, "gif", "episode_{}.gif".format(i)),
                fps=10,
                verbose=False,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-d", "--rosbag_dir", type=str, default="/tmp/dataset")
    parser.add_argument("-t", "--image_tune", action="store_true", default=False)
    parser.add_argument("--gif", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
