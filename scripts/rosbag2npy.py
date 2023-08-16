#!/usr/bin/env python3

import argparse
import random
from tqdm import tqdm
import os
import re
import time
import pprint
import numpy as np

import rospy
import rosbag
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, JointState
from eus_imitation.msg import Float32MultiArrayStamped

from moviepy.editor import ImageSequenceClip
from tunable_filter.composite_zoo import HSVBlurCropResolFilter

"""
Example script for convert rosbag to npy.
"""


def extract_number(name):
    match = re.search(r"\d+", name)
    if match:
        return int(match.group())
    else:
        return 0


def sort_names_by_number(names):
    sorted_names = sorted(names, key=extract_number)
    return sorted_names

class PatchTime(rospy.Time):
    """
    Time Patcher for rosbag processing.
    it is needed to run script without roscore and resolve time issues.
    """
    def __init__(self, secs=0, nsecs=0):
        super(rospy.Time, self).__init__(secs, nsecs)
    @staticmethod
    def now():
        float_secs = time.time()
        secs = int(float_secs)
        nsecs = int((float_secs - secs) * 1000000000)
        return PatchTime(secs, nsecs)

rospy.Time = PatchTime

def main(args):

    # rosbag files should be ordered by recorded time, in absolute path
    rosbags = []
    for file in os.listdir(args.rosbag_dir):
        if file.endswith(".bag"):
            rosbags.append(file)
    rosbags = sort_names_by_number(rosbags)
    for i, file in enumerate(rosbags):
        rosbags[i] = os.path.abspath(os.path.join(args.rosbag_dir, file))
    train_rosbags = random.sample(rosbags, int(len(rosbags) * args.ratio))
    val_rosbags = list(set(rosbags) - set(train_rosbags))

    # define keys and topics
    keys = ["image", "state", "action"]
    topics = ["/kinect_head/rgb/image_rect_color/compressed", "/eus_imitation/robot_state", "/eus_imitation/robot_action"]
    msg_types = [CompressedImage, Float32MultiArrayStamped, Float32MultiArrayStamped]
    topics_to_keys = dict(zip(topics, keys))

    # create subscribers
    subscriber_dict = {}
    subscriber_list = [message_filters.Subscriber(topic, msg_type) for topic, msg_type in zip(topics, msg_types)]
    queue_size = 1000
    slop = 0.1
    for subscriber in subscriber_list:
        subscriber_dict[subscriber.topic] = subscriber
    topic_names = [subscriber.topic for subscriber in subscriber_list]
    ts = message_filters.ApproximateTimeSynchronizer(subscriber_list,
                                                    queue_size=queue_size,
                                                    slop=slop,
                                                    allow_headerless=False)

    print("Observation and Action keys : {}".format(keys))
    print("Subscribing topics : {}".format(topic_names))

    # define callback
    def callback(*msgs):
        data_buffer = dict()
        for topic_name, msg in zip(topic_names, msgs):
            if "Image" in msg._type:
                # you can define some transformation here like cropping, resizing, etc.
                if "Compressed" in msg._type:
                    data = CvBridge().compressed_imgmsg_to_cv2(msg, "rgb8").astype(np.uint8)
                else:
                    data = CvBridge().imgmsg_to_cv2(msg, "rgb8").astype(np.uint8)
                if args.image_tune: # tuning image with first data
                    tunable = HSVBlurCropResolFilter.from_image(data)
                    print("Press q to finish tuning")
                    tunable.launch_window()
                    tunable.start_tuning(data)
                    pprint.pprint(tunable.export_dict())
                    tunable.dump_yaml(
                        os.path.join("data", "image_filter.yaml")
                    )
                    exit(0)
                else:
                    tunable = HSVBlurCropResolFilter.from_yaml(
                        os.path.join(
                            "data", "image_filter.yaml"
                        )
                    )
                    data = tunable(data)
            else:
                data = np.array(msg.data).astype(np.float32)
            data_buffer[topics_to_keys[topic_name]] = data
        episode_buffer.append(data_buffer)

    ts.registerCallback(callback)

    # process rosbags
    train_num = 0
    val_num = 0
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    print("Start processing train rosbags...")
    for i, bag in enumerate(tqdm(rosbags)):
        bag_reader = rosbag.Bag(bag, skip_index=True)
        episode_buffer = list() # list of dicts with keys: image, state, action

        for message_idx, (topic, msg, t) in enumerate(bag_reader.read_messages(topics=topic_names)):
            subscriber = subscriber_dict.get(topic)
            if subscriber:
                subscriber.signalMessage(msg)

        # save episode buffer to npy
        if bag in train_rosbags:
            np.save("data/train/episode_{}.npy".format(train_num), episode_buffer)
            train_num += 1
        elif bag in val_rosbags:
            np.save("data/val/episode_{}.npy".format(val_num), episode_buffer)
            val_num += 1


        if i % 1 == 0 and args.gif:
            os.makedirs("data/gif", exist_ok=True)
            # episode_buffer is list of dict
            test_images = []
            for data_buffer in episode_buffer:
                test_images.append(data_buffer["image"])
            clip = ImageSequenceClip(test_images, fps=10)
            clip.write_gif(
                os.path.join("data", "gif", "episode_{}.gif".format(i)),
                fps=10,
                verbose=False,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synchronize messages from a rosbag.')
    parser.add_argument('-d', "--rosbag_dir", type=str, help='The name of the input bag file directory.')
    parser.add_argument('-r', "--ratio", type=float, help="train/val ratio. e.g. 0.8", default=0.8)
    parser.add_argument("-t", "--image_tune", action="store_true", default=False)
    parser.add_argument("--gif", action="store_true", help="save gif")
    args = parser.parse_args()
    main(args)
