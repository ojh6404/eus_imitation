#!/usr/bin/env python3

import argparse
from tqdm import tqdm
import os
import glob
import time
import numpy as np

import rospy
import rosbag
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, JointState
from eus_imitation.msg import Float32MultiArrayStamped

"""
Example script for convert rosbag to npy.
"""

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
    rosbags = glob.glob(args.rosbag_dir + "/*.bag")

    # train val split
    # shuffle rosbags
    np.random.shuffle(rosbags)
    train_rosbags = rosbags[:int(len(rosbags) * args.ratio)]
    val_rosbags = rosbags[int(len(rosbags) * args.ratio):]
    print("Train rosbags : {}".format(train_rosbags))
    print("Val rosbags : {}".format(val_rosbags))


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
            else:
                data = np.array(msg.data).astype(np.float32)
            data_buffer[topics_to_keys[topic_name]] = data
        episode_buffer.append(data_buffer)

    ts.registerCallback(callback)

    # process rosbags
    print("Start processing train rosbags...")
    for i, bag in enumerate(tqdm(train_rosbags)):
        bag_reader = rosbag.Bag(bag, skip_index=True)
        episode_buffer = list() # list of dicts with keys: image, state, action

        for message_idx, (topic, msg, t) in enumerate(bag_reader.read_messages(topics=topic_names)):
            subscriber = subscriber_dict.get(topic)
            if subscriber:
                subscriber.signalMessage(msg)

        # save episode buffer to npy
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/train", exist_ok=True)
        np.save("data/train/episode_{}.npy".format(i), episode_buffer)

    print("Start processing val rosbags...")
    for i, bag in enumerate(tqdm(val_rosbags)):
        bag_reader = rosbag.Bag(bag, skip_index=True)
        episode_buffer = list() # list of dicts with keys: image, state, action

        for message_idx, (topic, msg, t) in enumerate(bag_reader.read_messages(topics=topic_names)):
            subscriber = subscriber_dict.get(topic)
            if subscriber:
                subscriber.signalMessage(msg)

        # save episode buffer to npy
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/val", exist_ok=True)
        np.save("data/val/episode_{}.npy".format(i), episode_buffer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synchronize messages from a rosbag.')
    parser.add_argument('-d', "--rosbag_dir", type=str, help='The name of the input bag file directory.')
    parser.add_argument('-r', "--ratio", type=float, help="train/val ratio. e.g. 0.8", default=0.8)
    args = parser.parse_args()
    main(args)
