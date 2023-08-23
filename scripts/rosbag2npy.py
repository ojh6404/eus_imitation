#!/usr/bin/env python3

import argparse
import random
import yaml
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

    action_max = None
    action_min = None
    state_max = None
    state_min = None

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
                        os.path.join(
                            args.project_name, "data", "image_filter.yaml"
                        )
                    )
                    exit(0)
                else:
                    tunable = HSVBlurCropResolFilter.from_yaml(
                        os.path.join(
                            args.project_name, "data", "image_filter.yaml"
                        )
                    )
                    data = tunable(data)
            else:
                data = np.array(msg.data).astype(np.float32)
                # data is x, y, z, gripper_pos, gripper_pos is 0.0 to 0.08
                # convert gripper_pos to 0, 1
                # if topic_name == "/eus_imitation/robot_action":
                    # data[3] = 0.0 if data[3] > 0.085 else 1.0

                # concat rpy angle [pi/2, pi/2 -pi/2] to data
                # data = np.concatenate([data, np.array([np.pi/2, np.pi/2, -np.pi/2])]).astype(np.float32)

                # # reorder data from [x,y,z,gripper,r,p,y] to [x,y,z,r,p,y,gripper]
                # data = np.concatenate([data[:3], data[4:], data[3:4]]).astype(np.float32)
            data_buffer[topics_to_keys[topic_name]] = data
        episode_buffer.append(data_buffer)

    ts.registerCallback(callback)

    # process rosbags
    train_num = 0
    val_num = 0
    os.makedirs(os.path.join(args.project_name, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.project_name, "data", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.project_name, "data", "val"), exist_ok=True)
    print("Start processing train rosbags...")
    for i, bag in enumerate(tqdm(rosbags)):
        bag_reader = rosbag.Bag(bag, skip_index=True)
        episode_buffer = list() # list of dicts with keys: image, state, action

        for message_idx, (topic, msg, t) in enumerate(bag_reader.read_messages(topics=topic_names)):
            subscriber = subscriber_dict.get(topic)
            if subscriber:
                subscriber.signalMessage(msg)


        # episode_buffer is list of dict containing keys: image, state, action
        # action is target end effector pos and rpy + gripper command
        # state is current end effector pos and rpy + gripper pos
        # we need delta of action and state as action
        # and concatenate terminal action, which is 1 only at the last action

        # convert action to delta
        # for i in range(len(episode_buffer)):
        #     # add terminal action flag
        #     if i != len(episode_buffer) - 1:
        #         episode_buffer[i]["action"] = np.concatenate([episode_buffer[i]["action"], np.array([0])])
        #     else:
        #         episode_buffer[i]["action"] = np.concatenate([episode_buffer[i]["action"], np.array([1])])
        #     # convert action to delta
        #     episode_buffer[i]["action"][0:3] = episode_buffer[i]["action"][0:3] - episode_buffer[i]["state"][0:3] # delta xyz
        #     episode_buffer[i]["action"][3:6] = episode_buffer[i]["action"][3:6] - episode_buffer[i]["state"][3:6] # delta rpy
        #
        for j in range(len(episode_buffer)):
            # add terminal action flag
            if j != len(episode_buffer) - 1:
                episode_buffer[j]["action"] = np.concatenate([episode_buffer[j]["action"], np.array([0])]).astype(np.float32)
            else:
                episode_buffer[j]["action"] = np.concatenate([episode_buffer[j]["action"], np.array([1])]).astype(np.float32)


        # print("Episode length: {}".format(len(episode_buffer)))
        # print("Episode action: {}".format(episode_buffer[0]["action"]))
        # print("Episode state: {}".format(episode_buffer[0]["state"]))

        # input('Press Enter to continue...')


        # save max and min of action and state over all episodes, action and state is vector and get max and min of each element
        # action_max, min, state_max, min is vector of max and min of each element
        if action_max is None:
            action_max = episode_buffer[0]["action"]
            action_min = episode_buffer[0]["action"]
            state_max = episode_buffer[0]["state"]
            state_min = episode_buffer[0]["state"]
            for k in range(len(episode_buffer)):
                action_max = np.maximum(action_max, episode_buffer[k]["action"])
                action_min = np.minimum(action_min, episode_buffer[k]["action"])
                state_max = np.maximum(state_max, episode_buffer[k]["state"])
                state_min = np.minimum(state_min, episode_buffer[k]["state"])
            # print("action_max: {}".format(action_max))
            # print("action_min: {}".format(action_min))
            # print("state_max: {}".format(state_max))
            # print("state_min: {}".format(state_min))
        else:
            for k in range(len(episode_buffer)):
                action_max = np.maximum(action_max, episode_buffer[k]["action"])
                action_min = np.minimum(action_min, episode_buffer[k]["action"])
                state_max = np.maximum(state_max, episode_buffer[k]["state"])
                state_min = np.minimum(state_min, episode_buffer[k]["state"])
            # print("action_max: {}".format(action_max))
            # print("action_min: {}".format(action_min))
            # print("state_max: {}".format(state_max))
            # print("state_min: {}".format(state_min))

        # save episode buffer to npy
        if bag in train_rosbags:
            np.save(os.path.join(args.project_name, "data", "train", "episode_{}.npy".format(train_num)), episode_buffer)
            train_num += 1
        elif bag in val_rosbags:
            np.save(os.path.join(args.project_name, "data", "val", "episode_{}.npy".format(val_num)), episode_buffer)
            val_num += 1


        if i % 5 == 0 and args.gif:
            os.makedirs(os.path.join(args.project_name, "data", "gif"), exist_ok=True)
            # os.makedirs("data/gif", exist_ok=True)
            # episode_buffer is list of dict
            test_images = []
            for data_buffer in episode_buffer:
                test_images.append(data_buffer["image"])
            clip = ImageSequenceClip(test_images, fps=10)
            clip.write_gif(
                os.path.join(args.project_name, "data", "gif", "episode_{}.gif".format(i)),
                fps=10,
                verbose=False,
            )

    # save max and min of action and state over all episodes to yaml
    with open(os.path.join(args.project_name, "data", "max_min.yaml"), "w") as f:
        yaml.dump({
            "action_max": action_max.tolist(),
            "action_min": action_min.tolist(),
            "state_max": state_max.tolist(),
            "state_min": state_min.tolist(),
        }, f)

    if args.normalize:
        # reprocess train and val data to normalize action and state
        # load max and min
        with open(os.path.join(args.project_name, "data", "max_min.yaml"), "r") as f:
            max_min = yaml.load(f, Loader=yaml.FullLoader)
            action_max = np.array(max_min["action_max"]).astype(np.float32)
            action_min = np.array(max_min["action_min"]).astype(np.float32)
            state_max = np.array(max_min["state_max"]).astype(np.float32)
            state_min = np.array(max_min["state_min"]).astype(np.float32)

        # action is [x, y, z, roll, pitch, yaw, gripper command]
        # state is [x, y, z, roll, pitch, yaw, gripper pos]
        # apply normalization only for x,y,z and gripper command and pos
        print("Start reprocessing train rosbags...")
        os.makedirs(os.path.join(args.project_name, "data", "processed_train"), exist_ok=True)
        os.makedirs(os.path.join(args.project_name, "data", "processed_val"), exist_ok=True)
        for i in tqdm(range(train_num)):
            episode_buffer = np.load(os.path.join(args.project_name, "data", "train", "episode_{}.npy".format(i)), allow_pickle=True)
            for j in range(len(episode_buffer)):
                # xyz
                episode_buffer[j]["action"][0:3] = (episode_buffer[j]["action"][0:3] - action_min[0:3]) / (action_max[0:3] - action_min[0:3])
                episode_buffer[j]["state"][0:3] = (episode_buffer[j]["state"][0:3] - state_min[0:3]) / (state_max[0:3] - state_min[0:3])
                # gripper
                episode_buffer[j]["action"][6] = (episode_buffer[j]["action"][6] - action_min[6]) / (action_max[6] - action_min[6])
                episode_buffer[j]["state"][6] = (episode_buffer[j]["state"][6] - state_min[6]) / (state_max[6] - state_min[6])
                # to numpy float32
                episode_buffer[j]["action"] = episode_buffer[j]["action"].astype(np.float32)
                episode_buffer[j]["state"] = episode_buffer[j]["state"].astype(np.float32)
            np.save(os.path.join(args.project_name, "data", "processed_train", "episode_{}.npy".format(i)), episode_buffer)

        print("Start reprocessing val rosbags...")
        for i in tqdm(range(val_num)):
            episode_buffer = np.load(os.path.join(args.project_name, "data", "val", "episode_{}.npy".format(i)), allow_pickle=True)
            for j in range(len(episode_buffer)):
                # xyz
                episode_buffer[j]["action"][0:3] = (episode_buffer[j]["action"][0:3] - action_min[0:3]) / (action_max[0:3] - action_min[0:3])
                episode_buffer[j]["state"][0:3] = (episode_buffer[j]["state"][0:3] - state_min[0:3]) / (state_max[0:3] - state_min[0:3])
                # gripper
                episode_buffer[j]["action"][6] = (episode_buffer[j]["action"][6] - action_min[6]) / (action_max[6] - action_min[6])
                episode_buffer[j]["state"][6] = (episode_buffer[j]["state"][6] - state_min[6]) / (state_max[6] - state_min[6])
                # to numpy float32
                episode_buffer[j]["action"] = episode_buffer[j]["action"].astype(np.float32)
                episode_buffer[j]["state"] = episode_buffer[j]["state"].astype(np.float32)
            np.save(os.path.join(args.project_name, "data", "processed_val", "episode_{}.npy".format(i)), episode_buffer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synchronize messages from a rosbag.')
    parser.add_argument('-pn', "--project_name", type=str, help='The name of project.')
    parser.add_argument('-d', "--rosbag_dir", type=str, help='The name of the input bag file directory.')
    parser.add_argument('-r', "--ratio", type=float, help="train/val ratio. e.g. 0.8", default=0.8)
    parser.add_argument("-t", "--image_tune", action="store_true", default=False)
    parser.add_argument("-n", "--normalize", action="store_true", default=False)
    parser.add_argument("--gif", action="store_true", help="save gif")
    args = parser.parse_args()
    main(args)
