#!/usr/bin/env python3
import os
import time
import argparse
from omegaconf import OmegaConf
import pprint
from tqdm import tqdm

import numpy as np
import cv2
import h5py
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

import mediapipe as mp

# for no roscore
rospy.Time = RosUtils.PatchTimer

def wrist2pos(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    # get wrist2's position
    wrist2_x = min(int(landmarks.landmark[1].x * image_width), image_width - 1)
    wrist2_y = min(int(landmarks.landmark[1].y * image_height), image_height - 1)

    # get wrist5's position
    wrist5_x = min(int(landmarks.landmark[5].x * image_width), image_width - 1)
    wrist5_y = min(int(landmarks.landmark[5].y * image_height), image_height - 1)

    # center of wrist2 and wrist5
    wrist2_x = (wrist2_x + wrist5_x) / 2
    wrist2_y = (wrist2_y + wrist5_y) / 2

    return [wrist2_x, wrist2_y]

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    # also calc center
    center_x = x + w / 2
    center_y = y + h / 2

    return [x, y, x + w, y + h], [center_x, center_y]



def main(args):
    """
    only process obs for human demo
    """
    config = FileUtils.get_config_from_project_name(args.project_name)
    rosbags = RosUtils.get_rosbag_abs_paths(args.rosbag_dir)
    print("Found {} rosbags".format(len(rosbags)))

    # get dataset config
    mf_cfg = config.ros.message_filters

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.1,
        # min_tracking_confidence=0.3,
    )

    # obs_keys = list(obs_cfg.keys())
    obs_keys = ["image"]
    obs_topics = ["/kinect_head/rgb/image_rect_color/compressed"]
    obs_msg_types = [CompressedImage]

    topics_to_keys = dict(zip(obs_topics, obs_keys))
    topics = obs_topics
    msg_types = obs_msg_types

    subscribers = dict()
    for topic, msg_type in zip(topics, msg_types):
        subscribers[topic] = message_filters.Subscriber(topic, msg_type)

    img_bridge = CvBridge()
    ts = message_filters.ApproximateTimeSynchronizer(subscribers.values(), queue_size=mf_cfg.queue_size, slop=mf_cfg.slop, allow_headerless=False)


    print("Observation and Action keys: {}".format(obs_keys + ["action"]))
    print("Subscribing to topics: {}".format(topics))


    def callback(*msgs):
        for topic, msg in zip(topics, msgs):
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
                    data_file.close()
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

                    results = hands.process(data)
                    if results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            # cx, cy = wrist2pos(data, hand_landmarks)
                            bbox, center = calc_bounding_rect(data, hand_landmarks)
                            cx, cy = center
                    # draw bbox and center
                    # for debug
                        # cv2.rectangle(data, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        # cv2.circle(data, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                        # cv2.imshow("image", data)
                        # cv2.waitKey(1)
                        obs_buf[topics_to_keys[topic]].append(data)
                        # obs_buf["hand_pos"].append(np.array([cx, cy]).astype(np.float32))
                        action_buf.append(np.array([cx, cy]).astype(np.float32))
                    else:
                        print("no hand")
            else:
                data = np.array(msg.data).astype(np.float32)
            # obs_buf[topics_to_keys[topic]].append(data)


    ts.registerCallback(callback)

    # create directory if not exist
    output_dir = os.path.join(FileUtils.get_project_folder(args.project_name), "data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_file = h5py.File(os.path.join(output_dir, "human_demo_dataset.hdf5"), mode="w")
    data = data_file.create_group("data")
    data.attrs["num_demos"] = len(rosbags)
    data.attrs["num_obs"] = len(obs_keys)
    data.attrs["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data.attrs["hz"] = config.ros.rate


    action_min = None
    action_max = None

    obs_max_buf = dict()
    obs_min_buf = dict()

    print("Processing rosbags...")
    for i, bag in enumerate(tqdm(rosbags)):
        obs_buf = dict() # {obs_key: [obs_data]}, obs_data is numpy array
        action_buf = list()
        for obs_key in obs_keys:
            obs_buf[obs_key] = []

        # obs_buf["hand_pos"] = []

        demo = data.create_group("demo_{}".format(i))
        bag_reader = rosbag.Bag(bag, skip_index=True)

        for message_idx, (topic, msg, t) in enumerate(
            bag_reader.read_messages(topics=topics)
        ):
            subscriber = subscribers.get(topic)
            if subscriber:
                subscriber.signalMessage(msg)

        for obs_key, obs_data in obs_buf.items():
            obs_data = np.array(obs_data)
            next_obs_data = np.concatenate(
                [obs_data[1:], obs_data[-1:]], axis=0
            )  # repeat last obs
            demo.create_dataset(
                "obs/{}".format(obs_key),
                data=obs_data,
                dtype=obs_data.dtype,
            )
            demo.create_dataset(
                "next_obs/{}".format(obs_key),
                data=next_obs_data,
                dtype=next_obs_data.dtype,
            )

            if obs_key in obs_max_buf.keys():
                if obs_max_buf[obs_key] is None:
                    obs_max_buf[obs_key] = np.max(obs_data, axis=0)
                    obs_min_buf[obs_key] = np.min(obs_data, axis=0)
                else:
                    obs_max_buf[obs_key] = np.maximum(
                        obs_max_buf[obs_key], np.max(obs_data, axis=0)
                    )
                    obs_min_buf[obs_key] = np.minimum(
                        obs_min_buf[obs_key], np.min(obs_data, axis=0)
                    )

        # action_dim = 4
        # action_data = np.zeros((len(obs_data), action_dim), dtype=np.float32) # dummy action
        action_data = np.array(action_buf).astype(np.float32)
        demo.create_dataset(
            "actions",
            data=action_data,
            dtype=action_data.dtype,
        )

        if action_min is None:
            action_min = np.min(action_data, axis=0)
            action_max = np.max(action_data, axis=0)
        else:
            action_min = np.minimum(action_min, np.min(action_data, axis=0))
            action_max = np.maximum(action_max, np.max(action_data, axis=0))


        assert len(obs_data) == len(action_data) # obs_data and action_data should have same length
        demo.attrs["num_samples"] = len(action_data)
        data_file.flush()

        os.makedirs(os.path.join(output_dir, "gif"), exist_ok=True)
        if i % 5 == 0 and args.gif:
            clip = ImageSequenceClip(obs_buf["image"], fps=10) # TODO image key
            clip.write_gif(
                os.path.join(output_dir, "gif", "demo_{}.gif".format(i)),
                fps=10,
                verbose=False,
            )



    normalize_data = dict()
    normalize_data["actions"] = dict()
    normalize_data["obs"] = dict()

    normalize_data["actions"]["min"] = action_min.tolist()
    normalize_data["actions"]["max"] = action_max.tolist()

    for obs_key in obs_max_buf.keys():
        normalize_data["obs"][obs_key] = dict()
        normalize_data["obs"][obs_key]["max"] = obs_max_buf[obs_key].tolist()
        normalize_data["obs"][obs_key]["min"] = obs_min_buf[obs_key].tolist()

    normalize_cfg = OmegaConf.create(normalize_data)
    OmegaConf.save(normalize_cfg, os.path.join(FileUtils.get_config_folder(args.project_name), "normalize.yaml"))

    data_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-d", "--rosbag_dir", type=str, default="/tmp/dataset")
    parser.add_argument("-t", "--image_tune", action="store_true", default=False)
    parser.add_argument("--gif", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
