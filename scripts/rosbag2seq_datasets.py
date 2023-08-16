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

# for no roscore
rospy.Time = RosUtils.PatchTimer

def main(args):
    config = FileUtils.get_config_from_project_name(args.project_name)
    rosbags = RosUtils.get_rosbag_abs_paths(args.rosbag_dir)
    print("Found {} rosbags".format(len(rosbags)))

    # get dataset config
    mf_cfg = config.ros.message_filters
    obs_cfg = config.obs
    action_cfg = config.actions


    obs_keys = list(obs_cfg.keys())
    obs_topics = [obs.topic_name for obs in obs_cfg.values()]
    obs_msg_types = [eval(obs.msg_type) for obs in obs_cfg.values()]
    action_topic = action_cfg.topic_name
    action_msg_type = eval(action_cfg.msg_type)

    topics_to_keys = dict(zip(obs_topics + [action_topic], obs_keys + ["action"]))
    topics = obs_topics + [action_topic]
    msg_types = obs_msg_types + [action_msg_type]

    subscribers = dict()
    for topic, msg_type in zip(topics, msg_types):
        subscribers[topic] = message_filters.Subscriber(topic, msg_type)

    img_bridge = CvBridge()
    ts = message_filters.ApproximateTimeSynchronizer(subscribers.values(), queue_size=mf_cfg.queue_size, slop=mf_cfg.slop, allow_headerless=False)


    print("Observation and Action keys: {}".format(obs_keys + ["action"]))
    print("Subscribing to topics: {}".format(topics))


    def callback(*msgs):
        for topic, msg in zip(topics, msgs):
            if topic == action_topic: # action
                data = np.array(msg.data).astype(np.float32)
                action_buf.append(data)
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
                else:
                    data = np.array(msg.data).astype(np.float32)
                obs_buf[topics_to_keys[topic]].append(data)

    ts.registerCallback(callback)

    # create directory if not exist
    output_dir = os.path.join(FileUtils.get_project_folder(args.project_name), "data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_file = h5py.File(os.path.join(output_dir, "dataset.hdf5"), mode="w")
    data = data_file.create_group("data")
    data.attrs["num_demos"] = len(rosbags)
    data.attrs["num_obs"] = len(obs_keys)
    data.attrs["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data.attrs["hz"] = config.ros.rate
    # data.attrs["config"] = json.dumps(config.dataset, indent=4)
    # data.attrs["env_args"] = json.dumps(config.dataset.data.env_args, indent=4) # TODO

    action_min = None
    action_max = None

    obs_max_buf = dict()
    obs_min_buf = dict()

    for obs_key in obs_keys:
        # consider min max for only float vector
        if obs_cfg[obs_key].modality == "FloatVectorModality":
            obs_max_buf[obs_key] = None
            obs_min_buf[obs_key] = None

    print("Processing rosbags...")
    for i, bag in enumerate(tqdm(rosbags)):
        obs_buf = dict() # {obs_key: [obs_data]}, obs_data is numpy array
        action_buf = [] # [action_data], action_data is numpy array
        for obs_key in obs_keys:
            obs_buf[obs_key] = []

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

        action_data = np.array(action_buf)
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

    # after processing all rosbags
    # save data for normalization
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
