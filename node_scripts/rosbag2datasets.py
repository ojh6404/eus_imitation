#!/usr/bin/env python3
import os
import time
import argparse
import yaml
import json

import numpy as np
import cv2
from tqdm import tqdm
from easydict import EasyDict as dict
import h5py
from moviepy.editor import ImageSequenceClip

import rospy
import rosbag
import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage, Image, JointState
from eus_imitation.msg import Float32MultiArrayStamped
from eus_imitation.utils.rosbag_utils import RosbagUtils, PatchTimer

# for no roscore
rospy.Time = PatchTimer


def main(args):
    with open(args.config, "r") as f:
        print("Processing config...")
        config = dict(yaml.safe_load(f))
        print(json.dumps(config, indent=4))

    rosbags = RosbagUtils.get_rosbag_abs_paths(args.rosbag_dir)
    print("Found {} rosbags".format(len(rosbags)))

    mf_cfg = config.dataset.rosbag.message_filters
    obs_cfg = config.dataset.data.obs
    action_cfg = config.dataset.data.actions

    def get_topic_to_obs(config):
        obs_cfg = config.dataset.data.obs
        reversed_dict = {value.topic_name: key for key, value in obs_cfg.items()}
        return reversed_dict

    def get_topic_to_action(config):
        action_cfg = config.dataset.data.actions
        reversed_dict = {value.topic_name: key for key, value in action_cfg.items()}
        return reversed_dict

    topic_to_obs = get_topic_to_obs(config)
    topic_to_action = get_topic_to_action(config)

    subscribers = dict()
    for obs in obs_cfg.values():
        subscribers[obs.topic_name] = message_filters.Subscriber(
            obs.topic_name, eval(obs.msg_type)
        )

    for action in action_cfg.values():
        subscribers[action.topic_name] = message_filters.Subscriber(
            action.topic_name, eval(action.msg_type)
        )

    topic_names = [topic_name for topic_name in subscribers.keys()]

    ts = message_filters.ApproximateTimeSynchronizer(
        subscribers.values(),
        queue_size=mf_cfg.queue_size,
        slop=mf_cfg.slop,
        allow_headerless=False,
    )

    def callback(*msgs):
        for topic_name, msg in zip(topic_names, msgs):
            if topic_name in topic_to_obs.keys():
                if "Image" in msg._type:
                    if "Compressed" in msg._type:
                        data = cv2.cvtColor(
                            cv2.imdecode(
                                np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR
                            ),
                            cv2.COLOR_BGR2RGB,
                        )
                    else:
                        data = CvBridge().imgmsg_to_cv2(msg, "rgb8")
                    crop_and_resize = obs_cfg[
                        topic_to_obs[topic_name]
                    ].image_tune.crop_and_resize
                    if crop_and_resize:
                        crop_x_offset = obs_cfg[
                            topic_to_obs[topic_name]
                        ].image_tune.crop_x_offset
                        crop_y_offset = obs_cfg[
                            topic_to_obs[topic_name]
                        ].image_tune.crop_y_offset
                        crop_height = obs_cfg[
                            topic_to_obs[topic_name]
                        ].image_tune.crop_height
                        crop_width = obs_cfg[
                            topic_to_obs[topic_name]
                        ].image_tune.crop_width
                        data = data[
                            crop_y_offset : crop_y_offset + crop_height,
                            crop_x_offset : crop_x_offset + crop_width,
                        ]
                        height = obs_cfg[topic_to_obs[topic_name]].image_tune.height
                        width = obs_cfg[topic_to_obs[topic_name]].image_tune.width
                        data = cv2.resize(
                            data, (width, height), interpolation=cv2.INTER_AREA
                        )
                else:
                    data = np.array(msg.data, dtype=np.float32)
                obs_buf[topic_to_obs[topic_name]].append(data)
            if topic_name in topic_to_action.keys():
                data = np.array(msg.data, dtype=np.float32)
                action_buf[topic_to_action[topic_name]].append(data)

    ts.registerCallback(callback)

    # create directory if not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data_file = h5py.File(os.path.join(args.output_dir, "dataset.hdf5"), mode="w")
    data = data_file.create_group("data")
    data.attrs["num_demos"] = len(rosbags)
    data.attrs["num_obs"] = len(topic_to_obs)
    data.attrs["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data.attrs["config"] = json.dumps(config.dataset, indent=4)
    data.attrs["env_args"] = json.dumps(config.dataset.data.env_args, indent=4)

    action_min = None
    action_max = None

    obs_max_buf = dict()
    obs_min_buf = dict()
    obs_scale_buf = dict()
    obs_bias_buf = dict()

    for obs_name in topic_to_obs.values():
        # only for low_dim
        if obs_cfg[obs_name].modality == "low_dim":
            obs_max_buf[obs_name] = None
            obs_min_buf[obs_name] = None
            obs_scale_buf[obs_name] = None
            obs_bias_buf[obs_name] = None

    for i, bag in enumerate(tqdm(rosbags)):
        obs_buf = dict()
        action_buf = dict()
        for obs_name in topic_to_obs.values():
            obs_buf[obs_name] = []
        for action_name in topic_to_action.values():
            action_buf[action_name] = []

        demo = data.create_group("demo_{}".format(i))

        # ts = message_filters.ApproximateTimeSynchronizer(
        #     subscribers.values(),
        #     queue_size=mf_cfg.queue_size,
        #     slop=mf_cfg.slop,
        #     allow_headerless=False,
        # )
        # ts.registerCallback(callback)

        bag_reader = rosbag.Bag(bag, skip_index=True)
        for message_idx, (topic, msg, t) in enumerate(
            bag_reader.read_messages(topics=topic_names)
        ):
            subscriber = subscribers.get(topic)
            if subscriber:
                subscriber.signalMessage(msg)

        for obs_name, obs_data in obs_buf.items():
            obs_data = np.array(obs_data)
            next_obs_data = np.concatenate(
                [obs_data[1:], obs_data[-1:]], axis=0
            )  # repeat last obs
            demo.create_dataset(
                "obs/{}".format(obs_name),
                data=obs_data,
                dtype=obs_data.dtype,
            )
            demo.create_dataset(
                "next_obs/{}".format(obs_name),
                data=next_obs_data,
                dtype=next_obs_data.dtype,
            )

            if obs_name in obs_max_buf.keys():
                if obs_max_buf[obs_name] is None:
                    obs_max_buf[obs_name] = np.max(obs_data, axis=0)
                    obs_min_buf[obs_name] = np.min(obs_data, axis=0)
                else:
                    obs_max_buf[obs_name] = np.maximum(
                        obs_max_buf[obs_name], np.max(obs_data, axis=0)
                    ).astype(np.float32)
                    obs_min_buf[obs_name] = np.minimum(
                        obs_min_buf[obs_name], np.min(obs_data, axis=0)
                    ).astype(np.float32)

        for action_name, action_data in action_buf.items():
            action_data = np.array(action_data)
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

        assert len(obs_data) == len(action_data)

        demo.attrs["num_samples"] = len(action_data)
        print(demo.attrs["num_samples"])
        data_file.flush()

        if i % 5 == 0 and args.gif:
            clip = ImageSequenceClip(obs_buf["image"], fps=10)
            clip.write_gif(
                os.path.join(args.output_dir, "demo_{}.gif".format(i)),
                fps=10,
                verbose=False,
            )

    # scale actions in data_file["data/demo_{i}/actions".format(i)] to [-1, 1]
    action_min = np.array(action_min)
    action_max = np.array(action_max)
    action_scale = (action_max - action_min) / 2
    action_bias = (action_max + action_min) / 2
    for i in range(len(rosbags)):
        # scale actions
        demo = data["demo_{}".format(i)]
        actions = demo["actions"]
        actions_scaled = (actions - action_bias) / action_scale
        demo["actions"][:] = actions_scaled

        # scale observations in obs_max_buf and obs_min_buf to [-1, 1]

        for obs_name in obs_max_buf.keys():
            obs_max = obs_max_buf[obs_name].astype(np.float32)
            obs_min = obs_min_buf[obs_name].astype(np.float32)
            obs_scale = (obs_max - obs_min) / 2
            obs_bias = (obs_max + obs_min) / 2
            obs_scale_buf[obs_name] = obs_scale
            obs_bias_buf[obs_name] = obs_bias
            obs = demo["obs/{}".format(obs_name)]
            obs_scaled = (obs - obs_bias) / obs_scale
            demo["obs/{}".format(obs_name)][:] = obs_scaled
            next_obs = demo["next_obs/{}".format(obs_name)]
            next_obs_scaled = (next_obs - obs_bias) / obs_scale
            demo["next_obs/{}".format(obs_name)][:] = next_obs_scaled

    data.attrs["action_scale"] = action_scale
    data.attrs["action_bias"] = action_bias
    data.attrs["action_min"] = action_min
    data.attrs["action_max"] = action_max

    for obs_name in obs_max_buf.keys():
        data.attrs["obs_max/{}".format(obs_name)] = obs_max_buf[obs_name]
        data.attrs["obs_min/{}".format(obs_name)] = obs_min_buf[obs_name]
        data.attrs["obs_scale/{}".format(obs_name)] = obs_scale_buf[obs_name]
        data.attrs["obs_bias/{}".format(obs_name)] = obs_bias_buf[obs_name]

    data_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--rosbag_dir", type=str, default="/tmp/dataset")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--gif", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
