#!/usr/bin/env python3
import sys
import os
import time
import rospy
import rosbag
import message_filters
import h5py

import numpy as np
import argparse
import yaml
from easydict import EasyDict as dict
import json
from collections import defaultdict
from cv_bridge import CvBridge

import cv2
from moviepy.editor import ImageSequenceClip

from tqdm import tqdm


from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.msg import JointState
from eus_imitation.msg import Float32MultiArrayStamped
from eus_imitation.utils.rosbag_utils import RosbagUtils, PatchTimer


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
        action_cfg = config.dataset.data.actions
        topic_obs_dict = {value.topic_name: key for key, value in obs_cfg.items()}
        topic_action_dict = {value.topic_name: key for key, value in action_cfg.items()}
        topic_to_obs = {**topic_obs_dict, **topic_action_dict}
        return topic_to_obs

    topic_to_obs = get_topic_to_obs(config)

    print("testeteste")
    print(topic_to_obs)

    obs_topic_names = [obs.topic_name for obs in obs_cfg.values()]
    print("obs_topic_names: {}".format(obs_topic_names))
    action_topic_names = [action.topic_name for action in action_cfg.values()]
    print("action_topic_names: {}".format(action_topic_names))
    topic_names = list(set(obs_topic_names + action_topic_names))

    subscribers = dict()
    # for obs in obs_cfg.values():
    #     subscribers[obs.topic_name] = message_filters.Subscriber(
    #         obs.topic_name, eval(obs.msg_type)
    #     )
    for topic_name in topic_names:
        subscribers[topic_name] = message_filters.Subscriber(
            topic_name, eval(topic_to_obs[topic_name])
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
            if topic_name in topic_names:
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

    ts.registerCallback(callback)

    data_file = h5py.File(os.path.join(args.output_dir, "dataset.hdf5"), mode="w")
    data = data_file.create_group("data")

    for i, bag in enumerate(tqdm(rosbags)):
        obs_buf = dict()
        for obs_name in topic_to_obs.values():
            obs_buf[obs_name] = []

        demo = data.create_group("demo_{}".format(i))

        bag_reader = rosbag.Bag(bag, skip_index=True)
        for message_idx, (topic, msg, t) in enumerate(
            bag_reader.read_messages(topics=topic_names)
        ):
            subscriber = subscribers.get(topic)
            if subscriber:
                subscriber.signalMessage(msg)

        for obs_name, obs_data in obs_buf.items():
            obs_data = np.array(obs_data)
            demo.create_dataset(
                obs_name,
                data=obs_data,
                dtype=obs_data.dtype,
            )

        data_file.flush()

        if i % 10 == 0 and args.gif:
            clip = ImageSequenceClip(obs_buf["image"], fps=30)
            clip.write_gif(
                os.path.join(args.output_dir, "demo_{}.gif".format(i)),
                fps=30,
                verbose=False,
            )

    data_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--rosbag_dir", type=str, default="rosbag.bag")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    main(args)
