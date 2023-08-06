#!/usr/bin/env python3
import os
import time
import argparse
import yaml
import json
import pprint
from tqdm import tqdm
from easydict import EasyDict as edict
from collections import OrderedDict

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

# for OrderedDict
yaml.add_representer(
    OrderedDict,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items()
    ),
)


def main(args):
    config = FileUtils.get_config_from_project_name(args.project_name)
    rosbags = RosUtils.get_rosbag_abs_paths(args.rosbag_dir)
    print("Found {} rosbags".format(len(rosbags)))

    # get dataset config
    mf_cfg = config.ros.message_filters
    obs_cfg = config.obs
    action_cfg = config.actions

    topic_name_to_obs_name = {value.topic_name: key for key, value in obs_cfg.items()}
    action_topic_name = action_cfg.topic_name

    # create dict of subscriber
    subscribers = OrderedDict()
    for obs in obs_cfg.values():
        subscribers[obs.topic_name] = message_filters.Subscriber(
            obs.topic_name, eval(obs.msg_type)
        )
    subscribers[action_topic_name] = message_filters.Subscriber(
        action_topic_name, eval(action_cfg.msg_type)
    )

    topic_names = [topic_name for topic_name in subscribers.keys()]
    print("Subscribing to topics: {}".format(topic_names))

    ts = message_filters.ApproximateTimeSynchronizer(
        subscribers.values(),
        queue_size=mf_cfg.queue_size,
        slop=mf_cfg.slop,
        allow_headerless=False,
    )

    def callback(*msgs):
        for topic_name, msg in zip(topic_names, msgs):
            if topic_name in topic_name_to_obs_name.keys():
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

                    if args.image_tune:
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
                    else:
                        tunable = HSVBlurCropResolFilter.from_yaml(
                            os.path.join(
                                FileUtils.get_config_folder(args.project_name),
                                "image_filter.yaml",
                            )
                        )
                        data = tunable(data)
                else:
                    data = np.array(msg.data, dtype=np.float32)
                obs_buf[topic_name_to_obs_name[topic_name]].append(data)
            if topic_name == action_topic_name:
                data = np.array(msg.data, dtype=np.float32)
                action_buf.append(data)

    ts.registerCallback(callback)

    # create directory if not exist
    output_dir = os.path.join(FileUtils.get_project_folder(args.project_name), "data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_file = h5py.File(os.path.join(output_dir, "dataset.hdf5"), mode="w")
    data = data_file.create_group("data")
    data.attrs["num_demos"] = len(rosbags)
    data.attrs["num_obs"] = len(topic_name_to_obs_name)
    data.attrs["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data.attrs["config"] = json.dumps(config.dataset, indent=4)
    # data.attrs["env_args"] = json.dumps(config.dataset.data.env_args, indent=4) # TODO

    action_min = None
    action_max = None

    obs_max_buf = OrderedDict()
    obs_min_buf = OrderedDict()
    obs_scale_buf = OrderedDict()
    obs_bias_buf = OrderedDict()

    for obs_name in topic_name_to_obs_name.values():
        # only for FloatVectorModality
        if obs_cfg[obs_name].modality == "FloatVectorModality":
            obs_max_buf[obs_name] = None
            obs_min_buf[obs_name] = None
            obs_scale_buf[obs_name] = None
            obs_bias_buf[obs_name] = None

    for i, bag in enumerate(tqdm(rosbags)):
        # obs_buf = dict()
        obs_buf = OrderedDict()
        action_buf = []
        for obs_name in topic_name_to_obs_name.values():
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

        assert len(obs_data) == len(action_data)
        demo.attrs["num_samples"] = len(action_data)
        data_file.flush()

        if i % 5 == 0 and args.gif:
            clip = ImageSequenceClip(obs_buf["image"], fps=10)
            clip.write_gif(
                os.path.join(output_dir, "demo_{}.gif".format(i)),
                fps=10,
                verbose=False,
            )

    # scale actions in data_file["data/demo_{i}/actions".format(i)] to [-1, 1]
    action_min = np.array(action_min)
    action_max = np.array(action_max)
    action_scale = (action_max - action_min) / 2
    action_bias = (action_max + action_min) / 2

    data.attrs["action_scale"] = action_scale
    data.attrs["action_bias"] = action_bias
    data.attrs["action_min"] = action_min
    data.attrs["action_max"] = action_max

    for obs_name in obs_max_buf.keys():
        obs_max = obs_max_buf[obs_name].astype(np.float32)
        obs_min = obs_min_buf[obs_name].astype(np.float32)
        obs_scale = (obs_max - obs_min) / 2
        obs_bias = (obs_max + obs_min) / 2
        obs_scale_buf[obs_name] = obs_scale
        obs_bias_buf[obs_name] = obs_bias

    for obs_name in obs_max_buf.keys():
        data.attrs["obs_max/{}".format(obs_name)] = obs_max_buf[obs_name]
        data.attrs["obs_min/{}".format(obs_name)] = obs_min_buf[obs_name]
        data.attrs["obs_scale/{}".format(obs_name)] = obs_scale_buf[obs_name]
        data.attrs["obs_bias/{}".format(obs_name)] = obs_bias_buf[obs_name]

    if args.normalize:
        for i in range(len(rosbags)):
            # scale actions to [-1, 1]
            demo = data["demo_{}".format(i)]
            action = demo["actions"]
            action_scaled = (action - action_bias) / action_scale
            demo["actions"][:] = action_scaled

            # scale observations to [-1, 1] only for FloatVectorModality
            for obs_name in obs_max_buf.keys():
                obs = demo["obs"]
                obs_scaled = (obs[obs_name] - obs_bias_buf[obs_name]) / obs_scale_buf[
                    obs_name
                ]
                demo["obs/{}".format(obs_name)][:] = obs_scaled
                next_obs = demo["next_obs"]
                next_obs_scaled = (
                    next_obs[obs_name] - obs_bias_buf[obs_name]
                ) / obs_scale_buf[obs_name]
                demo["next_obs/{}".format(obs_name)][:] = next_obs_scaled

    yaml_data = OrderedDict()
    yaml_data["actions"] = OrderedDict()
    yaml_data["obs"] = OrderedDict()

    yaml_data["actions"]["max"] = action_max.tolist()
    yaml_data["actions"]["min"] = action_min.tolist()
    yaml_data["actions"]["scale"] = action_scale.tolist()
    yaml_data["actions"]["bias"] = action_bias.tolist()

    # yaml_data["action_max"] = action_max.tolist()
    # yaml_data["action_min"] = action_min.tolist()
    # yaml_data["action_scale"] = action_scale.tolist()
    # yaml_data["action_bias"] = action_bias.tolist()

    for obs_name in obs_max_buf.keys():
        yaml_data["obs"][obs_name] = OrderedDict()
        yaml_data["obs"][obs_name]["max"] = obs_max_buf[obs_name].tolist()
        yaml_data["obs"][obs_name]["min"] = obs_min_buf[obs_name].tolist()
        yaml_data["obs"][obs_name]["scale"] = obs_scale_buf[obs_name].tolist()
        yaml_data["obs"][obs_name]["bias"] = obs_bias_buf[obs_name].tolist()

        # yaml_data[obs_name] = OrderedDict()
        # yaml_data[obs_name]["obs_max"] = obs_max_buf[obs_name].tolist()
        # yaml_data[obs_name]["obs_min"] = obs_min_buf[obs_name].tolist()
        # yaml_data[obs_name]["obs_scale"] = obs_scale_buf[obs_name].tolist()
        # yaml_data[obs_name]["obs_bias"] = obs_bias_buf[obs_name].tolist()

    yaml_file = open(
        os.path.join(FileUtils.get_config_folder(args.project_name), "normalize.yaml"),
        "w",
    )
    yaml.dump(yaml_data, yaml_file, default_flow_style=None)
    yaml_file.close()

    data_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-d", "--rosbag_dir", type=str, default="/tmp/dataset")
    parser.add_argument("-n", "--normalize", action="store_true", default=False)
    parser.add_argument("-t", "--image_tune", action="store_true", default=False)
    parser.add_argument("--gif", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
