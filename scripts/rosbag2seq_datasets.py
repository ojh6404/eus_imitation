#!/usr/bin/env python3

import os
import time
from absl import app, flags, logging
from omegaconf import OmegaConf
import pprint
from tqdm import tqdm
import numpy as np
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

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", None, "Name of the project to load config from.")
flags.DEFINE_string("rosbag_dir", None, "Name of the project to load config from.")
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 128, "Batch size for finetuning.")
flags.DEFINE_bool(
    "image_tune",
    False,
    "Whether to tune image filter.",
)
flags.DEFINE_string("vis_image", "head_image", "Image key to visualize.")
flags.DEFINE_bool("gif", False, "Whether to save gif.")
flags.DEFINE_float("ratio", 0.1, "Ratio of validation data.")

# for no roscore
rospy.Time = RosUtils.PatchTimer


def main(_):
    config = FileUtils.get_config_from_project_name(FLAGS.project_name)
    rosbags = RosUtils.get_rosbag_abs_paths(FLAGS.rosbag_dir)
    logging.info("Found {} rosbags".format(len(rosbags)))

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

    if FLAGS.vis_image:
        assert (
            FLAGS.vis_image in obs_keys
        ), "image key not found in obs_keys to write gif"

    subscribers = dict()
    for topic, msg_type in zip(topics, msg_types):
        subscribers[topic] = message_filters.Subscriber(topic, msg_type)

    img_bridge = CvBridge()
    ts = message_filters.ApproximateTimeSynchronizer(
        subscribers.values(),
        queue_size=mf_cfg.queue_size,
        slop=mf_cfg.slop,
        allow_headerless=False,
    )

    logging.info("Observation and Action keys: {}".format(obs_keys + ["action"]))
    logging.info("Subscribing to topics: {}".format(topics))

    def callback(*msgs):
        for topic, msg in zip(topics, msgs):
            if topic == action_topic:  # action
                data = np.array(msg.data).astype(np.float32)
                action_buf.append(data)
            else:
                if "Image" in msg._type:
                    if "Compressed" in msg._type:
                        data = img_bridge.compressed_imgmsg_to_cv2(msg, "rgb8").astype(
                            np.uint8
                        )
                    else:
                        data = img_bridge.imgmsg_to_cv2(msg, "rgb8").astype(np.uint8)
                    if FLAGS.image_tune:  # tuning image with first data
                        tunable = HSVBlurCropResolFilter.from_image(data)
                        logging.info("Press q to finish tuning")
                        tunable.launch_window()
                        tunable.start_tuning(data)
                        pprint.pprint(tunable.export_dict())
                        tunable.dump_yaml(
                            os.path.join(
                                FileUtils.get_config_folder(FLAGS.project_name),
                                "image_filter.yaml",
                            )
                        )
                        hdf5_file.close()
                        exit(0)
                    else:  # load from yaml and proess image
                        assert os.path.exists(
                            os.path.join(
                                FileUtils.get_config_folder(FLAGS.project_name),
                                "image_filter.yaml",
                            )
                        ), "image_filter.yaml not found. Please run with --image_tune first."
                        tunable = HSVBlurCropResolFilter.from_yaml(
                            os.path.join(
                                FileUtils.get_config_folder(FLAGS.project_name),
                                "image_filter.yaml",
                            )
                        )
                        data = tunable(data)
                else:
                    data = np.array(msg.data).astype(np.float32)
                obs_buf[topics_to_keys[topic]].append(data)

    ts.registerCallback(callback)

    # create directory if not exist
    output_dir = FileUtils.get_data_folder(FLAGS.project_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create train dataset
    hdf5_file = h5py.File(os.path.join(output_dir, "dataset.hdf5"), mode="w")
    demo_group = hdf5_file.create_group("data")
    demo_group.attrs["num_demos"] = len(rosbags)
    demo_group.attrs["num_obs"] = len(obs_keys)
    demo_group.attrs["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    demo_group.attrs["hz"] = config.ros.rate

    action_min = None
    action_max = None

    obs_max_buf = dict()
    obs_min_buf = dict()

    for obs_key in obs_keys:
        # consider min max for only float vector
        if obs_cfg[obs_key].modality == "FloatVectorModality":
            obs_max_buf[obs_key] = None
            obs_min_buf[obs_key] = None

    logging.info("Processing rosbags...")
    for i, bag in enumerate(tqdm(rosbags)):
        obs_buf = dict()  # {obs_key: [obs_data]}, obs_data is numpy array
        action_buf = []  # [action_data], action_data is numpy array
        for obs_key in obs_keys:
            obs_buf[obs_key] = []

        demo = demo_group.create_group("demo_{}".format(i))
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

        assert len(obs_data) == len(
            action_data
        )  # obs_data and action_data should have same length
        demo.attrs["num_samples"] = len(action_data)
        hdf5_file.flush()

        os.makedirs(os.path.join(output_dir, "gif"), exist_ok=True)
        if i % 5 == 0 and FLAGS.gif:
            clip = ImageSequenceClip(obs_buf[FLAGS.vis_image], fps=10)  # TODO image key
            clip.write_gif(
                os.path.join(output_dir, "gif", "demo_{}.gif".format(i)),
                fps=10,
                verbose=False,
            )

    # train val split
    demos = list(demo_group.keys())
    num_demos = len(demos)
    num_val = int(FLAGS.ratio * num_demos)
    mask = np.zeros(num_demos)
    mask[:num_val] = 1
    np.random.shuffle(mask)
    mask = mask.astype(int)
    train_ids = (1 - mask).nonzero()[0]
    valid_ids = mask.nonzero()[0]
    train_keys = [demos[i] for i in train_ids]
    valid_keys = [demos[i] for i in valid_ids]

    # create mask
    mask_group = hdf5_file.create_group("mask")
    mask_group.create_dataset("train", data=np.array(train_keys, dtype="S"))
    mask_group.create_dataset("valid", data=np.array(valid_keys, dtype="S"))

    # meta data
    hdf5_file.attrs["language_instruction"] = config.task.language_instruction

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
    OmegaConf.save(
        normalize_cfg,
        os.path.join(FileUtils.get_config_folder(FLAGS.project_name), "normalize.yaml"),
    )

    hdf5_file.close()


if __name__ == "__main__":
    app.run(main)
