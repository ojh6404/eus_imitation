#!/usr/bin/env python3
import os
import time
import argparse
import pprint
from tqdm import tqdm

import numpy as np
import cv2
import h5py
from moviepy.editor import ImageSequenceClip
from tunable_filter.composite_zoo import HSVBlurCropResolFilter

import rospy
import rosbag
from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage, Image
import imitator.utils.ros_utils as RosUtils
import imitator.utils.file_utils as FileUtils

# for no roscore
rospy.Time = RosUtils.PatchTimer

def main(args):
    config = FileUtils.get_config_from_project_name(args.project_name)
    rosbags = RosUtils.get_rosbag_abs_paths(args.rosbag_dir)
    print("Found {} rosbags".format(len(rosbags)))

    # get dataset config
    topic_name = config.obs[args.obs_key]["topic_name"]

    # create dict of subscriber
    print("Subscribing to obs {}'s topic: {}".format(args.obs_key, topic_name))

    # create directory if not exist
    output_dir = os.path.join(FileUtils.get_project_folder(args.project_name), "data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_file = h5py.File(os.path.join(output_dir, args.obs_key + "_dataset.hdf5"), mode="w")
    data = data_file.create_group("data")
    data.attrs["num_demos"] = len(rosbags)
    data.attrs["num_obs"] = 1
    data.attrs["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    img_bridge = CvBridge()

    for i, bag in enumerate(tqdm(rosbags)):
        obs_buf = []

        demo = data.create_group("demo_{}".format(i))
        bag_reader = rosbag.Bag(bag, skip_index=True)

        for topic, msg, t in bag_reader.read_messages(topics=[topic_name]):
            if config.obs[args.obs_key].msg_type == "CompressedImage":
                obs_data =  img_bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
            elif config.obs[args.obs_key].msg_type == "Image":
                obs_data = img_bridge.imgmsg_to_cv2(msg, "rgb8")
            else:
                raise NotImplementedError

            if args.image_tune:
                tunable = HSVBlurCropResolFilter.from_image(obs_data)
                print("Press q to finish tuning")
                tunable.launch_window()
                tunable.start_tuning(obs_data)
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
                obs_data = tunable(obs_data)
            obs_buf.append(obs_data)

        obs_data = np.array(obs_buf)
        demo.create_dataset(
            "obs/{}".format(args.obs_key),
            data=obs_data,
            dtype=obs_data.dtype,
        )

        demo.attrs["num_samples"] = len(obs_data)

        data_file.flush()

        if i % 5 == 0 and args.gif:
            clip = ImageSequenceClip(obs_buf, fps=30)
            clip.write_gif(
                os.path.join(output_dir, "demo_{}.gif".format(i)),
                fps=10,
                verbose=False,
            )

    lengths = []
    for i in range(len(rosbags)):
        lengths.append(data["demo_{}".format(i)].attrs["num_samples"])

    total_length = sum(lengths)
    print("Total number of {}".format(total_length))
    data_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-d", "--rosbag_dir", type=str, default="/tmp/dataset")
    parser.add_argument("-obs", "--obs_key", type=str, default="image")
    parser.add_argument("-t", "--image_tune", action="store_true", default=False)
    parser.add_argument("--gif", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
