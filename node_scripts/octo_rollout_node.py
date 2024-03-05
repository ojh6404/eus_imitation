#!/usr/bin/env python

import os
import time
from typing import Any, Dict
from collections import OrderedDict

import numpy as np
from tunable_filter.composite_zoo import HSVBlurCropResolFilter
import jax

import rospy
import message_filters
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from imitator.utils import file_utils as FileUtils
from eus_imitation.msg import Float32MultiArrayStamped

from octo.model.octo_model import OctoModel


class OctoROSRollout(object):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(OctoROSRollout, self).__init__()
        print("loading model...")
        model = OctoModel.load_pretrained(cfg.checkpoint_path, cfg.checkpoint_step)
        print("loaded model.")
        self.act_stats = model.dataset_statistics["action"]
        self.proprio_stats = model.dataset_statistics["proprio"]
        self.policy_fn = jax.jit(model.sample_actions)
        self.model = model

        self.instruction = cfg.task.language_instruction
        self.task = self.model.create_tasks(texts=[self.instruction])

        self.index = 0
        self.update_interval = 1

        self.cfg = cfg
        self.obs_keys = list(cfg.obs.keys())
        self.image_obs = [
            obs_key
            for obs_key in self.obs_keys
            if cfg.obs[obs_key].modality == "ImageModality"
        ]

        self.image_tuner = HSVBlurCropResolFilter.from_yaml(
            os.path.join(
                FileUtils.get_config_folder(cfg.project_name), "image_filter.yaml"
            )
        )
        self.ros_init()
        rospy.loginfo("PolicyExecutorNode initialized")
        self.inference_start = time.time()

    def ros_init(self):
        self.rate = self.cfg.ros.rate
        self.debug = self.cfg.ros.debug
        self.queue_size = self.cfg.ros.message_filters.queue_size
        self.slop = self.cfg.ros.message_filters.slop
        self.bridge = CvBridge()

        # publishers
        self.pub_action = rospy.Publisher(
            "/eus_imitation/robot_action", Float32MultiArrayStamped, queue_size=1
        )  # TODO
        self.pub_image_obs = [
            rospy.Publisher("/eus_imitation/" + image_obs, Image, queue_size=10)
            for image_obs in self.image_obs
        ]

        # subscribers
        self.sub_obs = [
            message_filters.Subscriber(
                self.cfg.obs[key].topic_name,
                eval(self.cfg.obs[key].msg_type),
            )
            for key in self.obs_keys
        ]
        if self.debug:
            self.sub_obs.append(
                message_filters.Subscriber(
                    "/eus_imitation/robot_action", Float32MultiArrayStamped
                )
            )
        self.obs_ts = message_filters.ApproximateTimeSynchronizer(
            self.sub_obs,
            self.queue_size,
            self.slop,
        )
        self.obs_ts.registerCallback(self.obs_callback)

    def rollout(self, obs_dict: Dict[str, Any]) -> None:
        proprio = obs_dict["proprio"]
        proprio = (proprio - self.proprio_stats["mean"]) / (
            self.proprio_stats["std"] + 1e-8
        )
        head_image = obs_dict["head_image"]

        obs = {
            "image_primary": head_image[None, None],
            "proprio": proprio[None, None],
            "pad_mask": np.asarray([[True]]),
        }

        if self.index % self.update_interval == 0:
            actions = self.policy_fn(
                jax.tree_map(lambda x: x, obs), self.task, rng=jax.random.PRNGKey(0)
            )
            self.actions = actions

        idx = self.index % self.update_interval
        action = self.actions[0, idx] * self.act_stats["std"] + self.act_stats["mean"]
        action = np.clip(action, self.act_stats["min"], self.act_stats["max"])

        self.index += 1
        return action, False

    def reset(self):
        self.index = 0

    def obs_callback(self, *msgs) -> None:
        processed_obs = OrderedDict()
        for key, msg in zip(self.obs_keys, msgs):
            if self.cfg.obs[key].msg_type == "Image":
                processed_obs[key] = self.image_tuner(
                    self.bridge.imgmsg_to_cv2(msg, "rgb8")
                )
            elif self.cfg.obs[key].msg_type == "CompressedImage":
                processed_obs[key] = self.image_tuner(
                    self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
                )
            elif self.cfg.obs[key].msg_type == "Float32MultiArrayStamped":
                processed_obs[key] = np.array(msg.data).astype(np.float32)
            else:
                raise NotImplementedError

        self.inference_start = time.time()
        pred_action, _ = self.rollout(processed_obs)
        if self.debug:
            print(
                "running count: ",
                self.index,
                "callback time: ",
                time.time() - self.inference_start,
            )
            print("pred action: ", pred_action)
            print("real action: ", list(msgs[-1].data))
        else:
            action_msg = Float32MultiArrayStamped()
            action_msg.header.stamp = rospy.Time.now()
            action_msg.data = pred_action.tolist()
            self.pub_action.publish(action_msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-ckpt", "--checkpoint", type=str)
    parser.add_argument("-step", "--step", type=int)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    config = FileUtils.get_config_from_project_name(args.project_name)
    config.network.policy.checkpoint = args.checkpoint
    config.project_name = args.project_name
    config.ros.debug = args.debug
    config.checkpoint_step = args.step
    config.checkpoint_path = args.checkpoint

    rospy.init_node("rollout_node")
    rospy.loginfo("RolloutNode start")
    policy_running_node = OctoROSRollout(config)
    rospy.spin()
