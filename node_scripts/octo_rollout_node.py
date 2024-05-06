#!/usr/bin/env python

import os
import time
from typing import Any, Dict
from collections import OrderedDict
from functools import partial

import numpy as np
from tunable_filter.composite_zoo import HSVBlurCropResolFilter
import jax

import rospy
import message_filters
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from imitator.utils import file_utils as FileUtils
from eus_imitation.msg import FloatVector

from octo.model.octo_model import OctoModel


def jax_has_gpu():
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices("gpu")[0])
        return True
    except:
        return False


assert (
    jax_has_gpu()
), "JAX does not have GPU support. Please install JAX with GPU support."


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
        self.instruction ="Pick up the string on the left side"
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

        self.obs_dict_buf = {obs_key: None for obs_key in self.obs_keys}

        self.image_tuner = HSVBlurCropResolFilter.from_yaml(
            os.path.join(
                FileUtils.get_config_folder(cfg.project_name), "image_filter.yaml"
            )
        )
        self.ros_init()
        rospy.loginfo("PolicyExecutorNode initialized")
        self.inference_start = time.time()

    def obs_callback(self, obs_key, msg):
        if self.cfg.obs[obs_key].msg_type == "Image":
            self.obs_dict_buf[obs_key] = self.image_tuner(
                self.bridge.imgmsg_to_cv2(msg, "rgb8")
            )
        elif self.cfg.obs[obs_key].msg_type == "CompressedImage":
            self.obs_dict_buf[obs_key] = self.image_tuner(
                self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
            )
        elif self.cfg.obs[obs_key].msg_type == "FloatVector":
            self.obs_dict_buf[obs_key] = np.array(msg.data).astype(np.float32)
        else:
            raise NotImplementedError

    def ros_init(self):
        self.rate = self.cfg.ros.rate
        self.debug = self.cfg.ros.debug
        self.queue_size = self.cfg.ros.message_filters.queue_size
        self.slop = self.cfg.ros.message_filters.slop
        self.bridge = CvBridge()

        # publishers
        self.pub_action = rospy.Publisher(
            "/eus_imitation/robot_action", FloatVector, queue_size=1
        )  # TODO
        self.pub_image_obs = [
            rospy.Publisher("/eus_imitation/" + image_obs, Image, queue_size=1)
            for image_obs in self.image_obs
        ]

        self.sub_obs = [
            rospy.Subscriber(
                self.cfg.obs[key].topic_name,
                eval(self.cfg.obs[key].msg_type),
                partial(self.obs_callback, key),
                queue_size=1,
                buff_size=2 ** 24,
            )
            for key in self.obs_keys
        ]

        if self.debug:
            self.sub_obs.append(
                rospy.Subscriber(
                    "/eus_imitation/robot_action", FloatVector, self.action_callback, queue_size=1
                )
            )

        # rospy Timer Callback
        self.callback_start = time.time()
        self.timer = rospy.Timer(rospy.Duration(1 / self.rate), self.timer_callback)


    def action_callback(self, msg):
        self.debug_action = np.array(msg.data).astype(np.float32)

    def rollout(self, obs_dict: Dict[str, Any]) -> None:
        proprio = obs_dict["proprio"]
        proprio = (proprio - self.proprio_stats["mean"]) / (
            self.proprio_stats["std"] + 1e-8
        )
        head_image = obs_dict["head_image"]

        obs = {
            "image_primary": head_image[None, None],
            "proprio": proprio[None, None],
            "pad_mask": np.asarray([[True]]),  # TODO only for window size 1
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

    def timer_callback(self, event):
        for self.obs_key in self.obs_keys:
            if self.obs_dict_buf[self.obs_key] is None:
                return
        print("callback time: ", time.time() - self.callback_start)
        self.callback_start = time.time()
        self.inference_start = time.time()
        pred_action, _ = self.rollout(self.obs_dict_buf)
        print(
            "running count: ",
            self.index,
            "inference time: ",
            time.time() - self.inference_start,
        )
        pred_action = pred_action.tolist()
        if self.debug:
            print("pred action: ", pred_action)
            print("real action: ", self.debug_action)
        else:
            action_msg = FloatVector()
            action_msg.header.stamp = rospy.Time.now()
            action_msg.data = pred_action
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
