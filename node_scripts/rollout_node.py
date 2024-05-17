#!/usr/bin/env python3
import os
import time
from typing import Any, Dict
from collections import OrderedDict
from functools import partial

import numpy as np
from tunable_filter.composite_zoo import HSVBlurCropResolFilter

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from imitator.utils import file_utils as FileUtils
from imitator.utils.env_utils import RolloutBase
from eus_imitation.msg import FloatVector


class ROSRollout(RolloutBase):
    """
    Wrapper class for rollout in ROS
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(ROSRollout, self).__init__(cfg)
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
        self.bridge = CvBridge()

        # publishers
        self.pub_action = rospy.Publisher(
            "/eus_imitation/robot_action", FloatVector, queue_size=1
        )  # TODO
        self.pub_image_obs = [
            rospy.Publisher("/eus_imitation/" + image_obs, Image, queue_size=1)
            for image_obs in self.image_obs
        ]

        # subscribers
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

    def reset(self):
        super(ROSRollout, self).reset()
        if self.cfg.network.policy.actor_type == "RNNActor":
            self.rnn_state = self.model.get_rnn_init_state(
                batch_size=1, device=self.device
            )

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

    if args.checkpoint is None:
        args.checkpoint = FileUtils.get_best_runs(args.project_name, "rnn")

    config = FileUtils.get_config_from_project_name(args.project_name)
    config.network.policy.checkpoint = args.checkpoint
    config.project_name = args.project_name
    config.ros.debug = args.debug
    config.checkpoint_step = args.step
    config.checkpoint_path = args.checkpoint

    rospy.init_node("rollout_node")
    rospy.loginfo("RolloutNode start")
    policy_running_node = ROSRollout(config)
    rospy.spin()
