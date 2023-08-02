#!/usr/bin/env python3
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn

from typing import Any, Dict, List, Optional, Tuple, Union

import rospy
import message_filters
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from tunable_filter.composite_zoo import HSVBlurCropResolFilter


# import gym
import yaml
from easydict import EasyDict as edict
from collections import OrderedDict

from imitator.utils import tensor_utils as TensorUtils
from imitator.utils import file_utils as FileUtils
from imitator.models.policy_nets import MLPActor, RNNActor
from imitator.utils.obs_utils import *
from imitator.utils.env_utils import RolloutBase
from eus_imitation.msg import Float32MultiArrayStamped


class ROSRollout(RolloutBase):
    """
    Wrapper class for rollout in ROS
    """
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(ROSRollout, self).__init__(cfg)

        self.rate = cfg.ros.rate
        self.topic_names = []
        for key in self.obs_keys:
            self.topic_names.append(cfg.obs[key].topic_name)

        self.debug = False

        self.ros_init()
        self.image_tuner = HSVBlurCropResolFilter.from_yaml(os.path.join(FileUtils.get_config_folder(cfg.project_name), "image_filter.yaml"))

        rospy.loginfo("PolicyExecutorNode initialized")
        self.callback_time = time.time()

    def ros_init(self):
        rospy.init_node("rollout_node")
        self.bridge = CvBridge()
        self.action_pub = rospy.Publisher("/eus_imitation/policy_action", Float32MultiArrayStamped, queue_size=1) # TODO

        self.obs_subs = OrderedDict()
        for key in self.obs_keys:
            self.obs_subs[key] = message_filters.Subscriber(self.topic_names[self.obs_keys.index(key)], eval(self.cfg.obs[key].msg_type))
        if self.debug:
            self.debug_sub = message_filters.Subscriber("/eus_imitation/robot_action", Float32MultiArrayStamped)
            self.obs_ts = message_filters.ApproximateTimeSynchronizer(list(self.obs_subs.values()) + [self.debug_sub], self.cfg.ros.message_filters.queue_size, self.cfg.ros.message_filters.slop)
        else:
            self.obs_ts = message_filters.ApproximateTimeSynchronizer(list(self.obs_subs.values()), self.cfg.ros.message_filters.queue_size, self.cfg.ros.message_filters.slop)
        self.obs_ts.registerCallback(self.obs_callback)


    @torch.no_grad()
    def rollout(self, obs: Dict[str, Any]) -> None:
        pred_action = super(ROSRollout, self).rollout(obs)
        action_msg = Float32MultiArrayStamped()
        action_msg.header.stamp = rospy.Time.now()
        action_msg.data = pred_action.tolist()
        self.action_pub.publish(action_msg)
        return pred_action

    def reset(self):
        super(ROSRollout, self).reset()
        if self.actor_type == RNNActor:
            self.rnn_state = self.model.get_rnn_init_state(batch_size=1, device=self.device)

    def obs_callback(self, *msgs) -> None:
        # assert len(msgs) == len(self.obs_keys)


        obs = self.process_obs(msgs)
        pred_action = self.rollout(obs)
        if self.debug:
            print("pred action", pred_action)
            print("real action", msgs[-1].data)


    def process_obs(self, msgs: Tuple[Any]) -> Dict[str, Any]:
        processed_obs = OrderedDict()
        print("test test test")
        print(len(msgs))
        print("this is msgs",msgs)
        for key, msg in zip(self.obs_keys, msgs):
            if self.cfg.obs[key].msg_type == "Image":
                processed_obs[key] = self.image_tuner(self.bridge.imgmsg_to_cv2(msg, "rgb8"))
            elif self.cfg.obs[key].msg_type == "CompressedImage":
                processed_obs[key] =  self.image_tuner(self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8"))
            elif self.cfg.obs[key].msg_type == "Float32MultiArrayStamped":
                processed_obs[key] = np.array(msg.data).astype(np.float32)
            else:
                raise NotImplementedError

        return processed_obs

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn","--project_name", type=str)
    parser.add_argument("-ckpt","--checkpoint", type=str)
    args = parser.parse_args()

    config = FileUtils.get_config_from_project_name(args.project_name)
    config.network.policy.checkpoint = args.checkpoint
    config.project_name = args.project_name

    rospy.loginfo("RolloutNode start")
    policy_running_node = ROSRollout(config)
    rospy.spin()
