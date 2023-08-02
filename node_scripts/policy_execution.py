#!/usr/bin/env python3


import argparse
import time
import yaml
from easydict import EasyDict as edict
import numpy as np
import cv2
import torch
import torch.nn as nn

from collections import OrderedDict
from tunable_filter.composite_zoo import HSVBlurCropResolFilter


import rospy
import rospkg
from cv_bridge import CvBridge
import message_filters

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage, JointState
from eus_imitation.msg import Float32MultiArrayStamped

import imitator.utils.tensor_utils as TensorUtils
from imitator.models.policy_nets import MLPActor, RNNActor
from imitator.utils.obs_utils import ImageModality, FloatVectorModality





class PolicyExecutorNode(object):
    def __init__(self, cfg=dict()):
        rospy.init_node("policy_execution")
        rospack = rospkg.RosPack()
        self.rospack = rospack
        cfg = edict(yaml.safe_load(open(self.rospack.get_path("eus_imitation") + "/config/config.yaml", "r")))

        self.cfg = cfg

        self.obs_keys = list(cfg.obs.keys())
        self.obs_dict = cfg.obs

        self.topic_names = []
        for key in self.obs_keys:
            self.topic_names.append(cfg.obs[key].topic_name)

        print(self.topic_names)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_model(cfg)
        self.running_cnt = 0

        self.hz = 10 # TODO

        self.normalize_cfg = edict(yaml.safe_load(open(rospack.get_path("eus_imitation") + "/config/normalize.yaml", "r")))
        self.action_max = np.array(self.normalize_cfg.action.max).astype(np.float32)
        self.action_min = np.array(self.normalize_cfg.action.min).astype(np.float32)
        self.action_mean = (self.action_max + self.action_min) / 2
        self.action_std = (self.action_max - self.action_min) / 2



        self.rnn_seq_len = cfg.network.policy.rnn.seq_length

        self.tunable = HSVBlurCropResolFilter.from_yaml(rospack.get_path("eus_imitation") + "/config/image_filter.yaml")

        self.obs_subs = OrderedDict()
        for key in self.obs_keys:
            self.obs_subs[key] = message_filters.Subscriber(self.topic_names[self.obs_keys.index(key)], eval(self.obs_dict[key].msg_type))
        self.obs_ts = message_filters.ApproximateTimeSynchronizer(list(self.obs_subs.values()), 10, 0.1)
        self.obs_ts.registerCallback(self.obs_callback)


        # for test
        # self.obs_subs = OrderedDict()
        # for key in self.obs_keys:
        #     self.obs_subs[key] = message_filters.Subscriber(self.topic_names[self.obs_keys.index(key)], eval(self.obs_dict[key].msg_type))
        # self.action_sub = message_filters.Subscriber("/eus_imitation/robot_action", Float32MultiArrayStamped)
        # self.obs_ts = message_filters.ApproximateTimeSynchronizer(list(self.obs_subs.values()) + [self.action_sub], 10, 0.1)

        # self.obs_ts = message_filters.ApproximateTimeSynchronizer(list(self.obs_subs.values()), 10, 0.1)
        # self.obs_ts.registerCallback(self.obs_callback)

        self.action_pub = rospy.Publisher("/eus_imitation/policy_action", Float32MultiArrayStamped, queue_size=1)
        self.bridge = CvBridge()


        self.obs_data = OrderedDict()

        self.callback_time = time.time()

        rospy.loginfo("PolicyExecutorNode initialized")

    def load_model(self, cfg):
        self.model = RNNActor(cfg)
        self.model.load_state_dict(torch.load(self.rospack.get_path("eus_imitation") + "/models/actor.pth"))
        self.model.eval()
        self.model.to(self.device)


    def obs_callback(self, *msgs):
        # assert len(msgs) == len(self.obs_keys)

        for key, msg in zip(self.obs_keys, msgs):
            if self.obs_dict[key].msg_type == "Image":
                self.obs_data[key] = self.tunable(self.bridge.imgmsg_to_cv2(msg, "rgb8"))
            elif self.obs_dict[key].msg_type == "CompressedImage":
                self.obs_data[key] =  self.tunable(self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8"))
            elif self.obs_dict[key].msg_type == "Float32MultiArrayStamped":
                self.obs_data[key] = np.array(msg.data).astype(np.float32)
            else:
                raise NotImplementedError

        self.obs_data = TensorUtils.to_batch(self.obs_data)
        if self.running_cnt % self.rnn_seq_len == 0:
            self.rnn_state = self.model.get_rnn_init_state(batch_size=1, device=self.device)

        with torch.no_grad():
            predicted_action, self.rnn_state = self.model.forward_step(self.obs_data, rnn_state=self.rnn_state, unnormalize=True)
        # predicted_action = (TensorUtils.squeeze(TensorUtils.to_numpy(predicted_action), 0) * self.action_std) + self.action_mean
        predicted_action = (TensorUtils.squeeze(TensorUtils.to_numpy(predicted_action), 0))
        predicted_action = predicted_action.tolist()
        print("predicted_action: ", predicted_action)
        # print("real action", msgs[-1].data)

        """
        predicted_action:  [465.9157409667969, -241.9090576171875, 881.3753051757812, 0.0038826465606689453]
        real action (475.433837890625, -231.9774169921875, 883.0661010742188, 0.0)
        """


        action_msg = Float32MultiArrayStamped()
        action_msg.header.stamp = rospy.Time.now()
        action_msg.data = predicted_action
        self.action_pub.publish(action_msg)
        self.running_cnt += 1

        print(self.running_cnt)





if __name__=="__main__":
    rospy.loginfo("policy_execution.py start")
    policy_running_node = PolicyExecutorNode()
    rospy.spin()
