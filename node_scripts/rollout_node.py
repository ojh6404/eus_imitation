#!/usr/bin/env python3
import os
import time
from typing import Any, Dict
from collections import OrderedDict

import numpy as np
from tunable_filter.composite_zoo import HSVBlurCropResolFilter

import rospy
import message_filters
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from imitator.utils import file_utils as FileUtils
from imitator.utils.env_utils import RolloutBase
from eus_imitation.msg import Float32MultiArrayStamped


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

    def rollout(self, obs: Dict[str, Any]) -> None:
        pred_action = super(ROSRollout, self).rollout(obs)
        action_msg = Float32MultiArrayStamped()
        action_msg.header.stamp = rospy.Time.now()
        action_msg.data = pred_action.tolist()
        self.pub_action.publish(action_msg)
        return pred_action

    def reset(self):
        super(ROSRollout, self).reset()
        if self.cfg.network.policy.actor_type == "RNNActor":
            self.rnn_state = self.model.get_rnn_init_state(
                batch_size=1, device=self.device
            )

    def obs_callback(self, *msgs) -> None:
        # parse msgs into input obs
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
        pred_action = self.rollout(processed_obs)
        if self.debug:
            print("=====================================================")
            print(
                "running count: {}, callback time: {}".format(
                    self.index, time.time() - self.inference_start
                )
            )
            print("pred action: {}".format(pred_action.tolist()))
            print("real action: {}".format(list(msgs[-1].data)))
            print("=====================================================")


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
