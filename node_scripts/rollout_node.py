#!/usr/bin/env python3
import os
import time
from typing import Any, Dict
from collections import OrderedDict

import numpy as np
import torch
from tunable_filter.composite_zoo import HSVBlurCropResolFilter

import rospy
import message_filters
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from imitator.utils import tensor_utils as TensorUtils
from imitator.utils import file_utils as FileUtils
from imitator.utils.obs_utils import *
from imitator.utils.env_utils import RolloutBase
from eus_imitation.msg import Float32MultiArrayStamped
from imitator.models.policy_nets import MLPActor, RNNActor, TransformerActor


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

        self.debug = cfg.ros.debug

        self.ros_init()
        self.image_tuner = HSVBlurCropResolFilter.from_yaml(
            os.path.join(
                FileUtils.get_config_folder(cfg.project_name), "image_filter.yaml"
            )
        )

        rospy.loginfo("PolicyExecutorNode initialized")
        self.inference_start = time.time()

    def ros_init(self):
        rospy.init_node("rollout_node")
        self.bridge = CvBridge()
        self.action_pub = rospy.Publisher(
            "/eus_imitation/robot_action", Float32MultiArrayStamped, queue_size=1
        )  # TODO
        self.image_obs_pubs = []
        for image_obs in self.image_obs:
            self.image_obs_pubs.append(
                rospy.Publisher(
                    "/eus_imitation/" + image_obs,
                    Image,
                    queue_size=10,
                )
            )

        self.obs_subs = OrderedDict()
        for key in self.obs_keys:
            self.obs_subs[key] = message_filters.Subscriber(
                self.topic_names[self.obs_keys.index(key)],
                eval(self.cfg.obs[key].msg_type),
            )
        if self.debug:
            self.debug_sub = message_filters.Subscriber(
                "/eus_imitation/robot_action", Float32MultiArrayStamped
            )
            self.obs_ts = message_filters.ApproximateTimeSynchronizer(
                list(self.obs_subs.values()) + [self.debug_sub],
                self.cfg.ros.message_filters.queue_size,
                self.cfg.ros.message_filters.slop,
            )
        else:
            self.obs_ts = message_filters.ApproximateTimeSynchronizer(
                list(self.obs_subs.values()),
                self.cfg.ros.message_filters.queue_size,
                self.cfg.ros.message_filters.slop,
            )
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
            self.rnn_state = self.model.get_rnn_init_state(
                batch_size=1, device=self.device
            )

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

        # processed_obs : dict of np.array like [D]

        self.inference_start = time.time()

        pred_action = self.rollout(processed_obs)
        if self.debug:
            print(
                "running count: ",
                self.running_cnt,
                "callback time: ",
                time.time() - self.inference_start,
            )
            print("pred action: ", pred_action.tolist())
            print("real action: ", list(msgs[-1].data))

    # def process_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
    #     return obs

    @torch.no_grad()
    def render(self, obs: Dict[str, Any]) -> None:

        if self.actor_type == TransformerActor: # obs is stacked if transformer like [1, T, D]
            # so we need to use last time step obs to render
            obs = {k: v[:, -1, :] for k, v in obs.items()}

        if self.image_obs:
            obs = TensorUtils.squeeze(obs, dim=0)
            for image_obs in self.image_obs:
                image_render = obs[image_obs]

                # if has_decoder, concat recon and original image to visualize
                if image_obs in self.image_decoder:
                    image_latent = self.image_encoder[image_obs](
                        image_render[None, ...]
                    )  # [1, C, H, W]
                    image_recon = (
                        self.image_decoder[image_obs](image_latent) * 255.0
                    )  # [1, C, H, W] TODO set unnormalizer
                    image_recon = image_recon.cpu().numpy().astype(np.uint8)
                    image_recon = np.transpose(
                        image_recon, (0, 2, 3, 1)
                    )  # [1, H, W, C]
                    image_recon = np.squeeze(image_recon)
                    image_render = concatenate_image(image_render, image_recon)

                image_msg = self.bridge.cv2_to_imgmsg(image_render, "rgb8")
                image_msg.header.stamp = rospy.Time.now()
                self.image_obs_pubs[self.image_obs.index(image_obs)].publish(image_msg)
        else:
            # pass when no image obs
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-ckpt", "--checkpoint", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = FileUtils.get_best_runs(args.project_name, "rnn")

    config = FileUtils.get_config_from_project_name(args.project_name)
    config.network.policy.checkpoint = args.checkpoint
    config.project_name = args.project_name
    config.ros.debug = args.debug

    rospy.loginfo("RolloutNode start")
    policy_running_node = ROSRollout(config)
    rospy.spin()
