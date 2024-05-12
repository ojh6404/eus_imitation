#!/usr/bin/env python

import time
from typing import Any, Dict
from functools import partial
from omegaconf import OmegaConf

import numpy as np
import jax
from tunable_filter.composite_zoo import HSVBlurCropResolFilter

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
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
        self.window_size = 2

        self.cfg = cfg
        self.obs_keys = list(cfg.obs.keys())
        self.image_obs = [
            obs_key
            for obs_key in self.obs_keys
            if cfg.obs[obs_key].modality == "ImageModality"
        ]

        self.obs_dict_buf = {obs_key: None for obs_key in self.obs_keys}
        self.obs_dict_input = {obs_key: [None for _ in range(self.window_size)] for obs_key in self.obs_keys}

        self.image_tuner = HSVBlurCropResolFilter.from_yaml(cfg.image_config)
        self.ros_init()
        rospy.loginfo("PolicyExecutorNode initialized")
        self.inference_start = time.time()

        self.timer = rospy.Timer(rospy.Duration(1 / self.rate), self.timer_callback)

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


        topic_names = [self.cfg.obs[key].topic_name for key in self.obs_keys]
        rospy.loginfo("Subscribing to topics: {}".format(topic_names))

        # rospy Timer Callback
        self.callback_start = time.time()
        self.timer = rospy.Timer(rospy.Duration(1 / self.rate), self.timer_callback)


    def action_callback(self, msg):
        self.debug_action = np.array(msg.data).astype(np.float32)

    def rollout(self, obs_dict: Dict[str, Any]) -> None:
        for obs_key in self.obs_keys:
            obs_dict[obs_key] = np.stack(obs_dict[obs_key])[None] # (1,WINDOW_SIZE,H,W,C)

        input_images = obs_dict["head_image"]
        input_proprios = obs_dict["proprio"]
        input_proprios = (input_proprios - self.proprio_stats['mean'][None][None]) / (self.proprio_stats['std'][None][None] + 1e-8)
        observation = {
            'image_primary': input_images,
            'prorpio' : input_proprios,
            'pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool) # (1,WINDOW_SIZE)
        }

        # this returns *normalized* actions --> we need to unnormalize using the dataset statistics
        # norm_actions = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
        norm_actions = policy_fn(jax.tree_map(lambda x: x, observation), task, rng=jax.random.PRNGKey(0))
        norm_actions = norm_actions[0]   # remove batch

        action = (
            norm_actions * self.act_stats['std']
            + self.act_stats['mean']
        )
        action = np.clip(action, self.act_stats["min"], self.act_stats["max"])
        return action

    def reset(self):
        self.index = 0

    def timer_callback(self, event):
        for self.obs_key in self.obs_keys:
            if self.obs_dict_buf[self.obs_key] is None:
                return

        # if None is in the obs_dict_input, then we need to fill it
        for obs_key in self.obs_keys:
            if None in self.obs_dict_input[obs_key]:
                self.obs_dict_input[obs_key].pop(0) #
                self.obs_dict_input[obs_key].append(self.obs_dict_buf[obs_key])
            else: # obs_dict_input[obs_key] is full, len = window_size
                print("callback time: ", time.time() - self.callback_start)
                self.callback_start = time.time()
                self.inference_start = time.time()
                pred_action = self.rollout(self.obs_dict_input)
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
    parser.add_argument("-conf", "--config", type=str)
    parser.add_argument("-img_conf", "--image_config", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.network.policy.checkpoint = args.checkpoint
    config.project_name = args.project_name
    config.ros.debug = args.debug
    config.checkpoint_step = args.step
    config.checkpoint_path = args.checkpoint
    config.image_config = args.image_config

    rospy.init_node("rollout_node")
    rospy.loginfo("RolloutNode start")
    policy_running_node = OctoROSRollout(config)
    rospy.spin()
