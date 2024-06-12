#!/usr/bin/env python
#-*- coding: utf-8 -*-

import argparse
import os
from typing import Any, Dict

import cv2
import jax
import numpy as np
import rospy
import tensorflow as tf
import tensorflow_datasets as tfds
from eus_imitation_msgs.msg import FloatVector
from eus_imitation_utils.ros_gym_wrapper import ROSRobotEnv
from imitator.utils.env.gym_wrapper import (HistoryWrapper, NormalizeProprio,
                                            ProcessObsWrapper,
                                            ResizeImageWrapper,
                                            TemporalEnsembleWrapper,
                                            UnnormalizeAction)
from imitator.utils.file_utils import (get_config_from_project_name,
                                       get_models_folder)
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache

initialize_compilation_cache()
# prevent tensorflow from using GPU memory since it's only used for data loading
tf.config.set_visible_devices([], "GPU")


class OctoROSRollout(object):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(OctoROSRollout, self).__init__()
        #################################
        # Parse configuration arguments #
        #################################

        if cfg.data_dir is None:
            cfg.data_dir = os.path.expanduser(
                "~/tensorflow_datasets/imitator_dataset/1.0.0"
            )
        if cfg.model_dir is None:
            cfg.model_dir = os.path.join(get_models_folder(cfg.project_name), "models")

        obs_keys = list(cfg.obs.keys())
        img_obs_keys = [
            key for key in obs_keys if cfg.obs[key].modality == "ImageModality"
        ]
        primary_img_key, wrist_img_key = None, None
        primary_img_size, wrist_img_size = None, None
        for img_obs_key in img_obs_keys:
            if cfg.obs[img_obs_key].camera == "primary":
                primary_img_key = img_obs_key
                primary_img_size = tuple(cfg.obs[primary_img_key].dim[:2])
            elif cfg.obs[img_obs_key].camera == "wrist":
                wrist_img_key = img_obs_key
                wrist_img_size = tuple(cfg.obs[wrist_img_key].dim[:2])
            else:
                raise ValueError("Invalid camera type")

        ################################
        # Load dataset for Goal Image  #
        ################################

        builder = tfds.builder_from_directory(
            builder_dir=cfg.data_dir,
        )
        ds = builder.as_dataset(split="train[:1]")  # only one episode

        # sample episode for goal image
        episode = next(iter(ds))
        steps = list(episode["steps"])
        if primary_img_key is not None:
            primary_goal_image = cv2.resize(
                np.array(steps[-1]["observation"][primary_img_key]),
                cfg.obs[primary_img_key].dim[:2],
            )  # last image is the goal image
        if wrist_img_key is not None:
            wrist_goal_image = cv2.resize(
                np.array(steps[-1]["observation"][wrist_img_key]),
                cfg.obs[wrist_img_key].dim[:2],
            )  # last image is the goal image

        ################################
        # Load model and create tasks  #
        ################################

        self.model = OctoModel.load_pretrained(cfg.model_dir)
        act_stats = self.model.dataset_statistics["action"]
        proprio_stats = self.model.dataset_statistics["proprio"]
        language_instruction = cfg.task.get("language_instruction", "Dummy Instruction")
        image_goals = {}  # set goals for images
        if primary_img_key is not None:
            image_goals["image_primary"] = primary_goal_image[None]
        if wrist_img_key is not None:
            image_goals["image_wrist"] = wrist_goal_image[None]
        self.task = self.model.create_tasks(
            texts=[language_instruction], goals=image_goals
        )
        self.policy_fn = jax.jit(self.model.sample_actions)

        ############################
        #    Create environment    #
        ############################

        env = ROSRobotEnv(cfg)
        env = ProcessObsWrapper(
            env,
            flatten_keys=["joint_state", "robot_state"],
            image_keys={
                "primary": primary_img_key,
                "wrist": wrist_img_key,
            },
        )  # process obs

        env = ResizeImageWrapper(
            env,
            resize_size={
                "primary": primary_img_size,
                "wrist": wrist_img_size,
            },
        )  # resize images
        env = NormalizeProprio(env, proprio_stats)  # normalize proprio
        env = UnnormalizeAction(env, act_stats)  # unnormalize actions
        env = TemporalEnsembleWrapper(
            env, pred_horizon=cfg.pred_horizon, exp_weight=0.0
        )  # action chunking and temporal ensemble
        self.env = HistoryWrapper(env, horizon=cfg.window_size)  # window size

        self.obs, _ = self.env.reset()
        self.actions = self.sample_action(self.obs)

        self.pub_action = rospy.Publisher(
            cfg.actions.topic_name, FloatVector, queue_size=1
        )
        self.timer = rospy.Timer(rospy.Duration(1.0 / cfg.ros.rate), self.timer_cb)

    def sample_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self.policy_fn(
            jax.tree.map(lambda x: x[None], obs), self.task, rng=jax.random.PRNGKey(0)
        )[0]  # remove batch dim

    def timer_cb(self, event):
        """
        It is normal that the action is sampled before the observation is updated.
        But in this case, the action is sampled after the observation is updated
        cause we need latest observation to sample the action.
        """
        self.obs, _, _, _, _ = self.env.step(self.actions)
        self.actions = self.sample_action(self.obs)
        action_msg = FloatVector()
        action_msg.header.stamp = rospy.Time.now()
        action_msg.data = self.actions.tolist()
        self.pub_action.publish(action_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-m", "--model_dir", type=str)
    parser.add_argument("-d", "--data_dir", type=str)
    parser.add_argument("-w", "--window_size", type=int, default=2)
    parser.add_argument("-p", "--pred_horizon", type=int, default=10)
    args = parser.parse_args()

    config = get_config_from_project_name(args.project_name)
    config.project_name = args.project_name
    config.model_dir = args.model_dir
    config.data_dir = args.data_dir
    config.window_size = args.window_size
    config.pred_horizon = args.pred_horizon

    rospy.init_node("rollout_node")
    rospy.loginfo("RolloutNode start")
    policy_running_node = OctoROSRollout(config)
    rospy.spin()
