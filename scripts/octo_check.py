#!/usr/bin/env python3

import os
import jax
import h5py
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from imitator.utils.file_utils import sort_names_by_number
from imitator.utils import file_utils as FileUtils
from imitator.utils import tensor_utils as TensorUtils

import time
from typing import Any, Dict
from collections import OrderedDict
from functools import partial

import numpy as np
# from tunable_filter.composite_zoo import HSVBlurCropResolFilter

from octo.model.octo_model import OctoModel

# debug script for octo with dataset

def jax_has_gpu():
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices("gpu")[0])
        return True
    except:
        return False


assert (
    jax_has_gpu()
), "JAX does not have GPU support. Please install JAX with GPU support."

class OctoRollout(object):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super(OctoRollout, self).__init__()
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

        self.obs_dict_buf = {obs_key: None for obs_key in self.obs_keys}
        self.inference_start = time.time()

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
        print("callback time: ", time.time() - self.callback_start)
        self.callback_start = time.time()
        self.inference_start = time.time()
        pred_action, _ = self.rollout(self.obs_dict_buf)
        pred_action = pred_action.tolist()


def main(args, config):
    octo_model = OctoRollout(config)

    f = h5py.File(args.dataset, "r")

    #pred_actions, _ = octo_model.rollout(f["data/{}".format(demos[0])]["obs"])
    demos = sort_names_by_number(f["data"].keys())
    # get several demo
    for j in range(10):
        demo = f["data/{}".format(demos[j])]
        obs_keys = list(demo["obs"].keys())
        actions = demo["actions"]

        pred_actions = []

        start_time = time.time()
        np.set_printoptions(precision=3)
        for i in range(len(actions)):
            action = actions[i]
            obs_dict = {obs_key: demo["obs/{}".format(obs_key)][i] for obs_key in obs_keys}

            pred_action, _ = octo_model.rollout(obs_dict)
            pred_actions.append(pred_action)
            print("time: ", time.time() - start_time)
            print(f"pred action: {pred_action}, real action: {action}, diff: {pred_action - action}")
            start_time = time.time()

        # plot and save
        pred_actions = np.array(pred_actions)
        fig_action = plt.figure()
        fig_action.suptitle("action")
        for i in range(pred_actions.shape[1]): # for each action dimension
            plt.subplot(pred_actions.shape[1], 1, i + 1)
            plt.plot(actions[:, i], label="real")
            plt.plot(pred_actions[:, i], label="pred")
            plt.legend()
        plt.savefig(f"octo_action_{args.project_name}_{j}.png")

    f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    parser.add_argument("-ckpt", "--checkpoint", type=str)
    parser.add_argument("-step", "--step", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    config = FileUtils.get_config_from_project_name(args.project_name)
    config.network.policy.checkpoint = args.checkpoint
    config.project_name = args.project_name
    config.ros.debug = args.debug
    config.checkpoint_step = args.step
    config.checkpoint_path = args.checkpoint

    main(args, config)
