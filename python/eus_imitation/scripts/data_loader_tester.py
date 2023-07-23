#!/usr/bin/env python3

import argparse
import json
import numpy as np
import time
import os
import psutil
import sys
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import eus_imitation.utils.train_utils as TrainUtils
import eus_imitation.utils.torch_utils as TorchUtils
import eus_imitation.utils.obs_utils as ObsUtils
import eus_imitation.utils.env_utils as EnvUtils
import eus_imitation.utils.file_utils as FileUtils

from eus_imitation.utils.dataset import SequenceDataset


from easydict import EasyDict as edict
import yaml


def main(args):
    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # train(config, device=device)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("config")
    print(config)
    obs_keys = ["image", "robot_ee_pos"]
    dataset_keys = ["actions"]

    ds_kwargs = dict(
        hdf5_path=args.dataset,
        obs_keys=obs_keys,
        dataset_keys=dataset_keys,
        hdf5_cache_mode=None,
    )
    train_dataset = SequenceDataset(**ds_kwargs)
    train_sampler = train_dataset.get_dataset_sampler()

    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=128,
        shuffle=(train_sampler is None),
        num_workers=0,
        drop_last=True,
    )

    train_num_steps = 100
    valid_num_steps = 10

    for epoch in range(1, 1000 + 1):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    args = parser.parse_args()
    main(args)
