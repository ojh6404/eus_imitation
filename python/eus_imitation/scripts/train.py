#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import yaml
from easydict import EasyDict as edict

from eus_imitation.util.datasets import SequenceDataset
from eus_imitation.base.policy_nets import RNNActor
import eus_imitation.util.tensor_utils as TensorUtils

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--dataset", type=str, default="data/dataset.hdf5")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(yaml.safe_load(f)).actor


    obs_keys = ["image", "robot_ee_pos"]
    dataset_keys = ["actions"]
    batch_size = 16
    num_epochs = 50
    gradient_steps_per_epoch = 100

    dataset = SequenceDataset(
        hdf5_path=args.dataset,
        obs_keys=obs_keys,  # observations we want to appear in batches
        dataset_keys=dataset_keys,  # keys we want to appear in batches
        load_next_obs=True,
        frame_stack=1,
        seq_length=10,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        # hdf5_cache_mode="all",  # cache dataset in memory to avoid repeated file i/o
        hdf5_cache_mode=None,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
    )
    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,  # batches of size 100
        shuffle=True,
        # shuffle=False,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
    )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RNNActor(config).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ensure model is in train mode
    for epoch in range(1, num_epochs + 1):  # epoch numbers start at 1
        # iterator for data_loader - it yields batches
        data_loader_iter = iter(data_loader)

        for _ in range(gradient_steps_per_epoch):
            # load next batch from data loader
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                # data loader ran out of batches - reset and yield first batch
                data_loader_iter = iter(data_loader)
                batch = next(data_loader_iter)
            batch = TensorUtils.to_device(batch, "cuda:0")

            # robot_ee_pos = batch["obs"]["robot_ee_pos"]
            # next_robot_ee_pos = batch["next_obs"]["robot_ee_pos"]
            # image = batch["obs"]["image"]
            # next_image = batch["next_obs"]["image"]
            # print("robot_ee_pos: ", robot_ee_pos.shape, robot_ee_pos.dtype) # [B, T, D]
            # print("next_robot_ee_pos: ", next_robot_ee_pos.shape, next_robot_ee_pos.dtype) # [B, T, D]
            # print("image: ", image.shape, image.dtype) # [B, T, H, W, C]
            # print("next_image: ", next_image.shape, next_image.dtype) # [B, T, H, W, C]



            prediction = model(batch["obs"]) # [B, T, D]
            action = batch["actions"] # [B, T, D]


            loss = nn.MSELoss()(prediction, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print("epoch: {}, loss: {}".format(epoch, loss.item()))

    import matplotlib.pyplot as plt

    random_batch = TensorUtils.to_device(next(data_loader_iter), device)

    actions_of_first_batch = random_batch["actions"][0] # [T, D]
    print("actions_of_first_batch shape: ", actions_of_first_batch.shape) # [T, D]


    # testing
    model.eval()
    with torch.no_grad():
        prediction = model(random_batch["obs"])
        actions = random_batch["actions"]

    prediction_of_first_batch = prediction[0] # [T, D]
    print("prediction_of_first_batch shape: ", prediction_of_first_batch.shape) # [T, D]

    plt.figure()
    plt.plot(
        actions_of_first_batch.cpu().numpy(),
        label="actions",
    )
    plt.plot(
        prediction_of_first_batch.cpu().numpy(),
        label="prediction",
    )
    plt.legend()
    plt.show()




    # testing step by step and whole sequence
