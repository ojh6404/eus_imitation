#!/usr/bin/env python3
#

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
    num_epochs = 3
    gradient_steps_per_epoch = 1000

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


    model = RNNActor(config)

    model.to("cuda:0")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ensure model is in train mode
    for epoch in range(1, num_epochs + 1):  # epoch numbers start at 1
        # iterator for data_loader - it yields batches
        data_loader_iter = iter(data_loader)

        # record losses

        for _ in range(gradient_steps_per_epoch):
            # load next batch from data loader
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                # data loader ran out of batches - reset and yield first batch
                data_loader_iter = iter(data_loader)
                batch = next(data_loader_iter)


            batch = TensorUtils.to_device(batch, "cuda:0")

            robot_ee_pos = batch["obs"]["robot_ee_pos"]
            next_robot_ee_pos = batch["next_obs"]["robot_ee_pos"]
            image = batch["obs"]["image"]
            next_image = batch["next_obs"]["image"]

            # print("robot_ee_pos: ", robot_ee_pos.shape, robot_ee_pos.dtype) # [B, T, D]
            # print("next_robot_ee_pos: ", next_robot_ee_pos.shape, next_robot_ee_pos.dtype) # [B, T, D]
            # print("image: ", image.shape, image.dtype) # [B, T, H, W, C]
            # print("next_image: ", next_image.shape, next_image.dtype) # [B, T, H, W, C]


            prediction = model(batch["obs"])
            loss = nn.MSELoss()(prediction, next_robot_ee_pos)

            if epoch == 1 and _ == 0:
                print("first loss: ", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch: {}, loss: {}".format(epoch, loss.item()))

    import matplotlib.pyplot as plt

    random_batch = next(data_loader_iter)
    first_of_batch = random_batch["obs"]["robot_ee_pos"][0].unsqueeze(0)
    first_of_batch_next = random_batch["next_obs"]["robot_ee_pos"][0].unsqueeze(0)

    print(first_of_batch.shape)
    print(first_of_batch_next.shape)

    model.eval()

    with torch.no_grad():
        prediction = model(first_of_batch)

    # plot prediction vs ground truth
    plt.figure()
    plt.plot(
        first_of_batch_next.squeeze().numpy(),
        label="ground truth",
    )
    plt.plot(
        prediction.squeeze().numpy(),
        label="prediction",
    )
    plt.legend()
    plt.show()

    # testing step by step and whole sequence

    step_by_step_pred = []
    model.eval()
    rnn_state = torch.zeros(2, 1, 50), torch.zeros(2, 1, 50)
    for step in range(10):
        with torch.no_grad():
            step_prediction, rnn_state = model.forward_step(
                first_of_batch[:, step, :].unsqueeze(1), rnn_state
            )
        step_by_step_pred.append(step_prediction.squeeze().numpy())

    plt.figure()
    plt.plot(
        prediction.squeeze().numpy(),
        label="whole seq",
    )
    plt.plot(
        np.array(step_by_step_pred),
        label="step by step",
    )
    plt.legend()
    plt.show()

    diff = prediction.squeeze().numpy() - np.array(step_by_step_pred)
    print("diff: ", diff)
