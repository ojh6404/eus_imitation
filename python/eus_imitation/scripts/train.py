#!/usr/bin/env python3

import os
import yaml
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict as edict

from eus_imitation.utils.datasets import SequenceDataset
from eus_imitation.models.policy_nets import RNNActor
import eus_imitation.utils.tensor_utils as TensorUtils

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/config.yaml")
    parser.add_argument("-d", "--dataset", type=str, default="data/dataset.hdf5")
    parser.add_argument("-e", "--num_epochs", type=int, default=3000)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-m", "--model", type=str, default="rnn")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(yaml.safe_load(f)).actor

    obs_keys = ["image", "robot_ee_pos"]
    dataset_keys = ["actions"]
    batch_size = args.batch_size
    num_epochs = args.num_epochs
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
        hdf5_cache_mode=None,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
    )
    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,  # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RNNActor(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.num_epochs // 2, args.num_epochs // 4 * 3],
        gamma=0.1,
    )

    print(model)

    # test
    normalize = True  # TODO
    if normalize:
        with open("./config/normalize.yaml", "r") as f:
            normalizer_cfg = dict(yaml.load(f, Loader=yaml.SafeLoader))

        action_max = torch.Tensor(normalizer_cfg["action"]["max"]).to(device)
        action_min = torch.Tensor(normalizer_cfg["action"]["min"]).to(device)
        action_mean = (action_max + action_min) / 2
        action_std = (action_max - action_min) / 2
        print("action_max: ", action_max)
        print("action_min: ", action_min)

    # make dir and tensorboard writer
    os.makedirs("runs", exist_ok=True)
    output_dir = os.path.join("runs", args.model + "_train")
    summary_writer = SummaryWriter(output_dir)

    model.train()

    best_loss = np.inf

    for epoch in range(1, num_epochs + 1):  # epoch numbers start at 1
        data_loader_iter = iter(data_loader)
        for _ in range(gradient_steps_per_epoch):
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                # data loader ran out of batches - reset and yield first batch
                data_loader_iter = iter(data_loader)
                batch = next(data_loader_iter)
            batch = TensorUtils.to_device(batch, device)

            # calculate time and loss
            start_time = time.time()
            prediction = model(batch["obs"])  # [B, T, D]
            end_time = time.time()
            action = (batch["actions"] - action_mean) / action_std

            loss = nn.MSELoss()(prediction, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if epoch == 1 and _ == 0:
            #     import cv2
            #     with torch.no_grad():
            #         test_image = TensorUtils.to_device(
            #             batch["obs"]["image"], "cuda:0"
            #         )  # [B, T, H, W, C]
            #         # batch_image = batch_image.reshape(
            #         #     -1, *batch_image.shape[2:]
            #         # ).permute(
            #         #     0, 3, 1, 2
            #         # )  # (B, C, H, W)
            #         batch_image = batch_image.contiguous().float() / 255.0
            #         vae_encoder = (
            #             model.nets["obs_encoder"].nets["image"].nets["encoder"]
            #         )
            #         vae_decoder = (
            #             model.nets["obs_encoder"].nets["image"].nets["decoder"]
            #         )
            #         print("batch_image: ", batch_image.shape, batch_image.dtype)
            #         latent, _, _ = vae_encoder(batch_image)
            #         recon = vae_decoder(latent)
            #         recon_image = recon.reshape(-1, *batch_image.shape[1:]).permute(
            #             0, 2, 3, 1
            #         )  # (B, H, W, C)
            #         recon_image = (recon_image * 255.0).byte().cpu().numpy()
            #         # visualize
            #         for i in range(0, batch_image.shape[0], 10):
            #             import cv2
            #             cv2.imshow(
            #                 "original", batch_image[i].permute(1, 2, 0).cpu().numpy()
            #             )
            #             cv2.imshow("recon", recon_image[i])
            #             cv2.waitKey(0)
            # input("Press Enter to continue...")

        if epoch % 10 == 0:
            print(f"epoch: {epoch}, loss: {loss.item():.5g}")
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, args.model + "_model_" + str(epoch) + ".pth"),
            )

        if loss.item() < best_loss:
            print(f"best model saved with loss {loss.item():.5g}")
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, args.model + "_model_best.pth"),
            )
            best_loss = loss.item()

        summary_writer.add_scalar("train/loss", loss.item(), global_step=epoch)
        # lr rate
        summary_writer.add_scalar(
            "train/lr", optimizer.param_groups[0]["lr"], global_step=epoch
        )
        # inference time
        summary_writer.add_scalar(
            "train/inference_time", end_time - start_time, global_step=epoch
        )

        scheduler.step()

    summary_writer.close()
    import matplotlib.pyplot as plt

    random_batch = TensorUtils.to_device(next(data_loader_iter), device)

    actions_of_first_batch = random_batch["actions"][0]  # [T, D]
    print("actions_of_first_batch shape: ", actions_of_first_batch.shape)  # [T, D]

    # testing
    model.eval()
    with torch.no_grad():
        prediction = model(random_batch["obs"])
        actions = random_batch["actions"]

    prediction_of_first_batch = prediction[0]  # [T, D]
    print(
        "prediction_of_first_batch shape: ", prediction_of_first_batch.shape
    )  # [T, D]

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
