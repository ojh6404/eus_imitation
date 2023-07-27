#!/usr/bin/env python3
import argparse
import os
import time
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import eus_imitation.utils.tensor_utils as TensorUtils
from eus_imitation.utils.datasets import SequenceDataset
from eus_imitation.models.base_nets import AutoEncoder, VariationalAutoEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="data/dataset.hdf5")
    parser.add_argument("-e", "--num_epochs", type=int, default=3000)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-m", "--model", type=str, default="ae")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    obs_keys = ["image"]
    dataset_keys = ["actions"]

    dataset = SequenceDataset(
        hdf5_path=args.dataset,
        obs_keys=obs_keys,  # observations we want to appear in batches
        dataset_keys=dataset_keys,  # keys we want to appear in batches
        load_next_obs=True,
        frame_stack=1,
        seq_length=1,  # length-10 temporal sequences
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
        batch_size=args.batch_size,  # batches of size 100
        shuffle=True,
        # shuffle=False,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
    )
    if args.model == "ae":
        model = AutoEncoder(
            input_size=(224, 224),
            input_channel=3,
            latent_dim=16,
            normalization=nn.BatchNorm2d,
            output_activation=nn.Sigmoid,
        ).to(device)
    elif args.model == "vae":
        model = VariationalAutoEncoder(
            input_size=(224, 224),
            input_channel=3,
            latent_dim=16,
            normalization=nn.BatchNorm2d,
            output_activation=nn.Sigmoid,
        ).to(device)
    else:
        raise ValueError("Invalid model type")

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.num_epochs // 2, args.num_epochs // 4 * 3],
        gamma=0.1,
    )
    best_loss = np.inf

    # make dir and tensorboard writer
    os.makedirs("runs", exist_ok=True)
    output_dir = os.path.join("runs", args.model + "_train")
    summary_writer = SummaryWriter(output_dir)

    for epoch in range(1, args.num_epochs + 1):  # epoch numbers start at 1
        data_loader_iter = iter(data_loader)
        try:
            batch = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            batch = next(data_loader_iter)

        batch_image = TensorUtils.to_device(batch["obs"]["image"], device)
        batch_image = batch_image.reshape(-1, *batch_image.shape[2:]).permute(
            0, 3, 1, 2
        )  # (B, C, H, W)
        batch_image = batch_image.contiguous().float() / 255.0

        loss_sum = 0
        loss_dict = model.loss(batch_image)
        for loss in loss_dict.values():
            loss_sum += loss

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        summary_writer.add_scalar("train/loss", loss_sum.item(), global_step=epoch)
        # lr rate
        summary_writer.add_scalar(
            "train/lr", optimizer.param_groups[0]["lr"], global_step=epoch
        )

        # print loss with 5 significant digits every 100 epochs
        if epoch % 100 == 0:
            loss = loss_sum.item()
            if args.model == "ae":
                print(f"epoch: {epoch}, loss: {loss:.5g}")
            elif args.model == "vae":
                recons_loss = loss_dict["reconstruction_loss"].item()
                kl_loss = loss_dict["kld_loss"].item()
                print(
                    f"epoch: {epoch}, loss: {loss:.5g}, recons_loss: {recons_loss:.5g}, kl_loss: {kl_loss:.5g}"
                )
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, args.model + "_model_" + str(epoch) + ".pth"),
            )

        if loss_sum.item() < best_loss and (epoch > args.num_epochs / 10):
            print(f"best model saved with loss {loss_sum.item():.5g}")
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, args.model + "_model_best.pth"),
            )
            best_loss = loss_sum.item()

        scheduler.step()

    summary_writer.close()
    del model
    # load model for test
    if args.model == "ae":
        model = AutoEncoder(
            input_size=(224, 224),
            input_channel=3,
            latent_dim=16,
            normalization=nn.BatchNorm2d,
        ).to(device)
    elif args.model == "vae":
        model = VariationalAutoEncoder(
            input_size=(224, 224),
            input_channel=3,
            latent_dim=16,
            normalization=nn.BatchNorm2d,
            output_activation=nn.Sigmoid,
        ).to(device)
    else:
        raise ValueError("Invalid model type")
    # model.load_state_dict(torch.load(args.model + "_model.pth"))
    model.load_state_dict(
        torch.load(os.path.join(output_dir, args.model + "_model_best.pth"))
    )
    model.eval()

    # test
    random_index = np.random.randint(0, len(dataset))
    test_image = dataset[random_index]["obs"]["image"]  # numpy ndarray [B,H,W,C]
    test_image_numpy = test_image.squeeze(0).astype(np.uint8)
    test_image_tensor = TensorUtils.to_device(TensorUtils.to_tensor(test_image), device)
    test_image_tensor = (
        test_image_tensor.permute(0, 3, 1, 2).float().contiguous() / 255.0
    )
    with torch.no_grad():
        if args.model == "ae":
            x, z = model(test_image_tensor)
        elif args.model == "vae":
            x, z, mu, logvar = model(test_image_tensor)

        test_image_recon = (
            TensorUtils.to_numpy(x.squeeze(0).permute(1, 2, 0)) * 255.0
        ).astype(np.uint8)
        test_image_recon = cv2.cvtColor(test_image_recon, cv2.COLOR_RGB2BGR)
        test_image_numpy = cv2.cvtColor(test_image_numpy, cv2.COLOR_RGB2BGR)
        cv2.imshow("test_image", test_image_numpy)
        cv2.imshow("test_image_recon", test_image_recon)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    with torch.no_grad():
        embedding = model.nets["encoder"](test_image_tensor)
        print(embedding[0].shape)

    del dataset
