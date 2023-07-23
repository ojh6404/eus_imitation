#!/usr/bin/env python3
#

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from eus_imitation.util.datasets import SequenceDataset
from eus_imitation.base.base_nets import AutoEncoder, VariationalAutoEncoder

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/dataset.hdf5")
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--model", type=str, default="ae")
    args = parser.parse_args()

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
        # hdf5_cache_mode="all",  # cache dataset in memory to avoid repeated file i/o
        hdf5_cache_mode="all",  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
    )

    batch_size = 128
    num_epochs = args.num_epochs

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,  # batches of size 100
        shuffle=True,
        # shuffle=False,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
    )
    if args.model == "ae":
        model = AutoEncoder(
            input_size=(224, 224),
            input_channel=3,
            channels=[8, 16, 32, 64, 128, 256],
            encoder_kernel_sizes=[3, 3, 3, 3, 3, 3],
            decoder_kernel_sizes=[3, 4, 4, 4, 4, 4],
            strides=[2, 2, 2, 2, 2, 2],
            paddings=[1, 1, 1, 1, 1, 1],
            latent_dim=16,
            activation=nn.ReLU,
            dropouts=None,
            normalization=nn.BatchNorm2d,
            output_activation=None,
        ).to("cuda:0")
    elif args.model == "vae":
        model = VariationalAutoEncoder(
            input_size=(224, 224),
            input_channel=3,
            channels=[8, 16, 32, 64, 128, 256],
            encoder_kernel_sizes=[3, 3, 3, 3, 3, 3],
            decoder_kernel_sizes=[3, 4, 4, 4, 4, 4],
            strides=[2, 2, 2, 2, 2, 2],
            paddings=[1, 1, 1, 1, 1, 1],
            latent_dim=16,
            activation=nn.ReLU,
            dropouts=None,
            normalization=nn.BatchNorm2d,
            output_activation=None,
        ).to("cuda:0")
    else:
        raise ValueError("Invalid model type")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ensure model is in train mode
    for epoch in range(1, num_epochs + 1):  # epoch numbers start at 1
        # iterator for data_loader - it yields batches
        data_loader_iter = iter(data_loader)

        # record losses

        # load next batch from data loader
        try:
            batch = next(data_loader_iter)
        except StopIteration:
            # data loader ran out of batches - reset and yield first batch
            data_loader_iter = iter(data_loader)
            batch = next(data_loader_iter)

        image = batch["obs"]["image"].to("cuda:0")

        # [B,T,H,W,C] -> [B*T,C,H,W]
        batch_image = image.reshape(-1, *image.shape[2:]).permute(0, 3, 1, 2)
        batch_image = batch_image.contiguous().float()

        if args.model == "ae":
            x, z = model(batch_image)
            loss = nn.MSELoss()(x, batch_image)
        elif args.model == "vae":
            x, z, mu, logvar = model(batch_image)

            kld_weight = 1e-1 * z.size(1) / (224 * 224 * 3 * batch_size)

            # kl divergence loss
            kl_loss = (
                torch.mean(
                    -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1),
                    dim=0,
                )
                * kld_weight
            )

            # reconstruction loss
            loss = nn.MSELoss()(x, batch_image) + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            if args.model == "ae":
                print("epoch: {}, loss: {}".format(epoch, loss.item()))
            elif args.model == "vae":
                print(
                    "epoch: {}, loss: {}, kl_loss: {}".format(
                        epoch, loss.item(), kl_loss.item()
                    )
                )

    # visualize autoencoder
    #
    model.eval()
    test_image = dataset[0]["obs"]["image"]  # numpy ndarray [B,H,W,C]
    test_image_numpy = test_image.squeeze(0).astype(np.uint8)
    test_image_tensor = (
        torch.from_numpy(test_image).permute(0, 3, 1, 2).to("cuda:0").float()
    ).contiguous()
    with torch.no_grad():
        if args.model == "ae":
            x, z = model(test_image_tensor)
        elif args.model == "vae":
            x, z, mu, logvar = model(test_image_tensor)

        test_image_recon = x.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        test_image_recon = cv2.cvtColor(test_image_recon, cv2.COLOR_RGB2BGR)
        test_image_numpy = cv2.cvtColor(test_image_numpy, cv2.COLOR_RGB2BGR)
        cv2.imshow("test_image", test_image_numpy)
        cv2.imshow("test_image_recon", test_image_recon)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # concat original and reconstructed image
