#!/usr/bin/env python3
import cv2
from collections import OrderedDict
import numpy as np
from robomimic.algo.algo import TensorUtils
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="ae")
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
    for epoch in range(1, args.num_epochs + 1):  # epoch numbers start at 1
        data_loader_iter = iter(data_loader)
        try:
            batch = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            batch = next(data_loader_iter)

        batch_image =  TensorUtils.to_device(batch["obs"]["image"], device)
        batch_image = batch_image.reshape(-1, *batch_image.shape[2:]).permute(0, 3, 1, 2) # (B, C, H, W)
        batch_image = batch_image.contiguous().float() / 255.0

        loss_sum = 0
        loss_dict = model.loss(batch_image)
        for loss in loss_dict.values():
            loss_sum += loss

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        if epoch % 10 == 0:
            if args.model == "ae":
                print("epoch: {}, loss: {}".format(epoch, loss_sum.item()))
            elif args.model == "vae":
                print(
                    "epoch: {}, loss: {}, mse: {}, kl: {}".format(
                        epoch,
                        loss_sum.item(),
                        loss_dict["reconstruction_loss"].item(),
                        loss_dict["kld_loss"].item(),
                    )
                )

    # save model
    torch.save(model.state_dict(), args.model + "_model.pth")
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
    model.load_state_dict(torch.load(args.model + "_model.pth"))
    model.eval()

    # test
    random_index = np.random.randint(0, len(dataset))
    test_image = dataset[random_index]["obs"]["image"]  # numpy ndarray [B,H,W,C]
    test_image_numpy = test_image.squeeze(0).astype(np.uint8)
    test_image_tensor = TensorUtils.to_device(TensorUtils.to_tensor(test_image), device)
    test_image_tensor = test_image_tensor.permute(0, 3, 1, 2).float().contiguous() / 255.0
    with torch.no_grad():
        if args.model == "ae":
            x, z = model(test_image_tensor)
        elif args.model == "vae":
            x, z, mu, logvar = model(test_image_tensor)

        test_image_recon = (TensorUtils.to_numpy(x.squeeze(0).permute(1, 2, 0)) * 255.0).astype(np.uint8)
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
