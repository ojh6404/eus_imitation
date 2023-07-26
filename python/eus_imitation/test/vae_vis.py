#!/usr/bin/env python3
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn

from eus_imitation.util import tensor_utils as TensorUtils
from eus_imitation.util.datasets import SequenceDataset
from eus_imitation.base.base_nets import AutoEncoder, VariationalAutoEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="data/dataset.hdf5")
    parser.add_argument("-m", "--model", type=str, default="ae")
    parser.add_argument("-ckpt", "--checkpoint", type=str, default=None)
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

    model.load_state_dict(torch.load(args.checkpoint))
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
