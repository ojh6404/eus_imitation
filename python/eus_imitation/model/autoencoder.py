#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import datetime
import random
import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from PIL import Image as plImage
from PIL import ImageOps, ImageMath
import cv2
import matplotlib.pyplot as plt
from image_utils import ImageUtils

from datasets import ImageDataset


class AEModel(nn.Module):
    def __init__(self, channel, height, width, layer1, units):
        self.channel = channel
        self.height = height
        self.width = width
        self.layer1 = layer1
        n_image = int((height / 32) * (width / 32) * layer1 * 16)

        super(AEModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=layer1,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.cbn1 = nn.BatchNorm2d(layer1)
        self.conv2 = nn.Conv2d(
            in_channels=layer1,
            out_channels=layer1 * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.cbn2 = nn.BatchNorm2d(layer1 * 2)
        self.conv3 = nn.Conv2d(
            in_channels=layer1 * 2,
            out_channels=layer1 * 4,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.cbn3 = nn.BatchNorm2d(layer1 * 4)
        self.conv4 = nn.Conv2d(
            in_channels=layer1 * 4,
            out_channels=layer1 * 8,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.cbn4 = nn.BatchNorm2d(layer1 * 8)
        self.conv5 = nn.Conv2d(
            in_channels=layer1 * 8,
            out_channels=layer1 * 16,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.cbn5 = nn.BatchNorm2d(layer1 * 16)

        self.fc1 = nn.Linear(n_image, units[0])
        self.fbn1 = nn.BatchNorm1d(units[0])
        self.fc2 = nn.Linear(units[0], units[1])
        self.fbn2 = nn.BatchNorm1d(units[1])
        self.fc3 = nn.Linear(units[1], units[0])
        self.fbn3 = nn.BatchNorm1d(units[0])
        self.fc4 = nn.Linear(units[0], n_image)
        self.fbn4 = nn.BatchNorm1d(n_image)

        self.dconv5 = nn.ConvTranspose2d(
            in_channels=layer1 * 16,
            out_channels=layer1 * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.dbn5 = nn.BatchNorm2d(layer1 * 8)
        self.dconv4 = nn.ConvTranspose2d(
            in_channels=layer1 * 8,
            out_channels=layer1 * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.dbn4 = nn.BatchNorm2d(layer1 * 4)
        self.dconv3 = nn.ConvTranspose2d(
            in_channels=layer1 * 4,
            out_channels=layer1 * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.dbn3 = nn.BatchNorm2d(layer1 * 2)
        self.dconv2 = nn.ConvTranspose2d(
            in_channels=layer1 * 2,
            out_channels=layer1,
            kernel_size=4,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.dbn2 = nn.BatchNorm2d(layer1)
        self.dconv1 = nn.ConvTranspose2d(
            in_channels=layer1,
            out_channels=self.channel,
            kernel_size=4,
            stride=2,
            padding=1,
            # bias=False,
        )

        self.encoder_lin = nn.Sequential(
            self.fc1,
            self.fbn1,
            nn.ReLU(),
            self.fc2,
            self.fbn2,
            nn.ReLU(),
        )

        self.decoder_lin = nn.Sequential(
            self.fc3,
            self.fbn3,
            nn.ReLU(),
            self.fc4,
            self.fbn4,
            nn.ReLU(),
        )

        self.encoder_cnn = nn.Sequential(
            self.conv1,
            self.cbn1,
            nn.ReLU(),
            self.conv2,
            self.cbn2,
            nn.ReLU(),
            self.conv3,
            self.cbn3,
            nn.ReLU(),
            self.conv4,
            self.cbn4,
            nn.ReLU(),
            self.conv5,
            self.cbn5,
            nn.ReLU(),
        )

        self.decoder_cnn = nn.Sequential(
            self.dconv5,
            self.dbn5,
            nn.ReLU(),
            self.dconv4,
            self.dbn4,
            nn.ReLU(),
            self.dconv3,
            self.dbn3,
            nn.ReLU(),
            self.dconv2,
            self.dbn2,
            nn.ReLU(),
            self.dconv1,
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(
                self.layer1 * 16,
                int(self.height / 32),
                int(self.width / 32),
            ),
        )

        self.criterion = nn.MSELoss()

    def encode(self, x):
        h = self.encoder_cnn(x)
        h = self.flatten(h)
        z = self.encoder_lin(h)
        return z

    def decode(self, z):
        h = self.decoder_lin(z)
        h = self.unflatten(h)
        x = self.decoder_cnn(h)
        x = torch.sigmoid(x)
        return x

    def loss(self, x):
        loss = self.criterion(self.forward(x), x) * 100
        return loss

    def forward(self, x):
        return self.decode(self.encode(x))


class VAEEncoder(nn.Module):
    def __init__(self, channel, height, width, layer1, units):
        self.channel = channel
        self.height = height
        self.width = width
        self.layer1 = layer1
        n_image = int((height / 32) * (width / 32) * layer1 * 16)

        super(VAEEncoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=layer1,
            kernel_size=3,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.cbn1 = nn.BatchNorm2d(layer1)
        self.conv2 = nn.Conv2d(
            in_channels=layer1,
            out_channels=layer1 * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.cbn2 = nn.BatchNorm2d(layer1 * 2)
        self.conv3 = nn.Conv2d(
            in_channels=layer1 * 2,
            out_channels=layer1 * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.cbn3 = nn.BatchNorm2d(layer1 * 4)
        self.conv4 = nn.Conv2d(
            in_channels=layer1 * 4,
            out_channels=layer1 * 8,
            kernel_size=3,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.cbn4 = nn.BatchNorm2d(layer1 * 8)
        self.conv5 = nn.Conv2d(
            in_channels=layer1 * 8,
            out_channels=layer1 * 16,
            kernel_size=3,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.cbn5 = nn.BatchNorm2d(layer1 * 16)

        self.fc1 = nn.Linear(n_image, units[0])
        self.fbn1 = nn.BatchNorm1d(units[0])
        self.fc2_mu = nn.Linear(units[0], units[1])
        self.fc2_log_var = nn.Linear(units[0], units[1])

        self.encoder_cnn = nn.Sequential(
            self.conv1,
            self.cbn1,
            nn.ReLU(),
            self.conv2,
            self.cbn2,
            nn.ReLU(),
            self.conv3,
            self.cbn3,
            nn.ReLU(),
            self.conv4,
            self.cbn4,
            nn.ReLU(),
            self.conv5,
            self.cbn5,
            nn.ReLU(),
        )

        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        h = self.encoder_cnn(x)
        h = self.flatten(h)
        h = F.relu(self.fbn1(self.fc1(h)))
        mu = self.fc2_mu(h)
        log_var = self.fc2_log_var(h)
        eps = torch.randn_like(torch.exp(log_var))
        z = mu + torch.exp(log_var / 2) * eps
        return z, mu, log_var


class VAEDecoder(nn.Module):
    def __init__(self, channel, height, width, layer1, units):
        self.channel = channel
        self.height = height
        self.width = width
        self.layer1 = layer1
        n_image = int((height / 32) * (width / 32) * layer1 * 16)

        super(VAEDecoder, self).__init__()
        self.fc3 = nn.Linear(units[1], units[0])
        self.fbn3 = nn.BatchNorm1d(units[0])
        self.fc4 = nn.Linear(units[0], n_image)
        self.fbn4 = nn.BatchNorm1d(n_image)

        self.dconv5 = nn.ConvTranspose2d(
            in_channels=layer1 * 16,
            out_channels=layer1 * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.dbn5 = nn.BatchNorm2d(layer1 * 8)
        self.dconv4 = nn.ConvTranspose2d(
            in_channels=layer1 * 8,
            out_channels=layer1 * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.dbn4 = nn.BatchNorm2d(layer1 * 4)
        self.dconv3 = nn.ConvTranspose2d(
            in_channels=layer1 * 4,
            out_channels=layer1 * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.dbn3 = nn.BatchNorm2d(layer1 * 2)
        self.dconv2 = nn.ConvTranspose2d(
            in_channels=layer1 * 2,
            out_channels=layer1,
            kernel_size=4,
            stride=2,
            padding=1,
            # bias=False,
        )
        self.dbn2 = nn.BatchNorm2d(layer1)
        self.dconv1 = nn.ConvTranspose2d(
            in_channels=layer1,
            out_channels=self.channel,
            kernel_size=4,
            stride=2,
            padding=1,
            # bias=False,
        )

        self.decoder_lin = nn.Sequential(
            self.fc3,
            self.fbn3,
            nn.ReLU(),
            self.fc4,
            self.fbn4,
            nn.ReLU(),
        )

        self.decoder_cnn = nn.Sequential(
            self.dconv5,
            self.dbn5,
            nn.ReLU(),
            self.dconv4,
            self.dbn4,
            nn.ReLU(),
            self.dconv3,
            self.dbn3,
            nn.ReLU(),
            self.dconv2,
            self.dbn2,
            nn.ReLU(),
            self.dconv1,
        )

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(
                self.layer1 * 16,
                int(self.height / 32),
                int(self.width / 32),
            ),
        )

        self.criterion = nn.MSELoss()

    def forward(self, z):
        h = self.decoder_lin(z)
        h = self.unflatten(h)
        x = self.decoder_cnn(h)
        x = torch.sigmoid(x)
        return x


class VAEModel(nn.Module):
    def __init__(self, channel, height, width, layer1, units):
        self.channel = channel
        self.height = height
        self.width = width
        self.layer1 = layer1
        n_image = int((height / 32) * (width / 32) * layer1 * 16)

        super(VAEModel, self).__init__()
        self.encoder = VAEEncoder(channel, height, width, layer1, units)
        self.decoder = VAEDecoder(channel, height, width, layer1, units)
        self.criterion = nn.MSELoss()

    def loss(self, x):
        pred_x, z, mu, log_var = self.forward(x)
        reconstruction_loss = -torch.mean(
            x * torch.log(pred_x + 1e-8) + (1 - x) * torch.log(1 - pred_x + 1e-8)
        )
        kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - torch.exp(log_var))
        return reconstruction_loss + kl_loss

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x = self.decoder(z)
        return x, z, mu, log_var


class ImageAutoEncoder(object):
    def __init__(
        self, channel, height, width, layer1, units, data_cfg, train_cfg, vae, gpu
    ):
        self.vae = vae
        self.channel = channel
        self.height = height
        self.width = width
        self.layer1 = layer1
        self.units = units
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.gpu = gpu

        self.lr = self.train_cfg["lr"]

        self._rgb_mean = self.data_cfg["image"]["rgb_mean"]
        self._rgb_std = self.data_cfg["image"]["rgb_std"]

        if self.vae:
            self.model = VAEModel(channel, height, width, layer1, units)
        else:
            self.model = AEModel(channel, height, width, layer1, units)
        self.device = torch.device("cuda:" + str(self.gpu) if self.gpu >= 0 else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_valid_loss = float("inf")
        if self.vae:
            self.out_dir = "runs/" + datetime.datetime.today().strftime(
                "%Y-%m-%d-%H-%M-%S-image-vae"
            )
        else:
            self.out_dir = "runs/" + datetime.datetime.today().strftime(
                "%Y-%m-%d-%H-%M-%S-image"
            )
        self.summary_writer = SummaryWriter(self.out_dir)

    def read_data(self, dataset_dir):
        self.datasets = ImageDataset(
            dataset_dir,
            "/tmp",
            self.data_cfg,
            self.channel,
            self.height,
            self.width,
        )
        self.n_data = len(self.datasets)

    def train(self, batch=10, epochs=100, train_rate=0.9):
        with_test = train_rate < 1.0

        if with_test:
            self.n_train = int(self.n_data * train_rate)
            self.n_test = self.n_data - self.n_train
            self.train, self.test = random_split(
                self.datasets, [self.n_train, self.n_test]
            )

            self.train_dataloader = DataLoader(
                self.train, batch_size=batch, shuffle=True, pin_memory=True
            )
            self.test_dataloader = DataLoader(self.test, batch_size=32, shuffle=False)
        else:
            self.train = self.datasets
            self.train_dataloader = DataLoader(
                self.train, batch_size=batch, shuffle=True, pin_memory=True
            )

        for epoch in range(epochs):
            # train
            losses = []
            self.model.train()
            for x in self.train_dataloader:
                x = x.to(self.device)

                self.model.zero_grad()

                loss = self.model.loss(x)
                self.summary_writer.add_scalar("Loss/train", loss, epoch)

                loss.backward()
                self.optimizer.step()

                losses.append(loss.cpu().detach().numpy())

            # validation
            if with_test:
                losses_val = []
                self.model.eval()
                for x in self.test_dataloader:
                    x = x.to(self.device)

                    loss = self.model.loss(x)
                    self.summary_writer.add_scalar("Loss/valid", loss, epoch)

                    losses_val.append(loss.cpu().detach().numpy())

                print(
                    f"epoch : {epoch}\t train loss : {np.average(losses):.3f}\t validation loss : {np.average(losses_val):.3f}"
                )

                if np.average(losses_val) < self.best_valid_loss:
                    self.best_valid_loss = np.average(losses_val)
                    self.save_model(
                        model_filename=self.out_dir + "/image_model_best.pth"
                    )

                if epoch % 10 == 0:
                    self.save_model(
                        model_filename="/tmp/image_model_" + str(epoch) + ".pth"
                    )
            else:
                print(f"epoch : {epoch}\t train loss : {np.average(losses):.3f}")
                if np.average(losses) < self.best_valid_loss:
                    self.best_valid_loss = np.average(losses)
                    self.save_model(
                        model_filename=self.out_dir + "/image_model_best.pth"
                    )
                if epoch % 10 == 0:
                    self.save_model(
                        model_filename="/tmp/image_model_" + str(epoch) + ".pth"
                    )

        self.summary_writer.close()

        if self.gpu >= 0:
            self.model.to("cpu")

        del self.datasets
        return self.out_dir

    def dump_args(self, args, out_dir):
        import json

        with open(out_dir + "/args.json", mode="w") as f:
            json.dump(args.__dict__, f, indent=4)

    def save_model(self, model_filename, debug=False):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
            },
            model_filename,
        )
        if debug:
            print(model_filename + " is saved.")

    def load_model(self, model_filename):
        model_cpt = torch.load(model_filename)
        model_state_dict = model_cpt["model_state_dict"]
        self.model.load_state_dict(model_state_dict)

    def verify(self):
        cnt = random.randint(0, self.n_data - 1)
        image = self.datasets[cnt].unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            image_pred = self.model(image)
        image_pred = image_pred.to("cpu").detach().numpy().copy().squeeze()
        image_real = image.to("cpu").detach().numpy().copy().squeeze()
        if self.data_cfg["image"]["color_norm"]:
            image_pred = ImageUtils.color_unnorm(
                image_pred, self._rgb_mean, self._rgb_std
            )
            image_real = ImageUtils.color_unnorm(
                image_real,
                self._rgb_mean,
                self._rgb_std,
            )
        is_color = self.channel == 3
        ImageUtils.show_image(
            ImageUtils.make_concatenated_image(
                [image_real, image_pred],
                self.height,
                self.width,
                color=is_color,
            ),
            color=is_color,
        )

        del self.datasets
