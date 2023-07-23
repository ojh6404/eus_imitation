#!/usr/bin/env python3
#

import numpy as np
import torch
import torch.nn as nn
from eus_imitation.util.datasets import SequenceDataset


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        self.input_dim = 4
        self.rnn_input_dim = 10
        self.output_dim = 4
        self.seq_len = 10
        self.rnn_hidden_dim = 50
        self.rnn_layers = 2

        self.fc_in = nn.Linear(self.input_dim, self.rnn_input_dim)
        self.lstm = nn.LSTM(
            self.rnn_input_dim,
            self.rnn_hidden_dim,
            self.rnn_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(self.rnn_hidden_dim, self.output_dim)

    def init_hidden_state(self, batch_size):
        return (
            torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_dim),
            torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_dim),
        )

    def set_init_hidden_state(self, rnn_state):
        self.rnn_state = rnn_state

    def forward(self, x):
        batch_size = x.size(0)
        self.rnn_state = self.init_hidden_state(batch_size)
        x = self.fc_in(x)
        x, self.rnn_state = self.lstm(x, self.rnn_state)
        x = self.fc_out(x)
        return x

    def forward_step(self, x, rnn_state):
        x = self.fc_in(x)
        x, rnn_state = self.lstm(x, rnn_state)
        x = self.fc_out(x)
        return x, rnn_state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/dataset.hdf5")
    args = parser.parse_args()

    obs_keys = ["image", "robot_ee_pos"]
    dataset_keys = ["actions"]

    dataset = SequenceDataset(
        hdf5_path=args.data_dir,
        obs_keys=("robot_ee_pos", "image"),  # observations we want to appear in batches
        dataset_keys=(  # can optionally specify more keys here if they should appear in batches
            "actions",
        ),
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

    from torch.utils.data import DataLoader
    import torch.optim as optim

    batch_size = 128
    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,  # batches of size 100
        shuffle=True,
        # shuffle=False,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
    )

    num_epochs = 3
    gradient_steps_per_epoch = 1000

    model = TestNet()
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

            robot_ee_pos = batch["obs"]["robot_ee_pos"]
            next_robot_ee_pos = batch["next_obs"]["robot_ee_pos"]

            image = batch["obs"]["image"]
            print(image.shape)

            prediction = model(robot_ee_pos)

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
