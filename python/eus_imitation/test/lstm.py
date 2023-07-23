#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        self.input_dim = 3
        self.rnn_input_dim = 4
        self.output_dim = 3
        self.seq_len = 5
        self.rnn_hidden_dim = 10
        self.rnn_layers = 2

        self.fc_in = nn.Linear(self.input_dim, self.rnn_input_dim)
        self.lstm = nn.LSTM(
            self.rnn_input_dim,
            self.rnn_hidden_dim,
            self.rnn_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(self.rnn_hidden_dim, self.output_dim)

    # def init_hidden_state(self, batch_size):
    #     # self.rnn_state = (
    #     #     torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_dim),
    #     #     torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_dim),
    #     # )
    #     # random init
    #     self.rnn_state = (
    #         torch.rand(self.rnn_layers, batch_size, self.rnn_hidden_dim),
    #         torch.rand(self.rnn_layers, batch_size, self.rnn_hidden_dim),
    #     )

    def set_init_hidden_state(self, rnn_state):
        self.rnn_state = rnn_state

    def forward(self, x):
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
    test_net = TestNet()

    batch_size = 4
    seq_len = 5
    feature_dim = 3
    # whole sequence
    test_init_state = (
        torch.rand(test_net.rnn_layers, batch_size, test_net.rnn_hidden_dim),
        torch.rand(test_net.rnn_layers, batch_size, test_net.rnn_hidden_dim),
    )
    test_input = torch.rand(batch_size, seq_len, feature_dim)
    print("whole sequence")
    with torch.no_grad():
        test_net.set_init_hidden_state(test_init_state)
        test_output = test_net(test_input)
        for i in range(seq_len):
            print(test_output[:, i, :])

    print("step by step")
    # step by step
    with torch.no_grad():
        test_input_step = test_input[:, 0, :].unsqueeze(1)
        test_net.set_init_hidden_state(test_init_state)
        test_output, test_net.rnn_state = test_net.forward_step(
            test_input_step, test_net.rnn_state
        )
        print(test_output)
        test_input_step = test_input[:, 1, :].unsqueeze(1)
        test_output, test_net.rnn_state = test_net.forward_step(
            test_input_step, test_net.rnn_state
        )
        print(test_output)
        test_input_step = test_input[:, 2, :].unsqueeze(1)
        test_output, test_net.rnn_state = test_net.forward_step(
            test_input_step, test_net.rnn_state
        )
        print(test_output)
        test_input_step = test_input[:, 3, :].unsqueeze(1)
        test_output, test_net.rnn_state = test_net.forward_step(
            test_input_step, test_net.rnn_state
        )
        print(test_output)
        test_input_step = test_input[:, 4, :].unsqueeze(1)
        test_output, test_net.rnn_state = test_net.forward_step(
            test_input_step, test_net.rnn_state
        )
        print(test_output)

    # with torch.no_grad():
    #     test_input = torch.ones(batch_size, 1, feature_dim)
    #     test_net.set_init_hidden_state(test_init_state)
    #     test_output, test_net.rnn_state = test_net.forward_step(
    #         test_input, test_net.rnn_state
    #     )
    #     print(test_output)
    #     test_output, test_net.rnn_state = test_net.forward_step(
    #         test_input, test_net.rnn_state
    #     )
    #     print(test_output)
    #     test_output, test_net.rnn_state = test_net.forward_step(
    #         test_input, test_net.rnn_state
    #     )
    #     print(test_output)
    #     test_output, test_net.rnn_state = test_net.forward_step(
    #         test_input, test_net.rnn_state
    #     )
    #     print(test_output)
    #     test_output, test_net.rnn_state = test_net.forward_step(
    #         test_input, test_net.rnn_state
    #     )
    #     print(test_output)
