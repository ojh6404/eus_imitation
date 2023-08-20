#!/usr/bin/env python3

import numpy as np

# plot action and state for debug, first of train episodes
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("-pn", "--project_name", type=str)
parser.add_argument("-d", "--data", type=str)
args = parser.parse_args()

# episode_buffer = np.load("data/train/episode_0.npy", allow_pickle=True)
# episode_buffer = np.load(args.project_name + "/data/train/episode_0.npy", allow_pickle=True)
episode_buffer = np.load(args.data, allow_pickle=True)
action = []
state = []
for i in range(len(episode_buffer)):
    action.append(episode_buffer[i]["action"])
    state.append(episode_buffer[i]["state"])
action = np.array(action)
state = np.array(state)


plt.figure()
plt.plot(action[:, 0:3])
plt.legend(["x", "y", "z"])
plt.title("action xyz")

plt.figure()
plt.plot(action[:, 3:6])
plt.legend(["r", "p", "y"])
plt.title("action rpy")

plt.figure()
plt.plot(action[:, 6])
plt.legend(["gripper"])
plt.title("action gripper")

plt.figure()
plt.plot(action[:, 7])
plt.legend(["terminal action"])
plt.title("action terminal")

plt.figure()
plt.plot(state[:, 0:3])
plt.legend(["x", "y", "z"])
plt.title("state xyz")

plt.figure()
plt.plot(state[:, 3:6])
plt.legend(["r", "p", "y"])
plt.title("state rpy")

plt.figure()
plt.plot(state[:, 6])
plt.legend(["gripper"])
plt.title("state gripper")
plt.show()
