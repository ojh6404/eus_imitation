#!/usr/bin/env python3

import numpy as np

# plot action and state for debug, first of train episodes
import matplotlib.pyplot as plt

episode_buffer = np.load("data/processed_train/episode_0.npy", allow_pickle=True)
action = []
state = []
for i in range(len(episode_buffer)):
    action.append(episode_buffer[i]["action"])
    state.append(episode_buffer[i]["state"])
action = np.array(action)
state = np.array(state)
plt.figure()
plt.plot(action[:, 0:3])
plt.title("action xyz")
plt.figure()
plt.plot(action[:, 3:6])
plt.title("action rpy")
plt.figure()
plt.plot(action[:, 6])
plt.title("action gripper")
plt.figure()
plt.plot(state[:, 0:3])
plt.title("state xyz")
plt.plot(state[:, 3:6])
plt.title("state rpy")
plt.figure()
plt.figure()
plt.plot(state[:, 6])
plt.title("state gripper")
plt.show()
