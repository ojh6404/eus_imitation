#!/usr/bin/env python3

import h5py
import numpy as np

if __name__ == "__main__":
    data_file = h5py.File("./dataset.hdf5", "r")
    print(data_file.keys())
    print(data_file["data"].keys())
    print(data_file["data/demo_0"].keys())
    print(data_file["data/demo_0/obs"])
    print(data_file["data/demo_0/actions"])
    actions = data_file["data/demo_0/actions"]
    print(actions.shape)
    print(actions[0])

    print("twesjiosdjioda")
    action_min = data_file["data"].attrs["action_min"]
    action_max = data_file["data"].attrs["action_max"]
    action_scale = data_file["data"].attrs["action_scale"]
    action_bias = data_file["data"].attrs["action_bias"]

    print("action :", actions[0] * action_scale + action_bias)

    data_file.close()
