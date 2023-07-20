#!/usr/bin/env python3

import h5py
import numpy as np

if __name__ == "__main__":
    data_file = h5py.File("rosbag-0.hdf5", "r")
    print(data_file.keys())
    # print(data_file["JointStates"].shape)
    # print(data_file["RobotAction"].shape)
    print(data_file["RGBCompressedImage"].shape)

    data_file.close()
