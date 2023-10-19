#!/usr/bin/env python3

import h5py
import numpy as np
import argparse

from imitator.utils.file_utils import sort_names_by_number


def main(args):
    f = h5py.File(args.dataset, "r")

    demos = sort_names_by_number(f["data"].keys())

    traj_lengths = []
    for ep in demos:
        for obs in f["data"][ep]["obs"].keys():
            traj_lengths.append(f["data"][ep]["obs"][obs].shape[0])
            break

    total_traj_length = np.sum(traj_lengths)

    train_mask = None
    valid_mask = None
    if "mask" in f.keys():
        if "train" in f["mask"].keys():
            train_mask = [
                elem.decode("utf-8")
                for elem in np.array(
                    f["mask/{}".format("train")][:]
                )
            ]
        if "valid" in f["mask"].keys():
            valid_mask = [
                elem.decode("utf-8")
                for elem in np.array(
                    f["mask/{}".format("valid")][:]
                )
            ]

    print("=============================")
    print("Dataset info")
    print("Total demos: {}".format(len(demos)))
    print("Total trajectories: {}".format(total_traj_length))
    if train_mask is not None:
        train_traj = 0
        for ep in train_mask:
            train_traj += f["data/{}".format(ep)]["actions"].shape[0]
        print("Train trajectories: {}".format(train_traj))
    if valid_mask is not None:
        valid_traj = 0
        for ep in valid_mask:
            valid_traj += f["data/{}".format(ep)]["actions"].shape[0]
        print("Valid trajectories: {}".format(valid_traj))
    print("Trajectory length mean: {}".format(np.mean(traj_lengths)))
    print("Trajectory length std: {}".format(np.std(traj_lengths)))
    print("Trajectory length min: {}".format(np.min(traj_lengths)))
    print("Trajectory length max: {}".format(np.max(traj_lengths)))
    print("Max length traj index: {}".format(np.argmax(traj_lengths)))
    print("Observations: {}".format(f["data"][demos[0]]["obs"].keys()))
    # f["data"][demos[0]] has actions, then print
    if "actions" in f["data"][demos[0]].keys():
        print("Actions: {}".format(f["data"][demos[0]]["actions"]))
    print("=============================")

    if args.verbose:
        print("=============================")
        print("Demo Lenghts: {}".format(traj_lengths))
        print("=============================")
        print("=============================")
        print("Obs info")
        for obs in f["data"][demos[0]]["obs"].keys():
            print("Obs: {}".format(obs))
            print("Shape: {}".format(f["data"][demos[0]]["obs"][obs].shape))
            print("Type: {}".format(f["data"][demos[0]]["obs"][obs].dtype))
            print("First data: {}".format(f["data"][demos[0]]["obs"][obs][0]))
            print("")
        if "actions" in f["data"][demos[0]].keys():
            print("First data: {}".format(f["data"][demos[0]]["actions"][0]))
        print("")
        for attr_name in f["data"].attrs.keys():
            if "obs_max" in attr_name:
                print(
                    "Observations {} max: {}".format(
                        attr_name.split("/")[-1], f["data"].attrs[attr_name]
                    )
                )
            elif "obs_min" in attr_name:
                print(
                    "Observations {} min: {}".format(
                        attr_name.split("/")[-1], f["data"].attrs[attr_name]
                    )
                )
        print("")
        print("=============================")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_args()
    main(args)
