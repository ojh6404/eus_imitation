#!/usr/bin/env python3

import h5py
import numpy as np
import argparse
import re
from PIL import Image
import matplotlib.pyplot as plt

def extract_number(name):
    match = re.search(r"\d+", name)
    if match:
        return int(match.group())
    else:
        return 0


def sort_names_by_number(names):
    sorted_names = sorted(names, key=extract_number)
    return sorted_names

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
                for elem in np.array(f["mask/{}".format("train")][:])
            ]
        if "valid" in f["mask"].keys():
            valid_mask = [
                elem.decode("utf-8")
                for elem in np.array(f["mask/{}".format("valid")][:])
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
        print("First action data: {}".format(f["data"][demos[0]]["actions"][0]))
        print("Second action data: {}".format(f["data"][demos[0]]["actions"][1]))
    print("=============================")

    print("=============================")
    print("Demo Lenghts: {}".format(traj_lengths))
    print("=============================")
    print("=============================")
    print("Obs info")
    for obs in f["data"][demos[0]]["obs"].keys():
        print("Obs: {}".format(obs))
        print("Shape: {}".format(f["data"][demos[0]]["obs"][obs].shape))
        print("Type: {}".format(f["data"][demos[0]]["obs"][obs].dtype))
        if "image" in obs:
            Image.fromarray(f["data"][demos[0]]["obs"][obs][0]).show()
        else:
            print("First data: {}".format(f["data"][demos[0]]["obs"][obs][0]))
            print("Second data: {}".format(f["data"][demos[0]]["obs"][obs][1]))
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

    # visualize image
    if args.visualize:
        # action
        action_dim = f["data"][demos[0]]["actions"].shape[-1]
        data = f["data"][demos[0]]["actions"]
        fig_action = plt.figure()
        fig_action.suptitle("Actions")
        for i in range(action_dim):
            plt.subplot(action_dim, 1, i + 1)
            plt.plot(data[:, i])

        # obs
        for obs in f["data"][demos[0]]["obs"].keys():
            if "image" in obs:
                print(
                    "Image observation: {}, Shape: {}".format(
                        obs, f["data"][demos[0]]["obs"][obs][0].shape
                    )
                )
                vis_img = f["data"][demos[0]]["obs"][obs][::10]
                img = Image.fromarray(np.concatenate(vis_img, axis=1))
                img.show()
            else:
                obs_dim = f["data"][demos[0]]["obs"][obs].shape[-1]
                print("Float observation: {}, Shape: {}".format(obs, obs_dim))
                data = f["data"][demos[0]]["obs"][obs]
                fig_obs = plt.figure()
                fig_obs.suptitle(obs)
                for i in range(obs_dim):
                    plt.subplot(obs_dim, 1, i + 1)
                    plt.plot(data[:, i])
                plt.title(obs)
                plt.show()
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument(
        "-v", "--visualize", action="store_true", help="visualize image"
    )
    args = parser.parse_args()
    main(args)
