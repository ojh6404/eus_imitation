#!/usr/bin/env python3

import h5py
import numpy as np
import argparse
import json

from imitator.utils.file_utils import sort_names_by_number


def main(args):
    f = h5py.File(args.dataset, "r")

    demos = sort_names_by_number(f["data"].keys())

    traj_lengths = []
    for ep in demos:
        traj_lengths.append(len(f["data"][ep]["action"]))

    total_traj_length = np.sum(traj_lengths)

    print("=============================")
    print("Dataset info")
    print("Total demos: {}".format(len(demos)))
    print("Total trajectories: {}".format(total_traj_length))
    print("Trajectory length mean: {}".format(np.mean(traj_lengths)))
    print("Trajectory length std: {}".format(np.std(traj_lengths)))
    print("Trajectory length min: {}".format(np.min(traj_lengths)))
    print("Trajectory length max: {}".format(np.max(traj_lengths)))
    print("Max length traj index: {}".format(np.argmax(traj_lengths)))
    print("Observations: {}".format(f["data"][demos[0]]["obs"].keys()))
    print("Actions: {}".format(f["data"][demos[0]]["action"]))
    print("=============================")

    if args.verbose:
        print("=============================")
        print("Date: {}".format(f["data"].attrs["date"]))
        print(
            "Config: \n{}".format(
                json.dumps(json.loads(f["data"].attrs["config"]), indent=4)
            )
        )
        print("Demo Lenghts: {}".format(traj_lengths))
        print("=============================")
        print("=============================")
        print("Obs info")
        for obs in f["data"][demos[0]]["obs"].keys():
            print("Obs: {}".format(obs))
            print("Shape: {}".format(f["data"][demos[0]]["obs"][obs].shape))
            print("Type: {}".format(f["data"][demos[0]]["obs"][obs].dtype))
            print(
                "Modality: {}".format(
                    json.loads(f["data"].attrs["config"])["data"]["obs"][obs][
                        "modality"
                    ]
                )
            )
            print("First data: {}".format(f["data"][demos[0]]["obs"][obs][0]))
            print("")
        print("Actions max: {}".format(f["data"].attrs["action_max"]))
        print("Actions min: {}".format(f["data"].attrs["action_min"]))
        print("Actions scale: {}".format(f["data"].attrs["action_scale"]))
        print("Actions bias: {}".format(f["data"].attrs["action_bias"]))
        print("First data: {}".format(f["data"][demos[0]]["action"][0]))
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
            elif "obs_scale" in attr_name:
                print(
                    "Observations {} scale: {}".format(
                        attr_name.split("/")[-1], f["data"].attrs[attr_name]
                    )
                )
            elif "obs_bias" in attr_name:
                print(
                    "Observations {} bias: {}".format(
                        attr_name.split("/")[-1], f["data"].attrs[attr_name]
                    )
                )

        print("")
        print("Env Meta: {}".format(f["data"].attrs["env_args"]))

        print("=============================")

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="path to hdf5 dataset")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    args = parser.parse_args()
    main(args)
