#!/usr/bin/env python3

import os
from typing import Iterator, Tuple, Any
import numpy as np
import tensorflow_datasets as tfds
import h5py
from imitator.utils.file_utils import get_config_from_project_name, get_data_hdf5, extract_number

project_name = os.environ.get("PROJECT_NAME", "imitator")
dataset = os.environ.get("DATASET", "dataset.hdf5")
imitator_config = get_config_from_project_name(project_name)
hdf5_path = get_data_hdf5(project_name, dataset)
obs_keys = list(imitator_config.obs.keys())

print(f"Building TFDS dataset for project: {project_name}, dataset: {hdf5_path}")


class ImitatorDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_beam = False  # TODO

    def _info(self) -> tfds.core.DatasetInfo:
        action_dim = imitator_config.actions.dim
        tfds_obs = {}
        for obs_key in obs_keys:
            obs_config = imitator_config.obs[obs_key]
            if obs_config.modality == "ImageModality":  # when ImageModality
                tfds_obs[obs_key] = tfds.features.Image(
                    shape=tuple(obs_config.dim),
                    dtype=np.uint8,
                    encoding_format="png",
                    doc="Camera RGB observation.",
                )
            elif obs_config.modality == "FloatVectorModality":  # when FloatVectorModality
                tfds_obs[obs_key] = tfds.features.Tensor(
                    shape=(obs_config.dim,),
                    dtype=np.float32,
                    doc="Float Vector observation.",
                )
            else:  # not implemented yet
                raise NotImplementedError
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(tfds_obs),
                            "action": tfds.features.Tensor(
                                shape=(action_dim,),
                                dtype=np.float32,
                                doc="Action taken by the agent.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step of the episode."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode."),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(doc="Language Instruction."),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(key="train"),
            "val": self._generate_examples(key="valid"),
        }

    def _generate_examples(self, key) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        f = h5py.File(hdf5_path, "r")
        # language_instruction = f.attrs.get("language_instruction", "Dummy Instruction")
        language_instruction = imitator_config.task.get("language_instruction", "Dummy Instruction")

        demos = sorted(f["data"].keys(), key=extract_number)
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(key)][:])]

        def _parse_example(demo_key):
            # load raw data --> this should change for your dataset
            demo = f["data"][demo_key]
            demo_len = demo["actions"].shape[0]

            obs_config = imitator_config.obs
            obs_dtype = {
                obs_key: np.uint8 if obs_config[obs_key].modality == "ImageModality" else np.float32
                for obs_key in obs_keys
            }

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(demo_len):
                episode.append(
                    {
                        "observation": {
                            obs_key: demo["obs"][obs_key][i].astype(obs_dtype[obs_key]) for obs_key in obs_keys
                        },
                        "action": demo["actions"][i].astype(np.float32),
                        "discount": 1.0,
                        "reward": float(i == (demo_len - 1)),
                        "is_first": i == 0,
                        "is_last": i == (demo_len - 1),
                        "is_terminal": i == (demo_len - 1),
                        "language_instruction": language_instruction,
                    }
                )
            return demo_key, dict(steps=episode)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        if self.use_beam:
            beam = tfds.core.lazy_imports.apache_beam
            return beam.Create(demos) | beam.Map(_parse_example)
        else:  # for smallish datasets, use single-thread parsing
            for demo in demos:
                yield _parse_example(demo)
