from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
import os


def get_config(config_string="full,multimodal"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)

    FINETUNING_KWARGS = {
        "name": "imitator_dataset",
        "data_dir": os.path.expanduser("~/tensorflow_datasets"),
        "image_obs_keys": {
            "primary": "primary_image",
            "wrist": "wrist_image",
        },
        "state_obs_keys": ["proprio"],
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        # All actions are relative deltas, except for the last one (gripper) which is absolute
        # Specifying this is only necessary if you want to predict > 1 step into the future
        # "absolute_action_mask": [False, False, False, False, False, False, True],
        "absolute_action_mask": [True, True, True, True, True, True, True],
        # standardize_fn is dynamically loaded from a file
        # for example: "experiments/kevin/custom_standardization_transforms.py:aloha_dataset_transform"
        # "standardize_fn": "scripts/configs/dataset_transforms.py:imitator_dataset_transform",
        # If the default data loading speed is too slow, try these:
        # "num_parallel_reads": 8,  # for reading from disk / GCS
        # "num_parallel_calls": 16,  # for initial dataset construction
    }

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    elif mode == "frozen_transformer":
        frozen_keys = ("octo_transformer.BlockTransformer_0.*",)
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=2)
    pred_horizon = FieldReference(default=4)
    pretrained_path = FieldReference(default="hf://rail-berkeley/octo-small")

    config = dict(
        pretrained_path=pretrained_path,
        pretrained_step=placeholder(int),
        batch_size=128,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=5000,
        save_interval=1000,
        save_dir="octo_models",
        seed=42,
        wandb=dict(project="imitator", group=placeholder(str), entity=placeholder(str)),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        pred_horizon=pred_horizon,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        future_action_window_size=pred_horizon,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
    )
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (112, 112),  # workspace (3rd person) camera is at 256x256
            "wrist": (112, 112),  # wrist camera is at 128x128
        },
        image_augment_kwargs=[
            workspace_augment_kwargs,
            wrist_augment_kwargs,
        ],
    )
    # If the default data loading speed is too slow, try these:
    config["frame_transform_threads"] = (
        16  # for the most CPU-intensive ops (decoding, resizing, augmenting)
    )

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
