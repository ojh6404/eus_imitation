import cv2
import jax
import tensorflow_datasets as tfds
import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from octo.model.octo_model import OctoModel
from functools import partial

model = OctoModel.load_pretrained("/home/leus/.imitator/kubota_spacenav/octo_models_rgb", 2999)
policy_fn = jax.jit(model.sample_actions)
act_stats = model.dataset_statistics["action"]
proprio_stats = model.dataset_statistics["proprio"]
print(proprio_stats['std'])

# create RLDS dataset builder
builder = tfds.builder_from_directory(builder_dir='/home/leus/tensorflow_datasets/imitator_dataset/1.0.0')
ds = builder.as_dataset(split='train[:1]')

# sample episode + resize to 256x256 (default third-person cam resolution)
episode = next(iter(ds))
steps = list(episode['steps'])
images = [cv2.resize(np.array(step['observation']['head_image']), (224, 224)) for step in steps]
proprios = [np.array(step['observation']['proprio']) for step in steps]

# extract goal image & language instruction
goal_image = images[-1]
language_instruction = steps[0]['language_instruction'].numpy().decode()

# visualize episode
print(f'Instruction: {language_instruction}')

WINDOW_SIZE = 1

# create `task` dict
# task = model.create_tasks(goals={"head_image": goal_image[None]})   # for goal-conditioned, (1, H, W, C)
task = model.create_tasks(texts=[language_instruction])                  # for language conditioned

# run inference loop, this model only uses single image observations for bridge
# collect predicted and true actions
pred_actions, true_actions = [], []
for step in tqdm.trange(len(images) - (WINDOW_SIZE - 1)):
    input_images = np.stack(images[step:step+WINDOW_SIZE])[None] # (1,WINDOW_SIZE,H,W,C)
    input_proprios = np.stack(proprios[step:step+WINDOW_SIZE])[None] # (1,WINDOW_SIZE,DIM)
    input_proprios = (input_proprios - proprio_stats['mean'][None][None]) - proprio_stats['std'][None][None]
    observation = {
        'image_primary': input_images,
        'prorpio' : input_proprios,
        'pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool) # (1,WINDOW_SIZE)
    }

    # this returns *normalized* actions --> we need to unnormalize using the dataset statistics
    # norm_actions = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
    norm_actions = policy_fn(jax.tree_map(lambda x: x, observation), task, rng=jax.random.PRNGKey(0))
    norm_actions = norm_actions[0]   # remove batch

    actions = (
        norm_actions * model.dataset_statistics['action']['std']
        + model.dataset_statistics['action']['mean']
    )


    pred_actions.append(actions)
    final_window_step = step + WINDOW_SIZE - 1

    true_actions.append(steps[final_window_step]['action'])

ACTION_DIM_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp']

# build image strip to show above actions
img_strip = np.concatenate(np.array(images[::3]), axis=1)

# set up plt figure
figure_layout = [
    ['image'] * len(ACTION_DIM_LABELS),
    ACTION_DIM_LABELS
]
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplot_mosaic(figure_layout)
fig.set_size_inches([45, 10])

# plot actions
pred_actions = np.array(pred_actions).squeeze()
true_actions = np.array(true_actions).squeeze()
for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
  # actions have batch, horizon, dim, in this example we just take the first action for simplicity
  axs[action_label].plot(pred_actions[:, 0, action_dim], label='predicted action')
  axs[action_label].plot(true_actions[:, action_dim], label='ground truth')
  axs[action_label].set_title(action_label)
  axs[action_label].set_xlabel('Time in one episode')

axs['image'].imshow(img_strip)
axs['image'].set_xlabel('Time in one episode (subsampled)')
plt.legend()
