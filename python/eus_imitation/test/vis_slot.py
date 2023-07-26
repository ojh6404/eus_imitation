import torch
import numpy as np
from torchvision.utils import save_image
from eus_imitation.base.base_nets import *
from eus_imitation.base.dataset import *

# Hyperparameters.
seed = 0
batch_size = 1
num_slots = 7
num_iterations = 3

resolution = (224, 224)
model = SlotAttentionAutoEncoder()
model.load_state_dict(torch.load("./tmp/model5.pth")["model_state_dict"])

test_set = CLEVR("test", resolution)
device = "cuda:0"
model = model.to(device)
image = test_set[2]["image"]
image = image.unsqueeze(0).to(device)
recon_combined, recons, masks, slots = model(image)

import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance

fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))
image = image.squeeze(0)
recon_combined = recon_combined.squeeze(0)
recons = recons.squeeze(0)
masks = masks.squeeze(0)
image = image.permute(1, 2, 0).cpu().numpy()
recon_combined = recon_combined.permute(1, 2, 0)
recon_combined = recon_combined.cpu().detach().numpy()
recons = recons.cpu().detach().numpy()
masks = masks.cpu().detach().numpy()
ax[0].imshow(image)
ax[0].set_title("Image")
# save original image
#
orig_image = image * 255
orig_image = orig_image.astype(np.uint8)
orig_image = Image.fromarray(orig_image)
orig_image.save("./tmp/orig.png")


ax[1].imshow(recon_combined)
ax[1].set_title("Recon.")
import cv2
from PIL import Image

for i in range(7):
    picture = recons[i] * masks[i] + (1 - masks[i])
    ax[i + 2].imshow(picture)
    ax[i + 2].set_title("Slot %s" % str(i + 1))
    # save image
    picture = picture * 255
    picture = picture.astype(np.uint8)
    picture = Image.fromarray(picture)
    picture.save("./tmp/slot%s.png" % str(i + 1))


for i in range(len(ax)):
    ax[i].grid(False)
    ax[i].axis("off")

# from torchview import draw_graph

# inputs = torch.randn(1, 3, 128, 128)

# model_graph = draw_graph(model, input_data=inputs)

# model_graph.visual_graph
