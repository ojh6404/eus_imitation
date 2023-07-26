#!/usr/bin/env python3
import os
import argparse
from eus_imitation.base.dataset import *
from eus_imitation.base.base_nets import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument(
    "--model_dir", default="./model10.pth", type=str, help="where to save models"
)
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument(
    "--num_slots", default=7, type=int, help="Number of slots in Slot Attention."
)
parser.add_argument(
    "--num_iterations", default=3, type=int, help="Number of attention iterations."
)
parser.add_argument("--hid_dim", default=64, type=int, help="hidden dimension size")
parser.add_argument("--learning_rate", default=0.0004, type=float)
parser.add_argument(
    "--warmup_steps",
    default=10000,
    type=int,
    help="Number of warmup steps for the learning rate.",
)
parser.add_argument(
    "--decay_rate", default=0.5, type=float, help="Rate for the learning rate decay."
)
parser.add_argument(
    "--decay_steps",
    default=100000,
    type=int,
    help="Number of steps for the learning rate decay.",
)
parser.add_argument(
    "--num_workers", default=4, type=int, help="number of workers for loading data"
)
parser.add_argument(
    "--num_epochs", default=1000, type=int, help="number of workers for loading data"
)

opt = parser.parse_args()
resolution = (224, 224)
# resolution = (128, 128)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = CLEVR("train", resolution)
model = SlotAttentionAutoEncoder().to(device)
# model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])

criterion = nn.MSELoss()

params = [{"params": model.parameters()}]

train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers
)

optimizer = optim.Adam(params, lr=opt.learning_rate)

start = time.time()
i = 0
for epoch in range(opt.num_epochs):
    model.train()

    total_loss = 0

    for sample in tqdm(train_dataloader):
        i += 1

        if i < opt.warmup_steps:
            learning_rate = opt.learning_rate * (i / opt.warmup_steps)
        else:
            learning_rate = opt.learning_rate

        learning_rate = learning_rate * (opt.decay_rate ** (i / opt.decay_steps))

        optimizer.param_groups[0]["lr"] = learning_rate

        image = sample["image"].to(device)
        recon_combined, recons, masks, slots = model(image)

        # print("recon_combined", recon_combined.shape)
        # print("recons", recons.shape)
        # print("masks", masks.shape)
        # print("slots", slots.shape)

        loss = criterion(recon_combined, image)
        total_loss += loss.item()

        del recons, masks, slots

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss /= len(train_dataloader)

    print(
        "Epoch: {}, Loss: {}, Time: {}".format(
            epoch, total_loss, datetime.timedelta(seconds=time.time() - start)
        )
    )

    # if not epoch % 10:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        "./tmp/model{}.pth".format(epoch),
    )
