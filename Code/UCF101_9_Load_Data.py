# %%
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import UCF101
import av
from torchvision import transforms
import os

# %%------------------------------------------------------------------------------------------------------------------

OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
LABEL_DIR = os.getcwd() + os.path.sep + 'Label' + os.path.sep
sep = os.path.sep

os.chdir(OR_PATH)  # Come back to the folder where the code resides , all files will be left on this directory

# %%
# From: https://www.kaggle.com/code/pevogam/starter-ucf101-with-pytorch/notebook
frames_per_clip = 5
step_between_clips = 1
batch_size = 32

# %%
tfs = transforms.Compose([
    # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
    # scale in [0, 1] of type float
    transforms.Lambda(lambda x: x / 255.),
    # reshape into (T, C, H, W) for easier convolutions
    transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
    # rescale to the most common size
    transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
])


# %%
def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


# %%
# create train loader (allowing batches and other extras)
train_dataset = UCF101(DATA_DIR, LABEL_DIR, frames_per_clip=frames_per_clip,
                       step_between_clips=step_between_clips, train=True, transform=tfs)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=custom_collate)
# create test loader (allowing batches and other extras)
test_dataset = UCF101(DATA_DIR, LABEL_DIR, frames_per_clip=frames_per_clip,
                      step_between_clips=step_between_clips, train=False, transform=tfs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=custom_collate)

# %%
print(f"Total number of train samples: {len(train_dataset)}")
print(f"Total number of test samples: {len(test_dataset)}")
print(f"Total number of (train) batches: {len(train_loader)}")
print(f"Total number of (test) batches: {len(test_loader)}")
print()
