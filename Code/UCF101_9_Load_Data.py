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
def read_data():
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
    return train_loader, test_loader


train_ds, test_ds = read_data()

# Create the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3D(3, 32, kernel_size=(3, 3, 2))
        self.convnorm1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv3D(32, 64, kernel_size=(3, 3, 2))
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.convnorm2 = nn.BatchNorm3d(64)
        self.drop2 = nn.Dropout(0.5)

        self.global_avg_pool = nn.AvgPool3d((1, 1, 2))
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 9)
        self.act = torch.relu


    def forward(self, x):
        x = self.drop1(self.pool1(self.convnorm1(self.act(self.conv1(x)))))
        x = self.drop2(self.pool2(self.convnorm2(self.act(self.conv2(x)))))
        # x = self.pool3(self.dropout3(self.act(self.conv4(self.act(self.conv3(x))))))
        return self.linear2(self.linear1(self.global_avg_pool(x).view(-1, -1, 64)))

LR = 0.001
n_epoch = 10
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = CNN()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
# Compile the model

model.summary()
# Fit data to model
for epoch in range(n_epoch):
    for xdata, xtarget in train_ds:
        xdata, xtarget = xdata.to(device), xtarget.to(device)
        optimizer.zero_grad()
        output = model(xdata)
        loss = criterion(output, xtarget)
        loss.backward()
        optimizer.step()
        print(loss)