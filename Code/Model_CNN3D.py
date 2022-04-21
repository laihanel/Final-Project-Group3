import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from UCF101_9_Load_Data import read_data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from dataset import VideoDataset

## gridsearch: scraed, Talas, Tensorboard, wand

# %% HyperParameters
NICKNAME = 'Trial_9class'
OUTPUTS_a = 9  # Subject to change, now we manually picked 9 classes to classify
LR = 0.001
n_epoch = 10
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PRETRAINED = False
# PRETRAINED = models.efficientnet_b4(pretrained=True)

# %%
# train_ds, test_ds = read_data()

# %% Create the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.convnorm1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.drop1 = nn.Dropout(0.5)

        self.conv1b = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.convnorm1b = nn.BatchNorm3d(128)
        self.pool1b = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.convnorm2 = nn.BatchNorm3d(256)
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 3, 3))
        self.convnorm3 = nn.BatchNorm3d(512)
        self.drop3 = nn.Dropout(0.5)

        # self.conv4 = nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv4b = nn.Conv3d(1024, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # self.convnorm4 = nn.BatchNorm3d(1024)
        # self.drop4 = nn.Dropout(0.5)


        self.global_avg_pool = nn.MaxPool3d((1, 1, 2))
        self.linear1 = nn.Linear(5120, 1280)
        self.linear2 = nn.Linear(1280, 640)
        self.linear3 = nn.Linear(640, 9)
        self.act = torch.relu


    def forward(self, x):  # x shape = (20, 3, 16, 112, 112)
        x = self.act(self.conv1(x))
        x = self.pool1(self.convnorm1(x))
        x = self.act(self.conv1b(x))
        x = self.pool1b(self.convnorm1b(x))
        x = self.act(self.conv2b(self.act(self.conv2(x))))
        x = self.drop2(self.pool2(self.convnorm2(x)))   # if kernal size of padding is 3, (20, 256, 1, 13, 13)
        x = self.act(self.conv3b(self.act(self.conv3(x))))
        x = self.drop3(self.pool3(self.convnorm3(x)))
        # x = self.act(self.conv4b(self.act(self.conv4(x))))
        # x = self.drop4(self.pool4(self.convnorm4(x)))
        return self.linear3(self.linear2(self.linear1(self.global_avg_pool(x).view(-1, 5120))))

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, pretrained=False):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 9)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv1(x))  # 20, 64, 16, 112, 112
        x = self.pool1(x)  # 20, 64, 16, 56, 56

        x = self.relu(self.conv2(x))  # 20, 128, 16, 56, 56
        x = self.pool2(x) # 20, 128, 8, 56, 56

        x = self.relu(self.conv3a(x))  # 20, 256, 8, 56, 56
        x = self.relu(self.conv3b(x))  # 20, 256, 8, 28, 28
        x = self.pool3(x) # 20, 256, 4, 14, 14

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)  # 20, 512, 1, 4, 4

        x = x.view(-1, 8192) # 20, 8192
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

# %% Utility Functions
def model_definition(pretrained=False):
    '''
        Define a Keras sequential model
        Compile the model
    '''

    if pretrained:
        model = PRETRAINED
        model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
    else:
        model = CNN()
        # model = C3D()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    save_model(model)

    return model, optimizer, criterion, scheduler


def save_model(model):
    '''
      Print Model Summary
    '''

    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))

# %% Compile the model
model, optimizer, criterion, scheduler = model_definition(PRETRAINED)
print(model)
# Fit data to model
train_loader = DataLoader(VideoDataset(dataset='ucf101', split='train',clip_len=16), batch_size=20, shuffle=True, num_workers=1)

for epoch in range(n_epoch):
    train_loss, steps_train = 0, 0
    with tqdm(total=len(train_loader), desc="Epoch {}".format(epoch)) as pbar:
        for xdata, xtarget in train_loader:
            xdata = Variable(xdata, requires_grad=True).to(device)
            xtarget = Variable(xtarget).to(device)
            # xdata, xtarget = xdata.to(device), xtarget.to(device)
            optimizer.zero_grad()
            output = model(xdata)
            loss = criterion(output, xtarget)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            steps_train += 1
            # print(loss)
            pbar.update(1)
            pbar.set_postfix_str("Test Loss: {:.5f}".format(train_loss / steps_train))

