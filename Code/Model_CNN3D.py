import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from UCF101_9_Load_Data import read_data

# %% HyperParameters
NICKNAME = Trial_9class
OUTPUTS_a = 9  # Subject to change, now we manually picked 9 classes to classify
LR = 0.001
n_epoch = 10
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PRETRAINED = None
# PRETRAINED = models.efficientnet_b4(pretrained=True)

# %%
train_ds, test_ds = read_data()

# %% Create the model
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
