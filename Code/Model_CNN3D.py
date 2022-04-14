import torch
import torch.nn as nn

from UCF101_9_Load_Data import train_ds, test_ds

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
