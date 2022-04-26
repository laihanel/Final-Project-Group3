import torch
import torch.nn as nn

NUM_CLASS = 9  # Subject to change, now we manually picked 9 classes to classify
# %% Create the model
class VC3D(nn.Module):
    def __init__(self):
        super(VC3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.convnorm1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv1b = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.convnorm1b = nn.BatchNorm3d(128)
        self.pool1b = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

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

        self.global_max_pool = nn.MaxPool3d((1, 1, 2))
        self.linear1 = nn.Linear(5120, 1280)
        self.linear2 = nn.Linear(1280, 640)
        self.linear3 = nn.Linear(640, NUM_CLASS)
        self.softmax = nn.Softmax(dim=1)
        self.act = torch.relu

    def forward(self, x):  # x shape = (20, 3, 16, 120, 120)
        x = self.act(self.conv1(x))
        x = self.pool1(self.convnorm1(x))
        x = self.act(self.conv1b(x))
        x = self.pool1b(self.convnorm1b(x))
        x = self.act(self.conv2b(self.act(self.conv2(x))))
        x = self.drop2(self.pool2(self.convnorm2(x)))
        x = self.act(self.conv3b(self.act(self.conv3(x))))
        x = self.drop3(self.pool3(self.convnorm3(x)))  # x.shape = (20, 512, 1, 5, 5)
        # x = self.act(self.conv4b(self.act(self.conv4(x))))
        # x = self.drop4(self.pool4(self.convnorm4(x)))
        x = self.global_max_pool(x)  # After pooling x.shape = (20, 512, 1, 5, 2)
        x = self.linear3(self.linear2(self.linear1(x.view(-1, 5120))))
        # x = self.softmax()
        return x


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
        self.fc7 = nn.Linear(4096, 2048)
        self.fc8 = nn.Linear(2048, NUM_CLASS)

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

        x = x.view(-1, 8192)  # 20, 8192
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

