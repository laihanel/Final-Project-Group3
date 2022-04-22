import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from UCF101_9_Load_Data import read_data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from dataset import VideoDataset
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef

# %% HyperParameters
NICKNAME = 'Trial_9class'
OUTPUTS_a = 9  # Subject to change, now we manually picked 9 classes to classify
BATCH_SIZE = 20
LR = 0.001
n_epoch = 50
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PRETRAINED = False
SAVE_MODEL = True
# PRETRAINED = models.efficientnet_b4(pretrained=True)


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
        self.softmax = nn.Softmax(dim=1)
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
        x = self.linear3(self.linear2(self.linear1(self.global_avg_pool(x).view(-1, 5120))))
        # x = self.softmax()
        return x


def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet = matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet = -hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict


def read_video()