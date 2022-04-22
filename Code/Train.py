import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from UCF101_9_Load_Data import read_data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef

from dataset import VideoDataset
from Code.Model_Definition import VC3D
## gridsearch: scraed, Talas, Tensorboard, wand

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

# %%
# train_ds, test_ds = read_data()

# %% Create the model
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.convnorm1 = nn.BatchNorm3d(64)
#         self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.drop1 = nn.Dropout(0.5)
#
#         self.conv1b = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.convnorm1b = nn.BatchNorm3d(128)
#         self.pool1b = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.drop1 = nn.Dropout(0.5)
#
#         self.conv2 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.conv2b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.convnorm2 = nn.BatchNorm3d(256)
#         self.drop2 = nn.Dropout(0.5)
#
#         self.conv3 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.conv3b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.pool3 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 3, 3))
#         self.convnorm3 = nn.BatchNorm3d(512)
#         self.drop3 = nn.Dropout(0.5)
#
#         # self.conv4 = nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         # self.conv4b = nn.Conv3d(1024, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         # self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         # self.convnorm4 = nn.BatchNorm3d(1024)
#         # self.drop4 = nn.Dropout(0.5)
#
#
#         self.global_avg_pool = nn.MaxPool3d((1, 1, 2))
#         self.linear1 = nn.Linear(5120, 1280)
#         self.linear2 = nn.Linear(1280, 640)
#         self.linear3 = nn.Linear(640, 9)
#         self.softmax = nn.Softmax(dim=1)
#         self.act = torch.relu
#
#
#
#     def forward(self, x):  # x shape = (20, 3, 16, 112, 112)
#         x = self.act(self.conv1(x))
#         x = self.pool1(self.convnorm1(x))
#         x = self.act(self.conv1b(x))
#         x = self.pool1b(self.convnorm1b(x))
#         x = self.act(self.conv2b(self.act(self.conv2(x))))
#         x = self.drop2(self.pool2(self.convnorm2(x)))   # if kernal size of padding is 3, (20, 256, 1, 13, 13)
#         x = self.act(self.conv3b(self.act(self.conv3(x))))
#         x = self.drop3(self.pool3(self.convnorm3(x)))
#         # x = self.act(self.conv4b(self.act(self.conv4(x))))
#         # x = self.drop4(self.pool4(self.convnorm4(x)))
#         x = self.linear3(self.linear2(self.linear1(self.global_avg_pool(x).view(-1, 5120))))
#         # x = self.softmax()
#         return x



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

# class C3D(nn.Module):
#     """
#     The C3D network.
#     """
#
#     def __init__(self, pretrained=False):
#         super(C3D, self).__init__()
#
#         self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
#
#         self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#
#         self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#
#         self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#
#         self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
#
#         self.fc6 = nn.Linear(8192, 4096)
#         self.fc7 = nn.Linear(4096, 4096)
#         self.fc8 = nn.Linear(4096, 9)
#
#         self.dropout = nn.Dropout(p=0.5)
#
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#
#         x = self.relu(self.conv1(x))  # 20, 64, 16, 112, 112
#         x = self.pool1(x)  # 20, 64, 16, 56, 56
#
#         x = self.relu(self.conv2(x))  # 20, 128, 16, 56, 56
#         x = self.pool2(x) # 20, 128, 8, 56, 56
#
#         x = self.relu(self.conv3a(x))  # 20, 256, 8, 56, 56
#         x = self.relu(self.conv3b(x))  # 20, 256, 8, 28, 28
#         x = self.pool3(x) # 20, 256, 4, 14, 14
#
#         x = self.relu(self.conv4a(x))
#         x = self.relu(self.conv4b(x))
#         x = self.pool4(x)
#
#         x = self.relu(self.conv5a(x))
#         x = self.relu(self.conv5b(x))
#         x = self.pool5(x)  # 20, 512, 1, 4, 4
#
#         x = x.view(-1, 8192) # 20, 8192
#         x = self.relu(self.fc6(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc7(x))
#         x = self.dropout(x)
#
#         logits = self.fc8(x)
#
#         return logits

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
        model = VC3D()
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
def train(n_epoch, list_of_metrics, list_of_agg, save_on, PRETRAINED=False):
    model, optimizer, criterion, scheduler = model_definition(PRETRAINED)
    print(model)
    # Fit data to model
    train_loader = DataLoader(VideoDataset(dataset='ucf101', split='train', clip_len=16),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(VideoDataset(dataset='ucf101', split='test', clip_len=16),
                             batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    met_test_best = -1
    trigger_times = 0
    last_loss = 0

    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0
        model.train()
        pred_logits_train, real_logits_train = [], []

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
                pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / steps_train))

                probs = nn.Softmax(dim=1)(output)
                pred_labels_train = list(torch.max(probs, 1)[1].detach().cpu().numpy())
                real_labels_train = list(xtarget.cpu().numpy())

                pred_logits_train += pred_labels_train
                real_logits_train += real_labels_train

        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_logits_train, pred_logits_train)
        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+ met + ' {:.5f}'.format(dat)

        xstrres = xstrres + " - "
        print(xstrres)

        test_loss, steps_test = 0, 0
        model.eval()
        pred_logits_test, real_logits_test = [], []

        with torch.no_grad():
            with tqdm(total=len(test_loader), desc="Epoch {}".format(epoch)) as pbar:
                for xdata, xtarget in test_loader:

                    xdata, xtarget = xdata.to(device), xtarget.to(device)
                    output = model(xdata)

                    loss = criterion(output, xtarget)
                    test_loss += loss.item()
                    steps_test += 1

                    pbar.update(1)
                    pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                    probs = nn.Softmax(dim=1)(output)
                    pred_labels_test = list(torch.max(probs, 1)[1].detach().cpu().numpy())
                    real_labels_test = list(xtarget.cpu().numpy())

                    pred_logits_test += pred_labels_test
                    real_logits_test += real_labels_test

        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_logits_test, pred_logits_test)
        xstrres = "Epoch {}: ".format(epoch)

        avg_test_loss = test_loss / steps_test

        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test ' + met + ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat

        xstrres = xstrres + " - "
        print(xstrres)

        if met_test > met_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
            print("The model has been saved!")
            met_test_best = met_test

        # early stopping
        if avg_test_loss > last_loss:
        # if avg_test_loss < 0.35:
        #     break
            trigger_times += 1
            print('Trigger Times:', trigger_times)
            if trigger_times >= 5:
                print('Early stopping!\nStart to test process.')
                break
        else:
            print('Trigger Times: 0')
            trigger_times = 0
        last_loss = avg_test_loss
        # https://pythonguides.com/pytorch-early-stopping/

        # learning rate scheduler
        scheduler.step(met_test_best)


if __name__ == '__main__':
    list_of_metrics = ['acc', 'hlm']
    list_of_agg = ['sum', 'avg']
    train(n_epoch, list_of_metrics, list_of_agg, save_on='sum', PRETRAINED=False)
