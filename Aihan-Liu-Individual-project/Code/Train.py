import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from UCF101_9_Load_Data import read_data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef

from mypath import NICKNAME
from dataset import VideoDataset
from Model_Definition import VC3D, NUM_CLASS
## gridsearch: scraed, Talas, Tensorboard, wand

# %% HyperParameters
BATCH_SIZE = 20
LR = 0.001
n_epoch = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED = False
SAVE_MODEL = True

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

# %% Utility Functions
def model_definition(pretrained=False):
    '''
        Define a Keras sequential model
        Compile the model
    '''

    if pretrained:
        model = PRETRAINED
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASS)
    else:
        model = VC3D()
        # model = C3D()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)

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
                output = model(xdata)  # does not contain softmax layer

                loss = criterion(output, xtarget)  # crossentropy loss has softmax layer
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                steps_train += 1
                # print(loss)
                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / steps_train))

                probs = nn.Softmax(dim=1)(output)  # add a softmax layer for output model
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

                    probs = nn.Softmax(dim=1)(output)  # add the softmax layer for output model
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
