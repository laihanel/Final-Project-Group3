import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from UCF101_9_Load_Data import read_data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from dataset import VideoDataset
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import argparse
import shutil
from mypath import DATA_DIR, PROCESS_DIR, MODEL_DIR, PATH, NICKNAME
from Model_Definition import VC3D, NUM_CLASS


TEST_DIR = PROCESS_DIR + os.path.sep + 'test'
OUT_DIR = PATH + os.path.sep + 'Result'

def check_folder_exist(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    else:
        os.makedirs(folder_name)

check_folder_exist(OUT_DIR)

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

def model_definition(pretrained=False):
    '''
        Define a Keras sequential model
        Compile the model
    '''

    model = VC3D()
    # model = C3D()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    return model, criterion

def test(list_of_metrics, list_of_agg, OUT_DIR, pretrained = False):
    # Create the test instructions to
    # Load the model
    # Create the loop to validate the data

    # create a excel file to save the result
    forldernames = os.listdir(TEST_DIR)
    forldernames.sort()
    filename = []
    for foldername in forldernames:
        filepath = os.path.join(TEST_DIR, foldername)
        filename += os.listdir(filepath)
    xdf_dset_test = pd.DataFrame(filename, columns =['filenames'])


    test_loader = DataLoader(VideoDataset(dataset='ucf101', split='test', clip_len=16),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model, criterion  = model_definition(pretrained)
    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))

    model.eval()
    test_loss, steps_test = 0, 0
    pred_logits_test, real_logits_test = [], []
    #  Create the evalution
    #  Run the statistics
    #  Save the results in the Excel file
    # Remember to wirte a string con the result (NO A COLUMN FOR each )
    with torch.no_grad():
        for xdata, xtarget in test_loader:
            xdata, xtarget = xdata.to(device), xtarget.to(device)
            output = model(xdata)

            loss = criterion(output, xtarget)
            test_loss += loss.item()
            steps_test += 1

            probs = nn.Softmax(dim=1)(output)
            pred_labels_test = list(torch.max(probs, 1)[1].detach().cpu().numpy())
            real_labels_test = list(xtarget.cpu().numpy())

            pred_logits_test += pred_labels_test
            real_logits_test += real_labels_test
            print("Test Loss: {:.5f}".format(test_loss / steps_test))

    test_metrics = metrics_func(list_of_metrics, list_of_agg, real_logits_test, pred_logits_test)

    avg_test_loss = test_loss / steps_test
    xstrres = ''
    for met, dat in test_metrics.items():
        xstrres = xstrres + ' Test ' + met + ' {:.5f}'.format(dat)
    xstrres = xstrres + " - "
    print(xstrres)

    ## The following code creates a string to be saved as 1,2,3,3,
    ## This code will be used to validate the model
    class_names = open(os.path.join(DATA_DIR, 'ucf_labels.txt'), "r")
    content_list = class_names.readlines()
    content = [x[2:-1] for x in content_list]
    print('Saving result!')

    real_labels = [content[i] for i in real_logits_test]
    pred_labels = [content[i] for i in pred_logits_test]

    xdf_dset_test['labels'] = real_labels
    xdf_dset_test['results'] = pred_labels
    xdf_dset_test.to_csv(os.path.join(OUT_DIR, 'results.csv'), index=False)
    if os.path.exists(os.path.join(OUT_DIR, 'results.csv')):
        print('Result Saved!')
    else:
        print('Error! No Path Found!')


if __name__ == '__main__':
    list_of_metrics = ['acc', 'hlm']
    list_of_agg = ['sum', 'avg']


    test(list_of_metrics, list_of_agg, OUT_DIR)