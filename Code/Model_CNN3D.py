import timeit
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from UCF101_9_Load_Data import train_ds, test_ds

# %% HyperParameters
NICKNAME = 'Trial_9class'
OUTPUTS_a = 9  # Subject to change, now we manually picked 9 classes to classify
LR = 0.001
n_epoch = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PRETRAINED = models.efficientnet_b4(pretrained=True)

# %% Create the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 2))
        self.convnorm1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 2))
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
print(model)
# Fit data to model
# for epoch in range(n_epoch):
#     for xdata, xtarget in train_ds:
#         xdata, xtarget = xdata.to(device), xtarget.to(device)
#         optimizer.zero_grad()
#         output = model(xdata)
#         loss = criterion(output, xtarget)
#         loss.backward()
#         optimizer.step()
#         print(loss)

for epoch in range(n_epoch):
    # each epoch has a training and validation step
    for phase in ['train', 'val']:
        start_time = timeit.default_timer()

        # reset the running loss and corrects
        running_loss = 0.0
        running_corrects = 0.0

        # set model to train() or eval() mode depending on whether it is trained
        # or being validated. Primarily affects layers such as BatchNorm or Dropout.
        if phase == 'train':
            # scheduler.step() is to be called once every epoch during training
            # optimizer.step()
            scheduler.step()
            model.train()
        else:
            model.eval()

        for inputs, labels in tqdm(trainval_loaders[phase]):
            # move inputs and labels to the device the training is taking place on
            inputs = Variable(inputs, requires_grad=True).to(device)
            labels = Variable(labels).to(device)
            optimizer.zero_grad()

            if phase == 'train':
                outputs = model(inputs)
            else:
                with torch.no_grad():
                    outputs = model(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss = criterion(outputs, labels)
            #             loss = criterion(preds, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        #             print(f'running loss: {running_loss},\n inputs: {inputs} \n output: {outputs} {len(outputs)} \n labels: {labels} {len(labels)}')

        epoch_loss = running_loss / trainval_sizes[phase]
        epoch_acc = running_corrects.double() / trainval_sizes[phase]

        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")