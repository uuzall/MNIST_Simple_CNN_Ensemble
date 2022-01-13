import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn_3(nn.Module):
    def __init__(self):
        super(cnn_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), bias=False)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=(3, 3), bias=False)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=(3, 3), bias=False)
        self.conv4 = nn.Conv2d(64, 80, kernel_size=(3, 3), bias=False)
        self.conv5 = nn.Conv2d(80, 96, kernel_size=(3, 3), bias=False)
        self.conv6 = nn.Conv2d(96, 112, kernel_size=(3, 3), bias=False)
        self.conv7 = nn.Conv2d(112, 128, kernel_size=(3, 3), bias=False)
        self.conv8 = nn.Conv2d(128, 144, kernel_size=(3, 3), bias=False)
        self.conv9 = nn.Conv2d(144, 160, kernel_size=(3, 3), bias=False)
        self.conv10 = nn.Conv2d(160, 176, kernel_size=(3, 3), bias=False)

        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(48)
        self.batch3 = nn.BatchNorm2d(64)
        self.batch4 = nn.BatchNorm2d(80)
        self.batch5 = nn.BatchNorm2d(96)
        self.batch6 = nn.BatchNorm2d(112)
        self.batch7 = nn.BatchNorm2d(128)
        self.batch8 = nn.BatchNorm2d(144)
        self.batch9 = nn.BatchNorm2d(160)
        self.batch10 = nn.BatchNorm2d(176)
        self.batch11 = nn.BatchNorm1d(10)

        self.linear = nn.Linear(11264, 10, bias=False)

    def forward(self, x):
        x = (x - 0.5) * 2.0
        out = F.relu(self.batch1(self.conv1(x)))
        out = F.relu(self.batch2(self.conv2(out)))
        out = F.relu(self.batch3(self.conv3(out)))
        out = F.relu(self.batch4(self.conv4(out)))
        out = F.relu(self.batch5(self.conv5(out)))
        out = F.relu(self.batch6(self.conv6(out)))
        out = F.relu(self.batch7(self.conv7(out)))
        out = F.relu(self.batch8(self.conv8(out)))
        out = F.relu(self.batch9(self.conv9(out)))
        out = F.relu(self.batch10(self.conv10(out)))
        out = torch.flatten(out.permute(0, 2, 3, 1), 1)
        out = self.batch11(self.linear(out))

        return F.log_softmax(out, dim=1)

class cnn_5(nn.Module):
    def __init__(self):
        super(cnn_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), bias=False)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=(5, 5), bias=False)
        self.conv4 = nn.Conv2d(96, 128, kernel_size=(5, 5), bias=False)
        self.conv5 = nn.Conv2d(128, 160, kernel_size=(5, 5), bias=False)

        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm2d(96)
        self.batch4 = nn.BatchNorm2d(128)
        self.batch5 = nn.BatchNorm2d(160)
        self.batch6 = nn.BatchNorm1d(10)

        self.linear = nn.Linear(10240, 10, bias=False)

    def forward(self, x):
        x = (x - 0.5) * 2.0
        out = F.relu(self.batch1(self.conv1(x)))
        out = F.relu(self.batch2(self.conv2(out)))
        out = F.relu(self.batch3(self.conv3(out)))
        out = F.relu(self.batch4(self.conv4(out)))
        out = F.relu(self.batch5(self.conv5(out)))
        out = torch.flatten(out.permute(0, 2, 3, 1), 1)
        out = self.batch6(self.linear(out))

        return F.log_softmax(out, dim=1)

class cnn_7(nn.Module):
    def __init__(self):
        super(cnn_7, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=(7, 7), bias=False)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=(7, 7), bias=False)
        self.conv3 = nn.Conv2d(96, 144, kernel_size=(7, 7), bias=False)
        self.conv4 = nn.Conv2d(144, 192, kernel_size=(7, 7), bias=False)

        self.batch1 = nn.BatchNorm2d(48)
        self.batch2 = nn.BatchNorm2d(96)
        self.batch3 = nn.BatchNorm2d(144)
        self.batch4 = nn.BatchNorm2d(192)
        self.batch5 = nn.BatchNorm1d(10)

        self.linear = nn.Linear(3072, 10, bias=False)

    def forward(self, x):
        x = (x - 0.5) * 2.0
        out = F.relu(self.batch1(self.conv1(x)))
        out = F.relu(self.batch2(self.conv2(out)))
        out = F.relu(self.batch3(self.conv3(out)))
        out = F.relu(self.batch4(self.conv4(out)))
        out = torch.flatten(out.permute(0, 2, 3, 1), 1)
        out =  self.batch5(self.linear(out))

        return F.log_softmax(out, dim=1)

def load_models(device):
    cnn3 = cnn_3().to(device)
    cnn5 = cnn_5().to(device)
    cnn7 = cnn_7().to(device)

    return cnn3, cnn5, cnn7

def load_model_states(cnn3, cnn5, cnn7):
    cnn3.load_state_dict(torch.load('pre_trained_models/cnn_3'))
    cnn5.load_state_dict(torch.load('pre_trained_models/cnn_5'))
    cnn7.load_state_dict(torch.load('pre_trained_models/cnn_7'))

    return cnn3, cnn5, cnn7


if __name__ == '__main__':
    cnn_3, cnn_5, cnn_7 = load_models('cpu')
    cnn_3, cnn_5, cnn_7 = load_model_states(cnn_3, cnn_5, cnn_7)

    print('test successful')
