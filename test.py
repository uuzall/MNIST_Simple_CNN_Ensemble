import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random

def models_accuracy(model_name, model, test_dataloader, device):
    correct, output, true_out = 0, list(), list()
    with torch.no_grad():
        for x, y in tqdm(test_dataloader):
            x, y = x.to(device), y.view(-1, 1).to(device)
            out = model(x)
            cat = out.argmax(dim=1, keepdim=True)
            correct += (cat == y).sum().item()
            output.append(cat.view(-1).tolist())
            true_out.append(y.view(-1).tolist())
    print(model_name, 'accuracy: ', correct / (len(test_dataloader)*100) * 100)

    return np.array(output), np.array(true_out)

def three_some(out3, out5, out7, true_out):
    out0, out1, out2 = out3.reshape(-1), out5.reshape(-1), out7.reshape(-1)
    to = true_out.reshape(-1)

    final_out = list()
    for index in range(len(out0)):
        if out0[index] == out1[index]:
            final_out.append(out0[index])
        elif out0[index] == out2[index]:
            final_out.append(out0[index])
        elif out1[index] == out2[index]:
            final_out.append(out1[index])
        else:
            final_out.append(random.randint(0, 10))

    print('Total Accuracy: ', (np.array(final_out) == to).mean() * 100)

def test_loop(cnn_3, cnn_5, cnn_7, test_dataloader, device):
    out3, to = models_accuracy('cnn_3', cnn_3, test_dataloader, device)
    out5, to = models_accuracy('cnn_5', cnn_5, test_dataloader, device)
    out7, to = models_accuracy('cnn_7', cnn_7, test_dataloader, device)

    three_some(out3, out5, out7, to)
