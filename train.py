import os
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.optim as optim

from data_loader import Dataset

from model import cnn


# Feature, Label Shape을 확인합니다.
#print(x_data.shape, y_data.shape)


#device = torch.device("cuda:0")

batch_size = 128
data_size = 20000

dl_params = {'batch_size': batch_size, 'shuffle': True}

training_set = Dataset(rng = [0, data_size])
training_gen = DataLoader(training_set, **dl_params)

test_set = Dataset(rng = [data_size, data_size + 10000])
test_gen = DataLoader(test_set, **dl_params)


device = torch.device("cuda:0")
model = cnn()
model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr = 1e-2)

criterion = torch.nn.KLDivLoss('batchmean')
#criterion = nn.BCEWithLogitsLoss()


for epoch in range(100):

    model.train()
    loss_train = 0
    idx = 0
    for x_batch, y_batch in training_gen:
        optimizer.zero_grad()
        #x_batch = x_batch.to(device)
        #y_batch = y_batch.to(device)
        y_predicted = model(x_batch)

        #print(x_batch)
        #print(y_batch, y_predicted)
        loss = criterion(y_predicted, y_batch.float().to(device))
        loss_train += loss.item()
        loss.backward()
        optimizer.step()
        if idx%(data_size//(10*batch_size)) == 0:
            print(f'Done : {idx/(data_size//batch_size):.2f} Batch loss : {loss.item():.5f}')
        idx += 1
        
    print(loss.item())

    model.eval()
    loss_test = 0
    for x_test, y_test in test_gen:
        y_predicted = model(x_test)
        loss = nn.KLDivLoss(reduction= "sum")(y_predicted, y_test.float().to(device))
        loss_test += loss.item()
    
    print(f'Training Loss : {loss_train:.4f} , val Loss : {loss_test:.4f}, val_train_ratio : {loss_test/loss_train:.4f}')
        










