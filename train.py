from dataset import Sentence, NeuralNetwork
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import os
import numpy as np
import pandas as pd

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        pred = model(X)
        loss = loss_fn(pred, y.reshape(-1, 1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            y = y.reshape(-1,1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_tag = torch.round(torch.sigmoid(pred))
            correct += (pred_tag == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

bags = np.load('bags.npy', allow_pickle='TRUE').item()
training_data = Sentence('data/train_vector', 'data/train_label')
dev_data = Sentence('data/dev_vector', 'data/dev_label')
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
dev_dataloader = DataLoader(dev_data, batch_size=64, shuffle=True)

learning_rate = 1e-4
batch_size = 64
epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(len(bags))
model.to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(dev_dataloader, model, loss_fn)

torch.save(model, 'trained_model')
print("Train Done and Model Saved!")
