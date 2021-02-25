import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import os
import numpy as np
import pandas as pd

class Sentence(Dataset):
    def __init__(self, vectors_path, labels_path):
        self.vectors = torch.load(vectors_path).float()
        self.labels = torch.load(labels_path).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vector = self.vectors[idx]
        label = self.labels[idx]
        return vector, label


class NeuralNetwork(nn.Module):
    def __init__(self, input_len):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_len, 512)
        self.layer_2 = nn.Linear(512, 512)
        self.layer_out = nn.Linear(512, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(512)
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x
