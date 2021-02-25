import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

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

def create_bags(threshold):
    bags = {}
    df = pd.read_csv('data/train.csv')
    for text in df.iloc[:, 1]:
        tokens = text.split(' ')
        for token in tokens:
            if token not in bags.keys():
                bags[token] = 1
            else:
                bags[token] += 1
    #print('original bags length:', len(bags))
    for key in list(bags):
        if bags[key] <= threshold:
            del bags[key]

    #print('cut bags length:', len(bags))
    np.save('bags.npy', bags)


def preprocessing(root, vector_file, label_file):
    bags = np.load('bags.npy',allow_pickle='TRUE').item()
    df = pd.read_csv(root)
    vectors = []
    pbar = tqdm(total=len(df))
    for text in df.iloc[:, 1]:
        vector = []
        for key in bags:
            vector.append(text.count(key))
        vectors.append(vector)
        pbar.update(1)
    pbar.close()
    labels = df.iloc[:, 2]
    torch.save(torch.tensor(vectors), vector_file)
    torch.save(torch.tensor(labels), label_file)

