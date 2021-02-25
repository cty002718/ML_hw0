from dataset import Sentence
from preprocessing import preprocessing 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import os
import numpy as np
import pandas as pd

def predict(path):
    preprocessing(path, 'data/test_vector', 'data/test_label')
    test_data = Sentence('data/test_vector', 'data/test_label')
    test_dataloader = DataLoader(test_data, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('trained_model')
    model.to(device)
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X, y in test_dataloader:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            y = y.reshape(-1,1)
            pred = model(X)
            pred_tag = torch.round(torch.sigmoid(pred))
            y_pred_list.append(pred_tag.cpu().numpy())

    df = pd.read_csv('data/test.csv')
    y_id = df.iloc[:,0]
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    dictionary = {"Id": y_id,
        "Category": y_pred_list
       }

    df = pd.DataFrame(dictionary)
    df = df.astype({"Category": int})
    df.to_csv('data/result.csv', index=0)

predict('data/test.csv')
