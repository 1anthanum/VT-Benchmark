import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_ecg5000():

    train_data = pd.read_csv("ECG5000_TRAIN.txt", header=None, delim_whitespace=True)
    test_data = pd.read_csv("ECG5000_TEST.txt", header=None, delim_whitespace=True)
    data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

    y = data.iloc[:, 0].values - 1
    X = data.iloc[:, 1:].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

class ECG5000Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(batch_size=32):
    X_train, X_test, y_train, y_test = load_ecg5000()

    train_dataset = ECG5000Dataset(X_train, y_train)
    test_dataset = ECG5000Dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader