import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def load_ecg5000():

    train_data = pd.read_csv("ECG5000_TRAIN.txt", header=None, delim_whitespace=True)
    test_data = pd.read_csv("ECG5000_TEST.txt", header=None, delim_whitespace=True)
    data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

    y = data.iloc[:, 0].values - 1
    X = data.iloc[:, 1:].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X, y

class ECG5000Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(batch_size=32):
    X, y = load_ecg5000()
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_dataset = ECG5000Dataset(X_train, y_train)
    test_dataset = ECG5000Dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader