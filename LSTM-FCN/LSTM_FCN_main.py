import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from data_utils import get_dataloaders
from models import LSTM_Model, FCN_Model, LSTM_FCN
from train import train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_dataloaders(batch_size=32)

print("\nTraining LSTM...")
model = LSTM_Model(1, 5)
train_model(model, train_loader, test_loader)
evaluate_model(model, test_loader)

print("\nTraining FCN...")
model = FCN_Model(1, 5)
train_model(model, train_loader, test_loader)
evaluate_model(model, test_loader)

print("\nTraining LSTM-FCN...")
model = LSTM_FCN(1, 5)
train_model(model, train_loader, test_loader)
evaluate_model(model, test_loader)