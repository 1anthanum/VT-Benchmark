import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, lstm_layers=1, dropout=0.3):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class FCN_Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FCN_Model, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=8, padding='same')
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding='same')
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding='same')
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.global_pooling(x).squeeze(-1)
        return self.fc(x)

class LSTM_FCN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, lstm_layers=1, dropout=0.3):
        super(LSTM_FCN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True, dropout=dropout)
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=8, padding='same')
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding='same')
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding='same')
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim + 128, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        x_fcn = x.permute(0, 2, 1)
        x_fcn = torch.relu(self.conv1(x_fcn))
        x_fcn = torch.relu(self.conv2(x_fcn))
        x_fcn = torch.relu(self.conv3(x_fcn))
        x_fcn = self.global_pooling(x_fcn).squeeze(-1)

        combined = torch.cat([lstm_out, x_fcn], dim=1)
        return self.fc(combined)
    
class ALSTM_FCN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, lstm_layers=1, dropout=0.3):
        super(ALSTM_FCN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=8, padding='same')
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding='same')
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding='same')
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim + 128, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        x_fcn = x.permute(0, 2, 1)
        x_fcn = torch.relu(self.conv1(x_fcn))
        x_fcn = torch.relu(self.conv2(x_fcn))
        x_fcn = torch.relu(self.conv3(x_fcn))
        x_fcn = self.global_pooling(x_fcn).squeeze(-1)

        combined = torch.cat([context_vector, x_fcn], dim=1)
        return self.fc(combined)