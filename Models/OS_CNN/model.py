import torch
import torch.nn as nn

class OSCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(OSCNN, self).__init__()
        # 多尺度卷积层
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=2, padding='same')
        self.conv2 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=4, padding='same')
        self.conv3 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=8, padding='same')
        self.conv4 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=16, padding='same')
        self.conv5 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=32, padding='same')
        
        self.relu = nn.ReLU()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 * 5, num_classes)

    def forward(self, x):
        # 输入格式: (batch_size, 1, sequence_length)
        # 确保数据格式为 (batch, 1, sequence_length)
        assert x.shape[1] == 1, f"输入通道数错误, 期望 1, 但得到 {x.shape[1]}"

        # 多尺度卷积
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x4 = self.relu(self.conv4(x))
        x5 = self.relu(self.conv5(x))

        # 全局平均池化
        x1 = self.global_pooling(x1).squeeze(-1)
        x2 = self.global_pooling(x2).squeeze(-1)
        x3 = self.global_pooling(x3).squeeze(-1)
        x4 = self.global_pooling(x4).squeeze(-1)
        x5 = self.global_pooling(x5).squeeze(-1)

        # 拼接所有特征
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        return self.fc(x)