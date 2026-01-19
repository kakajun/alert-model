import torch
import torch.nn as nn


class SolarPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SolarPredictor, self).__init__()
        # 架构灵感来自典型的表格数据 MLP
        # 输入 -> 隐藏层1 -> ReLU -> 隐藏层2 -> ReLU -> 输出
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)  # 回归输出
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x
