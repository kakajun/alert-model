import torch
import torch.nn as nn

class SolarPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SolarPredictor, self).__init__()
        # Architecture inspired by typical tabular data MLPs
        # Input -> Hidden1 -> ReLU -> Hidden2 -> ReLU -> Output
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1) # Regression output
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
