import torch  # PyTorch
import torch.nn as nn  # Neural network module
import torch.optim as optim  # Optimizer
from torch.utils.data import TensorDataset, DataLoader  # Data loader PyTorch

# Model MLP dengan 1 hidden layer
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 16)       # Layer 1: input 4 fitur → 16 neuron
        self.relu = nn.ReLU()             # Fungsi aktivasi ReLU
        self.fc2 = nn.Linear(16, 3)       # Output: 3 neuron (kelas)
        self.softmax = nn.Softmax(dim=1)  # Softmax untuk klasifikasi multi-kelas

    def forward(self, x):
        x = self.relu(self.fc1(x))        # Forward ke hidden layer + aktivasi
        x = self.softmax(self.fc2(x))     # Output layer + softmax
        return x