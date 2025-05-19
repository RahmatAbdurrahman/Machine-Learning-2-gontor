from sklearn.datasets import load_iris  # Dataset Iris
from sklearn.model_selection import train_test_split  # Membagi data train/test
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Normalisasi dan encoding
from sklearn.linear_model import LogisticRegression  # Model baseline
from sklearn.metrics import accuracy_score  # Evaluasi akurasi

import torch  # PyTorch
import torch.nn as nn  # Neural network module
import torch.optim as optim  # Optimizer
from torch.utils.data import TensorDataset, DataLoader  # Data loader PyTorch

# Mengubah data ke tensor PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Membuat dataset dan dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Mini-batch training

# Inisialisasi model, loss, dan optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()  # Loss untuk klasifikasi multi-kelas
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Optimizer Adam

losses = []       # Simpan loss tiap epoch
accuracies = []   # Simpan akurasi tiap epoch

# Loop training selama 100 epoch
for epoch in range(100):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in train_loader:  # Mini-batch training
        optimizer.zero_grad()         # Reset gradien
        outputs = model(xb)           # Feedforward
        loss = criterion(outputs, yb) # Hitung loss
        loss.backward()              # Backpropagation
        optimizer.step()             # Update bobot

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == yb).sum().item()
        total += yb.size(0)

    acc = correct / total
    losses.append(total_loss)
    accuracies.append(acc)

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100] - Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")
