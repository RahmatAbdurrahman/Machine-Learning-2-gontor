# Klasifikasi Bunga Iris menggunakan Multi-Layer Perceptron (MLP)

Proyek ini merupakan implementasi sederhana neural network (MLP) menggunakan PyTorch untuk melakukan klasifikasi bunga Iris. 
Dataset Iris merupakan dataset klasik yang terdiri dari tiga kelas bunga: *setosa*, *versicolor*, dan *virginica*, 
berdasarkan empat fitur morfologi bunga.

## Dataset

Dataset yang digunakan adalah **Iris dataset** dari `sklearn.datasets.load_iris()`, yang berisi 150 sampel dengan 4 fitur:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Target klasifikasinya adalah spesies bunga (3 kelas):
- *setosa*
- *versicolor*
- *virginica*

## Arsitektur Model

Model MLP yang digunakan terdiri dari:
- Input Layer: 4 neuron (jumlah fitur)
- Hidden Layer: 16 neuron + ReLU activation
- Output Layer: 3 neuron + Softmax (untuk klasifikasi multi-kelas)

# Building a Neural Network for Iris Classification with PyTorch

---

## 1. Import Library yang Diperlukan

Untuk memulai, impor modul PyTorch dan dataset Iris yang diperlukan:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## 2. Menyiapkan Dataset

