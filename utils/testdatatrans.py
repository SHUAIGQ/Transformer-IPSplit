import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import utils

# Example data
data = np.random.randn(100, 5, 3)  # 100 sequences, window size of 5, 3 features
data = torch.tensor(data, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(data)
data_loader = DataLoader(dataset, batch_size=10)

# Instantiate model
d_model = 3  # Number of features
N_layers = 2
attention = 8
window = 5
device = 'cpu'
dropout = 0.1
d_ff = 512

model =  utils.Transformer(d_model, N_layers, attention, window, device, dropout, d_ff)

# Train the model
model.train_model(data_loader, epochs=2)

# Detect anomalies
anomaly_scores = model.detect(data_loader)
print("Anomaly Scores:", anomaly_scores)
