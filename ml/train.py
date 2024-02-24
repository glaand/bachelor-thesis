# try implement paper: https://arxiv.org/pdf/2203.11025.pdf
# loss function: Equation (3.7)

import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # Output
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x1 = F.selu(self.enc_conv1(x))
        x2 = F.selu(self.enc_conv2(x1))
        x3 = F.selu(self.enc_conv3(x2))

        # Bottleneck
        x_bottleneck = F.selu(self.bottleneck_conv(x3))

        # Output
        dx1 = F.selu(self.dec_conv1(x_bottleneck))
        x_out = F.selu(self.final_conv(dx1))


        return x_out
    
def custom_loss(output, target):

    loss = torch.mean(torch.abs(output - target) / (torch.abs(target) + 1))
    return loss

def evaluate_regression(model, test_data, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data[0])
        loss = criterion(outputs, test_data[1])
        true_data = test_data[1].to('cpu').numpy().flatten()
        predicted_data = outputs.to('cpu').numpy().flatten()
        mae = mean_absolute_error(true_data, predicted_data)
        rmse = np.sqrt(mean_squared_error(true_data, predicted_data))

    return loss.item(), mae, rmse

# Load data
pressure_data = torch.tensor([])
RHS_data = torch.tensor([])

with h5py.File("pressure.h5", "r") as f:
    keys = list(f.keys())
    for key in keys:
        pressure_data = torch.cat((pressure_data, torch.tensor(f[key]).unsqueeze(0)), 0)

with h5py.File("RHS.h5", "r") as f:
    keys = list(f.keys())
    for key in keys:
        RHS_data = torch.cat((RHS_data, torch.tensor(f[key]).unsqueeze(0)), 0)

# Prepare data
pressure_data = pressure_data.view(pressure_data.shape[0], 1, 34, 34).float()
RHS_data = RHS_data.view(RHS_data.shape[0], 1, 34, 34).float()

# Split data into train and test sets
total_samples = pressure_data.shape[0]
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size

train_data, test_data = random_split(list(zip(pressure_data, RHS_data)), [train_size, test_size])

train_pressure_data, train_RHS_data = zip(*train_data)
test_pressure_data, test_RHS_data = zip(*test_data)

train_pressure_data = torch.stack(train_pressure_data).to("cuda")
train_RHS_data = torch.stack(train_RHS_data).to("cuda")
test_pressure_data = torch.stack(test_pressure_data).to("cuda")
test_RHS_data = torch.stack(test_RHS_data).to("cuda")

# Create the model
model = SimpleCNN()
model.to("cuda")

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.000001)

# Set the model in training mode
model.train()

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    output = model(train_RHS_data)

    # Calculate the loss
    loss = custom_loss(output, train_pressure_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}")

# Evaluate the model on the test set
test_loss, test_mae, test_rmse = evaluate_regression(model, (test_RHS_data, test_pressure_data), custom_loss)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, Test RMSE: {test_rmse}")

# Convert to Torchscript via Annotation
model.eval()
traced = torch.jit.trace(model, test_RHS_data[0])
traced.save("model.pt")

# generate random tensor to test the model
input_tensor = torch.rand(1, 34, 34).to("cuda")
output = traced(input_tensor)
print(output)