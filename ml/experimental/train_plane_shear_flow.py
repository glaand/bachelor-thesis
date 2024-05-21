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
from tqdm import tqdm

# generate laplacian matrix
def generate_laplacian_matrix(n):
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = 4
        if i > 0:
            L[i, i-1] = -1
            L[i-1, i] = -1
    # convert to tensor
    L = torch.tensor(L).float().to("cuda")
    return L

class Kaneda(nn.Module):
    def __init__(self, N, dim, fil_num):
        super(Kaneda, self).__init__()
        self.N = N
        self.conv1 = nn.Conv2d(1, fil_num, kernel_size=(2, 2), padding='same')
        self.conv2 = nn.Conv2d(fil_num, fil_num, kernel_size=(2, 2), padding='same')
        self.conv3 = nn.Conv2d(fil_num, fil_num, kernel_size=(2, 2), padding='same')
        self.conv4 = nn.Conv2d(fil_num, fil_num, kernel_size=(2, 2), padding='same')
        self.conv5 = nn.Conv2d(fil_num, fil_num, kernel_size=(2, 2), padding='same')
        self.conv6 = nn.Conv2d(fil_num, fil_num, kernel_size=(2, 2), padding='same')
        self.conv7 = nn.Conv2d(fil_num, fil_num, kernel_size=(2, 2), padding='same')
        self.conv8 = nn.Conv2d(fil_num, fil_num, kernel_size=(2, 2), padding='same')
        self.conv9 = nn.Conv2d(fil_num, fil_num, kernel_size=(2, 2), padding='same')
        self.conv10 = nn.Conv2d(fil_num, fil_num, kernel_size=(2, 2), padding='same')
        self.conv11 = nn.Conv2d(fil_num, fil_num, kernel_size=(2, 2), padding='same')
        self.conv12 = nn.Conv2d(fil_num, fil_num, kernel_size=(2, 2), padding='same')

        self.reduce_channels = nn.Conv2d(fil_num, 1, kernel_size=1)  # Reduce channels to 1

        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        la = self.relu(self.conv2(x))
        lb = self.relu(self.conv3(la))
        la = self.relu(self.conv4(lb)) + la
        lb = self.relu(self.conv5(la))
        
        if self.N < 130:
            apa = self.avgpool(lb)
            apb = self.relu(self.conv6(apa))
            apa = self.relu(self.conv7(apb)) + apa
            apb = self.relu(self.conv8(apa))
            apa = self.relu(self.conv9(apb)) + apa
            apb = self.relu(self.conv10(apa))
            apa = self.relu(self.conv11(apb)) + apa
        else:
            apa = self.avgpool(lb)
            apb = self.relu(self.conv6(apa))
            apa = self.relu(self.conv7(apb)) + apa
            apb = self.relu(self.conv8(apa))
            apa = self.relu(self.conv9(apb)) + apa
            apb = self.relu(self.conv10(apa))
            apa = self.relu(self.conv11(apb)) + apa
            apb = self.relu(self.conv12(apa))
            apa = self.relu(self.conv11(apb)) + apa
            apb = self.relu(self.conv12(apa))
            apa = self.relu(self.conv11(apb)) + apa

        upa = F.interpolate(apa, scale_factor=2, mode='nearest') + lb
        upb = self.relu(self.conv5(upa))
        upa = self.relu(self.conv4(upb)) + upa
        upb = self.relu(self.conv3(upa))
        upa = self.relu(self.conv2(upb)) + upa
        upb = self.relu(self.conv1(self.reduce_channels(upa)))
        upa = self.relu(self.conv2(upb)) + upa

        out = self.reduce_channels(upa)
        return out

def custom_loss(predicted_error_vector, residual_data, A):
    # Initialize loss accumulator
    total_loss = 0.0

    # Reshape tensors if needed
    residual_data = residual_data.view(residual_data.shape[0], -1)
    predicted_error_vector = predicted_error_vector.view(predicted_error_vector.shape[0], -1)

    # Loop through each vector in residual_data
    for i in range(residual_data.shape[0]):
        # Calculate alpha for each vector
        alpha_top = torch.matmul(residual_data[i].T, predicted_error_vector[i])
        alpha_bottom = torch.matmul(torch.matmul(predicted_error_vector[i].T, A), predicted_error_vector[i])
        alpha = alpha_top / alpha_bottom

        # Calculate loss for each vector
        loss = torch.norm(residual_data[i] - alpha * torch.matmul(A, predicted_error_vector[i])) ** 2

        # Accumulate loss
        total_loss += loss

    return total_loss / residual_data.shape[0]

def evaluate_regression(model, test_data, criterion):
    model.eval()
    with torch.no_grad():
        predicted_error_vector = model(test_data[0])
        loss = criterion(predicted_error_vector, test_data[0], A)

    return loss.item()

# Load data
residual_data = torch.tensor([])

# list files for given path
import os
residual_files = {}
for file in os.listdir("../experiments/2d/mgpcg/ML_data/"):
    if file.endswith(".dat"):
        if "res_" in file:
            _, number = file.split("_")
            number, _ = number.split(".")
            residual_files[int(number)] = os.path.join("../experiments/2d/mgpcg/ML_data/", file)

# sort files
residual_files = dict(sorted(residual_files.items()))

# load data
print("Loading residual data...")
for key in list(residual_files.keys()):
    filepath = residual_files[key]
    loaded_residual_data = np.loadtxt(filepath)
    # norm 2
    normed_residual_data = loaded_residual_data / np.linalg.norm(loaded_residual_data)
    residual_data = torch.cat((residual_data, torch.tensor(normed_residual_data).unsqueeze(0)), 0)

# Prepare data
grid_size_x = 18
grid_size_y = 66
# get minimum grid size
vector_size = np.min([grid_size_x, grid_size_y])
A = generate_laplacian_matrix(grid_size_x*grid_size_y)
print(residual_data.shape)
residual_data = residual_data.view(residual_data.shape[0], 1, grid_size_x, grid_size_y).float()
residual_data = residual_data.to("cuda")
model = Kaneda(vector_size, 1, 16)
model.to("cuda")

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# Set the model in training mode
model.train()

# Train the model
num_epochs = 1
for epoch in tqdm(range(num_epochs)):
    # Forward pass
    predicted_error_vector = model(residual_data)

    # Calculate the loss
    loss = custom_loss(predicted_error_vector, residual_data, A)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update the learning rate scheduler
    scheduler.step(loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}")

# Evaluate the model on the test set
test_loss = evaluate_regression(model, (residual_data, A), custom_loss)
print(f"Test Loss: {test_loss}")

# Convert to Torchscript via Annotation
model.eval()
traced = torch.jit.trace(model, residual_data)
traced.save("model.pt")

# generate random tensor to test the model
input_tensor = residual_data[0].unsqueeze(0)
output = traced(input_tensor)
# save output to e.dat
output = output.cpu().detach().numpy()
np.savetxt("e.dat", output.reshape(grid_size_x, grid_size_y))
print(output)
print(output.shape)