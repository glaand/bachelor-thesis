# try implement paper: https://arxiv.org/pdf/2203.11025.pdf
# loss function: Equation (3.7)

import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import glob
import os

class Azulay(nn.Module):
    def __init__(self):
        super(Azulay, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Decoder
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.SiLU()

    def forward(self, x):
        # x, get inner without boundary
        x = x[:, :, 1:-1, 1:-1]
        x = x / torch.max(x)

        # Encoder
        x1 = self.activation(self.downsample(self.conv1(x)))
        x2 = self.activation(self.downsample(self.conv2(x1)))
        x3 = self.activation(self.downsample(self.conv3(x2)))
        x4 = self.activation(self.downsample(self.conv4(x3)))

        # Decoder
        x5 = self.activation(self.upsample(self.conv5(x4)))
        x6 = torch.cat([x5, x3], dim=1)
        x6 = self.activation(self.conv6(x6))

        x7 = self.activation(self.upsample(self.conv7(x6)))
        x8 = torch.cat([x7, x2], dim=1)
        x8 = self.activation(self.conv8(x8))

        x9 = self.activation(self.upsample(self.conv9(x8)))
        x10 = torch.cat([x9, x1], dim=1)
        x10 = self.activation(self.conv10(x10))

        x11 = self.upsample(self.conv11(x10))

        # add zero boundaries
        x11 = F.pad(x11, (1, 1, 1, 1), "constant", 0)

        return x11

""" def custom_loss(pred_error, true_error, residual, grid_size_x, grid_size_y):
    # Compute losses
    # normalise true_error
    true_error = true_error / torch.max(true_error)
    loss_deep_learning = torch.sqrt(torch.mean((pred_error - true_error) ** 2, dim=[2, 3]))
    total_loss = torch.mean(loss_deep_learning)
    return total_loss """

def custom_loss(pred_error, true_error, residual, grid_size_x, grid_size_y):
    # Compute residual - A*pred_error for each sample
    residual_pred_error = torch.zeros_like(residual)
    for i in range(residual.shape[0]):
        residual_pred_error[i, 0] = torch.matmul(A, pred_error[i].view(-1)).view(grid_size_x, grid_size_y)

    # Compute losses
    loss_deep_learning = torch.sqrt(torch.mean((residual_pred_error - residual) ** 2, dim=[2, 3]))
    total_loss = torch.mean(loss_deep_learning)
    return total_loss

if __name__ == "__main__":

    # Define data loading function
    def load_data(folder_path, prefix, skip=1):
        print(f"Skipping every {skip} files")
        data = []
        files = glob.glob(os.path.join(folder_path, f"{prefix}_*.dat"))
        files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        for i, file in enumerate(tqdm(files)):
            if i % skip == 0:
                loaded_data = np.loadtxt(file)
                data.append(torch.tensor(loaded_data))
        return torch.stack(data)

    # Load data
    residual_data = load_data("ML_data/", "res", 1)
    error_data = load_data("ML_data/", "e", 1)

    print("Residual data shape:", residual_data.shape)
    print("Error data shape:", error_data.shape)

    # Prepare data
    grid_size_x = 34
    grid_size_y = 34
    dx = 1 / (grid_size_x)
    dy = 1 / (grid_size_y)
    dx2 = dx ** 2
    dy2 = dy ** 2
    vector_size = np.min([grid_size_x, grid_size_y])
    residual_data = residual_data.view(residual_data.shape[0], 1, grid_size_x, grid_size_y).float()
    residual_data = residual_data.to("cuda")
    error_data = error_data.view(error_data.shape[0], 1, grid_size_x, grid_size_y).float()
    error_data = error_data.to("cuda")

    # Generate Matrix A (Laplacian matrix) for the Poisson equation
    A = np.zeros((grid_size_x * grid_size_y, grid_size_x * grid_size_y))
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            k = i * grid_size_y + j
            A[k, k] = -2 * (1 / dx2 + 1 / dy2)
            if i > 0:
                A[k, k - grid_size_y] = 1 / dx2
            if i < grid_size_x - 1:
                A[k, k + grid_size_y] = 1 / dx2
            if j > 0:
                A[k, k - 1] = 1 / dy2
            if j < grid_size_y - 1:
                A[k, k + 1] = 1 / dy2

    A = torch.tensor(A).float().to("cuda")

    # Split data into train and test sets
    total_samples = residual_data.shape[0]
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size

    train_data, test_data = random_split(
        list(zip(residual_data, error_data)), 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_residual_data, train_error_data = zip(*train_data)
    test_residual_data, test_error_data = zip(*test_data)

    train_residual_data = torch.stack(train_residual_data).to("cuda")
    train_error_data = torch.stack(train_error_data).to("cuda")
    test_residual_data = torch.stack(test_residual_data).to("cuda")
    test_error_data = torch.stack(test_error_data).to("cuda")


    model = Azulay()
    model.to("cuda")

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5, amsgrad=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # Set the model in training mode
    model.train()

    writer = SummaryWriter('logs')

    # Train the model
    num_epochs = 1000
    for epoch in tqdm(range(num_epochs)):
        # Forward pass
        predicted_error_vector = model(train_residual_data)

        # Calculate the loss
        loss = custom_loss(predicted_error_vector, train_error_data, train_residual_data, grid_size_x, grid_size_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Update the learning rate scheduler
        scheduler.step(loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}")

    writer.close()

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        predicted_error_vector = model(test_residual_data)
        test_loss = custom_loss(predicted_error_vector, test_error_data, test_residual_data, grid_size_x, grid_size_y)
        print(f"Test Loss: {test_loss}")

    # Convert to Torchscript via Annotation
    model.eval()
    traced = torch.jit.trace(model, test_residual_data[0].unsqueeze(0))
    traced.save("model_azulay.pt")

    # generate random tensor to test the model
    input_tensor = test_residual_data[0].unsqueeze(0)
    output = traced(input_tensor)
    # save output to e.dat
    output = output.squeeze(0).squeeze(0).cpu().detach().numpy()
    np.savetxt("e.dat", output)
    print(output)
    print(output.shape)