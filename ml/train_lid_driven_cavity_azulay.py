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
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()
        self.activation3 = nn.PReLU()
        self.activation4 = nn.PReLU()
        self.activation5 = nn.PReLU()
        self.activation6 = nn.PReLU()
        self.activation7 = nn.PReLU()
        self.activation8 = nn.PReLU()
        self.activation9 = nn.PReLU()
        self.activation10 = nn.PReLU()
        self.activation11 = nn.PReLU()

    def forward(self, rhs):
        # x, get inner without boundary
        x = rhs[:, :, 1:-1, 1:-1]
        x = F.normalize(x, dim=[2, 3])

        # Encoder
        x1 = self.activation1(self.downsample(self.conv1(x)))
        x2 = self.activation2(self.downsample(self.conv2(x1)))
        x3 = self.activation3(self.downsample(self.conv3(x2)))
        x4 = self.activation4(self.downsample(self.conv4(x3)))

        # Decoder
        x5 = self.activation5(self.upsample(self.conv5(x4)))
        x6 = torch.cat([x5, x3], dim=1)
        x6 = self.activation6(self.conv6(x6))

        x7 = self.activation7(self.upsample(self.conv7(x6)))
        x8 = torch.cat([x7, x2], dim=1)
        x8 = self.activation8(self.conv8(x8))

        x9 = self.activation9(self.upsample(self.conv9(x8)))
        x10 = torch.cat([x9, x1], dim=1)
        x10 = self.activation10(self.conv10(x10))

        x11 = self.upsample(self.conv11(x10))
        x11 = self.activation11(x11)

        # add zero boundaries
        x11 = F.pad(x11, (1, 1, 1, 1), "constant", 0)

        return x11

def custom_loss(pred_error, true_error, residual, grid_size_x, grid_size_y):
    # grid-like to vector-like
    true_error = true_error.view(-1, grid_size_x*grid_size_y)
    pred_error = pred_error.view(-1, grid_size_x*grid_size_y)
    
    # Compute cosine similarity
    cosine_similarity = F.cosine_similarity(pred_error, true_error, dim=1)  # Along the batch dimension
    
    # Compute the distance from 1.0
    alignment_loss = 1.0 - cosine_similarity
    
    # Take mean over the spatial dimensions and then mean across the batch
    alignment_loss = torch.mean(alignment_loss)
    
    # Additionally, you might want to incorporate your existing loss term
    loss_deep_learning = torch.sqrt(torch.mean((pred_error - true_error) ** 2))
    
    # Total loss
    total_loss = alignment_loss
    
    return total_loss

"""def custom_loss(pred_error, true_error, residual, grid_size_x, grid_size_y):
    dx2 = (1.0 / (grid_size_x-2)) ** 2
    dy2 = (1.0 / (grid_size_y-2)) ** 2

    # Reshape for batch operations
    pred_error = pred_error.view(-1, grid_size_x, grid_size_y)
    true_error = true_error.view(-1, grid_size_x, grid_size_y)
    residual = residual.view(-1, grid_size_x, grid_size_y)

    # Compute losses
    loss_deep_learning = F.mse_loss(pred_error, true_error, reduction='none')

    loss_simulation = torch.norm(
        (residual[:, 1:-1, 1:-1] - (
            (1/dx2) * (pred_error[:, 2:, 1:-1] - 2*pred_error[:, 1:-1, 1:-1] + pred_error[:, :-2, 1:-1]) +
            (1/dy2) * (pred_error[:, 1:-1, 2:] - 2*pred_error[:, 1:-1, 1:-1] + pred_error[:, 1:-1, :-2])
        )),
        dim=[1, 2]
    )

    total_loss = torch.mean(loss_simulation)*torch.mean(loss_deep_learning)

    return total_loss"""

"""def custom_loss(pred_error, true_error, residual, grid_size_x, grid_size_y):
    dx2 = (1.0 / grid_size_x) ** 2
    dy2 = (1.0 / grid_size_y) ** 2

    # Reshape for batch operations
    pred_error = pred_error.view(-1, grid_size_x, grid_size_y)
    true_error = true_error.view(-1, grid_size_x, grid_size_y)
    residual = residual.view(-1, grid_size_x, grid_size_y)

    Asearch_vector = torch.zeros_like(pred_error)
    # Calculate Asearch_vector for the entire batch
    Asearch_vector[:, 1:-1, 1:-1] = (
        (1/dx2) * (pred_error[:, 2:, 1:-1] - 2*pred_error[:, 1:-1, 1:-1] + pred_error[:, :-2, 1:-1]) +
        (1/dy2) * (pred_error[:, 1:-1, 2:] - 2*pred_error[:, 1:-1, 1:-1] + pred_error[:, 1:-1, :-2])
    )

    # Compute alpha for the entire batch
    alpha_top = torch.sum(pred_error.view(-1, grid_size_x*grid_size_y) * residual.view(-1, grid_size_x*grid_size_y))
    alpha_bottom = torch.sum(pred_error.view(-1, grid_size_x*grid_size_y) * Asearch_vector.view(-1, grid_size_x*grid_size_y))
    alpha = alpha_top / alpha_bottom

    # Reduce grid-like to vector-like
    Asearch_vector = Asearch_vector.view(-1, grid_size_x*grid_size_y)
    residual = residual.view(-1, grid_size_x*grid_size_y)
    alpha = alpha.view(-1, 1)

    # copy alpha to the same shape as Asearch_vector
    alpha = alpha.repeat(1, grid_size_x*grid_size_y)

    # Compute losses
    loss_simulation = torch.norm(
        (residual - alpha*Asearch_vector)
    ) / pred_error.shape[0]

    total_loss = torch.mean(loss_simulation)

    return total_loss"""

if __name__ == "__main__":
    # random state 
    torch.manual_seed(42)
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

    # Define batch size
    batch_size = 32

    # Train the model
    num_epochs = 100
    total_batches = len(train_residual_data) // batch_size

    for epoch in tqdm(range(num_epochs)):
        # Shuffle the training data
        shuffled_indices = torch.randperm(len(train_residual_data))
        train_residual_data_shuffled = train_residual_data[shuffled_indices]
        train_error_data_shuffled = train_error_data[shuffled_indices]
        
        for i in range(total_batches):
            # Get the current batch
            batch_residual_data = train_residual_data_shuffled[i * batch_size: (i + 1) * batch_size]
            batch_error_data = train_error_data_shuffled[i * batch_size: (i + 1) * batch_size]

            # Forward pass
            predicted_error_vector = model(batch_residual_data)

            # Calculate the loss
            loss = custom_loss(predicted_error_vector, batch_error_data, batch_residual_data, grid_size_x, grid_size_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train', loss.item(), epoch * total_batches + i)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch * total_batches + i)

            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{total_batches}, Train Loss: {loss.item()}")

        # Update the learning rate scheduler
        scheduler.step(loss)

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
    input_tensor = train_residual_data[0].unsqueeze(0)
    output = traced(input_tensor)
    # save output to e.dat
    output = output.squeeze(0).squeeze(0).cpu().detach().numpy()
    np.savetxt("e.dat", output)
    print(output)
    print(output.shape)