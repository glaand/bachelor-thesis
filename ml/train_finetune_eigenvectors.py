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

from train_lid_driven_cavity_eigenvectors import Kaneda, custom_loss

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

    # Load eigenvectors-based data
    # eigenvector_b = load_data("eigenvectors/", "b", 1)
    # eigenvector_x = load_data("eigenvectors/", "x", 1)

    # Load simulation data
    simulation_b = load_data("ML_data/", "res", 100)
    simulation_x = load_data("ML_data/", "e", 100)

    # Concat eigenvectors-based data with simulation data
    residual_data = simulation_b
    error_data = simulation_x

    augment_data = False

    if augment_data:
        # Data augmentation - flip the data along the x-axis
        residual_data = torch.cat([residual_data, residual_data.flip(1)])
        error_data = torch.cat([error_data, error_data.flip(1)])
        # Data augmentation - flip the data along the y-axis
        #residual_data = torch.cat([residual_data, residual_data.flip(2)])
        #error_data = torch.cat([error_data, error_data.flip(2)])

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

    train_residual_data = torch.stack(train_residual_data)
    train_error_data = torch.stack(train_error_data)
    test_residual_data = torch.stack(test_residual_data)
    test_error_data = torch.stack(test_error_data)


    model = Kaneda()
    # load jit trace model
    model = torch.jit.load("model_eigenvectors.pt")
    model.to("cuda")

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    model.conv12.weight.requires_grad = True
    model.conv12.bias.requires_grad = True

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5, amsgrad=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # Set the model in training mode
    model.train()

    writer = SummaryWriter('logs')

    # Define batch size
    batch_size = 8

    # Train the model
    num_epochs = 20
    total_batches = len(train_residual_data) // batch_size

    for epoch in tqdm(range(num_epochs)):
        # Shuffle the training data
        shuffled_indices = torch.randperm(len(train_residual_data))
        train_residual_data_shuffled = train_residual_data[shuffled_indices]
        train_error_data_shuffled = train_error_data[shuffled_indices]
        
        for i in range(total_batches):
            # Get the current batch
            batch_residual_data = train_residual_data_shuffled[i * batch_size: (i + 1) * batch_size].to("cuda")
            batch_error_data = train_error_data_shuffled[i * batch_size: (i + 1) * batch_size].to("cuda")

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
    traced.save("finetuned_eigenvectors.pt")

    # generate random tensor to test the model
    input_tensor = train_residual_data[0].unsqueeze(0)
    output = traced(input_tensor)
    # save output to e.dat
    output = output.squeeze(0).squeeze(0).cpu().detach().numpy()
    np.savetxt("e.dat", output)
    print(output)
    print(output.shape)