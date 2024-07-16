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
import argparse
from torch.utils.data import DataLoader, Subset

from sklearn.decomposition import PCA

from model import Model
from model_fourier import Model as FourierModel
from model_nobias import Model as NoBiasModel
from model_azulay import Model as AzulayModel
from gauss_fourier import GaussianFourierFeatureTransform
from loss import loss_cosine_similarity, loss_mse, loss_rmse, loss_huber

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

# Generate inverse
grid_size_x = 34
grid_size_y = 34
dx = 1 / (grid_size_x)
dy = 1 / (grid_size_y)
dx2 = dx ** 2
dy2 = dy ** 2

# generate Matrix A which is the discretized Laplacian in 2D
A = np.zeros((grid_size_x * grid_size_y, grid_size_x * grid_size_y))
for i in range(grid_size_x):
    for j in range(grid_size_y):
        row = i * grid_size_y + j
        A[row, row] = -2 * (1 / dx2 + 1 / dy2)
        if i > 0:
            A[row, row - grid_size_y] = 1 / dx2
        if i < grid_size_x - 1:
            A[row, row + grid_size_y] = 1 / dx2
        if j > 0:
            A[row, row - 1] = 1 / dy2
        if j < grid_size_y - 1:
            A[row, row + 1] = 1 / dy2

A_inverse = np.linalg.inv(A)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--data_type', type=str, choices=['simulation', 'eigenvectors', 'eigenvectors_low', 'eigenvectors_high', 'simulation_low', 'pressure'], required=True, help='Type of data to use: simulation or eigenvectors')
    parser.add_argument('--loss_function', type=str, choices=['cosine_similarity', 'mse', 'rmse', 'huber'], default='cosine_similarity', help='Loss function to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--skip_files', type=int, default=1, help='Number of files to skip while loading data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--model', type=str, default='standard', help='Model to use for training', choices=['standard', 'fourier', 'nobias','azulay'])
    parser.add_argument('--act', type=str, default='relu', help='Activation function to use', choices=[
        'elu', 
        'gelu',
        'leaky_relu',
        'prelu',
        'relu',
        'selu',
        'sigmoid',
        'swish',
        'softmax',
        'softplus',
        'tanh',
        'mish',
        'linear'
    ])
    
    args = parser.parse_args()

    # random state 
    torch.manual_seed(42)

    if args.data_type == 'eigenvectors':
        residual_data = load_data("data/eigenvectors_data/", "b", args.skip_files)
        error_data = load_data("data/eigenvectors_data/", "x", args.skip_files)
    elif args.data_type == 'eigenvectors_low':
        residual_data = load_data("data/eigenvectors_data_low/", "b", args.skip_files)
        error_data = load_data("data/eigenvectors_data_low/", "x", args.skip_files)
    elif args.data_type == 'eigenvectors_high':
        residual_data = load_data("data/eigenvectors_data_high/", "b", args.skip_files)
        error_data = load_data("data/eigenvectors_data_high/", "x", args.skip_files)
    elif args.data_type == 'simulation_low':
        residual_data = load_data("data/simulation_data_low/", "res", args.skip_files)
        error_data = load_data("data/simulation_data_low/", "e", args.skip_files)
    elif args.data_type == 'pressure':
        residual_data = load_data("data/pressure_data/", "RHS", args.skip_files)
        error_data = load_data("data/pressure_data/", "p", args.skip_files)
    elif args.data_type == 'pca':
        residual_data = load_data("data/simulation_data/", "res", args.skip_files)

        eigenvectors_input= load_data("data/eigenvectors_data/", "b", 1)
        residual_data = torch.cat([residual_data, eigenvectors_input], dim=0)

        error_data = load_data("data/simulation_data/", "e", args.skip_files)
        eigenvectors_output = load_data("data/eigenvectors_data/", "x", 1)

        error_data = torch.cat([error_data, eigenvectors_output], dim=0)
        
    else:
        residual_data = load_data("data/simulation_data/", "res", args.skip_files)
        error_data = load_data("data/simulation_data/", "e", args.skip_files)

    print("Residual data shape:", residual_data.shape)
    print("Error data shape:", error_data.shape)

    # Prepare data
    grid_size_x = residual_data.shape[1]
    grid_size_y = residual_data.shape[2]
    dx = 1 / (grid_size_x)
    dy = 1 / (grid_size_y)
    dx2 = dx ** 2
    dy2 = dy ** 2
    vector_size = np.min([grid_size_x, grid_size_y])
    residual_data = residual_data.view(residual_data.shape[0], 1, grid_size_x, grid_size_y).float()
    residual_data = residual_data.to("cuda")
    error_data = error_data.view(error_data.shape[0], 1, grid_size_x, grid_size_y).float()
    error_data = error_data.to("cuda")

    if args.model == 'fourier':
        model = FourierModel()
        model.to("cuda")
    elif args.model == 'nobias':
        model = NoBiasModel()
        model.to("cuda")
    elif args.model == 'azulay':
        model = AzulayModel()
        model.to("cuda")
    else :
        model = Model()
        model.to("cuda")

    if args.act == 'elu':
        model.act = nn.ELU()
    elif args.act == 'gelu':
        model.act = nn.GELU()
    elif args.act == 'leaky_relu':
        model.act = nn.LeakyReLU()
    elif args.act == 'prelu':
        model.act = nn.PReLU()
    elif args.act == 'relu':
        model.act = nn.ReLU()
    elif args.act == 'selu':
        model.act = nn.SELU()
    elif args.act == 'sigmoid':
        model.act = nn.Sigmoid()
    elif args.act == 'swish':
        model.act = nn.SiLU()
    elif args.act == 'softmax':
        model.act = nn.Softmax()
    elif args.act == 'softplus':
        model.act = nn.Softplus()
    elif args.act == 'tanh':
        model.act = nn.Tanh()
    elif args.act == 'mish':
        model.act = nn.Mish()
    elif args.act == 'linear':
        model.act = nn.Identity()

    # Split data into train, validation, and test sets
    # Define batch size
    batch_size = args.batch_size
    total_samples = residual_data.shape[0]
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    train_data, val_data, test_data = random_split(
        list(zip(residual_data, error_data)), 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Convert to DataLoader for easier handling
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    if args.loss_function == 'cosine_similarity':
        loss_fn = loss_cosine_similarity
    elif args.loss_function == 'mse':
        loss_fn = loss_mse
    elif args.loss_function == 'rmse':
        loss_fn = loss_rmse
    elif args.loss_function == 'huber':
        loss_fn = loss_huber
    else:
        raise ValueError("Invalid loss function")

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5, amsgrad=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    writer = SummaryWriter('logs')

    # Train the model
    num_epochs = args.epochs

    epoch_train_losses = []
    epoch_val_losses = []
    
    # Train the model
    for epoch in tqdm(range(num_epochs)):
        # Training loop
        model.train()
        train_losses = []
        for batch_residual_data, batch_error_data in train_loader:
            predicted_error_vector = model(batch_residual_data)
            loss = loss_fn(predicted_error_vector, batch_error_data, batch_residual_data, grid_size_x, grid_size_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss)

        # Calculate average training loss for the epoch
        avg_train_loss = torch.stack(train_losses).mean()
        epoch_train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch_residual_data, batch_error_data in val_loader:
                # Forward pass and loss calculation
                # Append validation loss to val_losses list
                predicted_error_vector = model(batch_residual_data)
                val_loss = loss_fn(predicted_error_vector, batch_error_data, batch_residual_data, grid_size_x, grid_size_y)
                val_losses.append(val_loss)

        # Calculate average validation loss for the epoch
        avg_val_loss = torch.stack(val_losses).mean()
        epoch_val_losses.append(avg_val_loss)

        # Update the learning rate scheduler
        scheduler.step(avg_train_loss)

        # Print or log training and validation loss
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss.item()}, Val Loss: {avg_val_loss.item()}")

    writer.close()

    model_variation = f"{args.data_type}_{args.loss_function}_skip{args.skip_files}_batch{args.batch_size}_model{args.model}_act{args.act}"

    # Evaluate the model on the test set
    model.eval()
    test_losses = []
    with torch.no_grad():
        for test_residual_data, test_error_data in test_loader:
            predicted_error_vector = model(test_residual_data)
            test_loss = loss_fn(predicted_error_vector, test_error_data, test_residual_data, grid_size_x, grid_size_y)
            print(f"Test Loss: {test_loss}")
            test_losses.append(test_loss)

    avg_test_loss = torch.stack(test_losses).mean()
    print(f"Average Test Loss: {avg_test_loss.item()}")

    # plot train / val / test loss
    epoch_train_losses = [x.item() for x in epoch_train_losses]
    epoch_val_losses = [x.item() for x in epoch_val_losses]

    import matplotlib.pyplot as plt
    plt.plot(epoch_train_losses, label='train loss')
    plt.plot(epoch_val_losses, label='val loss')
    # add labels
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"paper_assets/{model_variation}_loss_plot.pdf", format="pdf")

    # Convert to Torchscript via Annotation
    model.eval()
    traced = torch.jit.trace(model, test_residual_data[0].unsqueeze(0))
    traced.save(f"models/{model_variation}.pt")
