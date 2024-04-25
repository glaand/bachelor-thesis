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

class Kaneda(nn.Module):
    def __init__(self):
        super(Kaneda, self).__init__()
        fil_num = 16
        self.conv1 = nn.Conv2d(1, fil_num, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv6 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv7 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv8 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv9 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv10 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv11 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)
        self.conv12 = nn.Conv2d(fil_num, fil_num, kernel_size=(3, 3), padding=1)

        self.reduce_channels = nn.Conv2d(fil_num, 1, kernel_size=(3, 3), padding=1)  # Reduce channels to 1

        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))

        self.act = nn.ReLU()

    def forward(self, x):
        # normalise input
        x = x / torch.max(x)
        x = self.conv1(x)
        la = self.act(self.conv2(x))
        lb = self.act(self.conv3(la))
        la = self.act(self.conv4(lb)) + la
        lb = self.act(self.conv5(la))
        
        apa = self.avgpool(lb)
        apb = self.act(self.conv6(apa))
        apa = self.act(self.conv7(apb)) + apa
        apb = self.act(self.conv8(apa))
        apa = self.act(self.conv9(apb)) + apa
        apb = self.act(self.conv10(apa))
        apa = self.act(self.conv11(apb)) + apa
        apb = self.act(self.conv12(apa))
        apa = self.act(self.conv11(apb)) + apa
        apb = self.act(self.conv12(apa))
        apa = self.act(self.conv11(apb)) + apa

        upa = F.interpolate(apa, scale_factor=2, mode='bicubic') + lb
        upb = self.act(self.conv5(upa))
        upa = self.act(self.conv4(upb)) + upa
        upb = self.act(self.conv3(upa))
        upa = self.act(self.conv2(upb)) + upa
        upb = self.act(self.conv1(self.reduce_channels(upa)))
        upa = self.act(self.conv2(upb)) + upa

        out = self.reduce_channels(upa)

        # set boundary to 0
        out[:, :, 0, :] = 0
        out[:, :, -1, :] = 0
        out[:, :, :, 0] = 0
        out[:, :, :, -1] = 0

        return out
    
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
    
    # Total loss
    total_loss = alignment_loss
    
    return total_loss

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
    residual_data = load_data("eigenvectors/", "b", 1)
    error_data = load_data("eigenvectors/", "x", 1)
    # only first 100
    residual_data = residual_data
    error_data = error_data

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
    model.to("cuda")

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5, amsgrad=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # Set the model in training mode
    model.train()

    writer = SummaryWriter('logs')

    # Define batch size
    batch_size = 8

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
    traced.save("model_eigenvectors.pt")

    # generate random tensor to test the model
    input_tensor = train_residual_data[0].unsqueeze(0)
    output = traced(input_tensor)
    # save output to e.dat
    output = output.squeeze(0).squeeze(0).cpu().detach().numpy()
    np.savetxt("e.dat", output)
    print(output)
    print(output.shape)