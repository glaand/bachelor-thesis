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
    def __init__(self, N, dim, fil_num):
        super(Kaneda, self).__init__()
        self.N = N
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

        self.act = nn.SiLU()

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
    # Compute losses
    loss_deep_learning = torch.sqrt(torch.mean((pred_error - true_error) ** 2, dim=[2, 3]))
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


    model = Kaneda(vector_size, 1, 16)
    model.to("cuda")

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5, amsgrad=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # Set the model in training mode
    model.train()

    writer = SummaryWriter('logs')

    # Train the model
    num_epochs = 100
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
    traced.save("model_kaneda.pt")

    # generate random tensor to test the model
    input_tensor = test_residual_data[0].unsqueeze(0)
    output = traced(input_tensor)
    # save output to e.dat
    output = output.cpu().detach().numpy()
    np.savetxt("e.dat", output.reshape(vector_size, vector_size))
    print(output)
    print(output.shape)