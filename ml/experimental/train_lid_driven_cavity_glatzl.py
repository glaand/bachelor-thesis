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

class Glatzl(nn.Module):
    def __init__(self):
        super(Glatzl, self).__init__()
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

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
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

    def forward(self, error):
        # x, get inner without boundary
        x = error[:, :, 1:-1, 1:-1]

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

        # add zero boundaries
        x11 = F.pad(x11, (1, 1, 1, 1), "constant", 0)

        return x11

def custom_loss(pred_correction, ideal_error, input_error, grid_size_x, grid_size_y):
    # Compute losses, mean absolute error
    mae = nn.L1Loss()
    total_loss = mae(pred_correction+input_error, ideal_error)
    return total_loss

if __name__ == "__main__":
    # random state 
    torch.manual_seed(42)
    # Define data loading function
    def load_data(folder_path, prefix, skip=1):
        print(f"Skipping every {skip} files")
        data = []
        files = glob.glob(os.path.join(folder_path, f"{prefix}_*.dat"))
        files.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
        for i, file in enumerate(tqdm(files)):
            if i % skip == 0:
                loaded_data = np.loadtxt(file)
                data.append(torch.tensor(loaded_data))
        return torch.stack(data)

    # Load data
    input_error = load_data("ML_data/", "input_error", 100)[:10]
    ideal_error = load_data("ML_data/", "ideal_error", 100)[:10]

    print("Input error data shape:", input_error.shape)
    print("Ideal error data shape:", ideal_error.shape)

    # Prepare data
    grid_size_x = 34
    grid_size_y = 34
    dx = 1 / (grid_size_x)
    dy = 1 / (grid_size_y)
    dx2 = dx ** 2
    dy2 = dy ** 2
    vector_size = np.min([grid_size_x, grid_size_y])
    input_error = input_error.view(input_error.shape[0], 1, grid_size_x, grid_size_y).float()
    input_error = input_error.to("cuda")
    ideal_error = ideal_error.view(ideal_error.shape[0], 1, grid_size_x, grid_size_y).float()
    ideal_error = ideal_error.to("cuda")

    # Split data into train and test sets
    total_samples = input_error.shape[0]
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size

    train_data, test_data = random_split(
        list(zip(input_error, ideal_error)), 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_input_error, train_ideal_error = zip(*train_data)
    test_input_error, test_ideal_error = zip(*test_data)

    train_input_error = torch.stack(train_input_error).to("cuda")
    train_ideal_error = torch.stack(train_ideal_error).to("cuda")
    test_input_error = torch.stack(test_input_error).to("cuda")
    test_ideal_error = torch.stack(test_ideal_error).to("cuda")


    model = Glatzl()
    model.to("cuda")

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5, amsgrad=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.1, verbose=True)

    # Set the model in training mode
    model.train()

    writer = SummaryWriter('logs')

    # Train the model
    num_epochs = 10000
    for epoch in tqdm(range(num_epochs)):
        # Forward pass
        predicted_error_vector = model(train_input_error)

        # Calculate the loss
        loss = custom_loss(predicted_error_vector, train_ideal_error, train_input_error, grid_size_x, grid_size_y)

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
        predicted_error_vector = model(test_input_error)
        test_loss = custom_loss(predicted_error_vector, test_ideal_error, test_input_error, grid_size_x, grid_size_y)
        print(f"Test Loss: {test_loss}")

    # Convert to Torchscript via Annotation
    model.eval()
    traced = torch.jit.trace(model, test_input_error[0].unsqueeze(0))
    traced.save("model_glatzl.pt")

    # generate random tensor to test the model
    input_tensor = test_input_error[0].unsqueeze(0)
    output = traced(input_tensor)
    # save output to e.dat
    output = output.squeeze(0).squeeze(0).cpu().detach().numpy()
