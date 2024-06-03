import numpy as np
import os
import shutil
from tqdm import tqdm

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

def create_low_pass_filter(shape, radius):
    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1
    return mask

def pipeline(data, filepath):
    # Perform Fourier Transform
    fourier_transform = np.fft.fft2(data)

    # Perform Shift
    shifted = np.fft.fftshift(fourier_transform)

    radius = 5  # Adjust radius to keep more or less low-frequency components
    low_pass_filter = create_low_pass_filter(shifted.shape, radius)

    # Apply the filter
    filtered = shifted * low_pass_filter

    # Perform Inverse Shift
    inverse_shifted = np.fft.ifftshift(filtered)

    # Perform Inverse Fourier Transform
    inverse_fourier_transform = np.fft.ifft2(inverse_shifted)

    # Take the abs part
    inverse_fourier_transform = np.abs(inverse_fourier_transform)

    return inverse_fourier_transform


if __name__ == "__main__":
    # Define the directory
    directory = "data/simulation_data/"

    # Initialize lists to hold the loaded data
    residual_files = []

    # Iterate over files in the directory and load the relevant ones
    for file in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, file)
        if file.startswith("res"):
            residual_files.append(np.loadtxt(file_path))

    # Check if directory exists, if yes, delete it
    low_frequent_directory = "data/simulation_data_low"

    if os.path.exists(low_frequent_directory):
        shutil.rmtree(low_frequent_directory)

    # Create the directory
    os.makedirs(low_frequent_directory)

    # Process the residual files with tqdm
    for i, residual_file in enumerate(tqdm(residual_files)):
        residual_grid = pipeline(residual_file, os.path.join(low_frequent_directory, f"res_{i}.dat"))
        np.savetxt(os.path.join(low_frequent_directory, f"res_{i}.dat"), residual_grid)

        # grid-like vector to one-dimensional vector
        residual_vector = residual_grid.flatten()

        error_vector = A_inverse @ residual_vector

        # one-dimensional vector to grid-like vector
        error_grid = error_vector.reshape(grid_size_x, grid_size_y)

        np.savetxt(os.path.join(low_frequent_directory, f"e_{i}.dat"), error_grid)