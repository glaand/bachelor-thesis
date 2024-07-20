import numpy as np
import shutil
from tqdm import tqdm

grid_size_x = 34
grid_size_y = 34
dx = 1 / (grid_size_x)
dy = 1 / (grid_size_y)
dx2 = dx ** 2
dy2 = dy ** 2

# Generate Matrix A which is the discretized Laplacian in 2D
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

# Delete folder if exists
shutil.rmtree('data/random_data', ignore_errors=True)

# Create folder
shutil.os.mkdir('data/random_data')

# Parameters
n = 34  # Size of the grid (34x34)
num_samples = 1156  # Number of samples

# Function to generate a band matrix
def generate_band_matrix(n, bandwidth):
    band_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            band_matrix[i, j] = np.random.randn()
    return band_matrix

# Function to generate eigenvectors of a band matrix
def generate_eigenvectors(n, bandwidth):
    band_matrix = generate_band_matrix(n, bandwidth)
    symmetric_band_matrix = (band_matrix + band_matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(band_matrix)
    return eigvecs

# Generate eigenvectors of a band matrix
bandwidth = 5  # Example bandwidth
eigvecs = generate_eigenvectors(n * n, bandwidth)

# Generate data
for i in tqdm(range(num_samples)):
    b = eigvecs[:, i]
    x = np.dot(A_inverse, b)
    np.savetxt(f'data/random_data/b_{i}.dat', b.reshape((n, n)))
    np.savetxt(f'data/random_data/x_{i}.dat', x.reshape((n, n)))
