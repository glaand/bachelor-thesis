import numpy as np
import os
import shutil

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

# Compute eigenvalues and eigenvectors
print('Computing eigenvalues and eigenvectors...')
eigenvalues, eigenvectors = np.linalg.eig(A)

eigenvectors_mapped = []
for i in range(eigenvectors.shape[1]):
    eigenvectors_mapped.append(eigenvectors[:,i])


# Augment data by generating linear combinations of eigenvectors
b_vectors = eigenvectors_mapped
for _ in range(2):
    for i in range(len(eigenvectors_mapped)):
        # generate random linear combination of eigenvectors
        b_vector = np.zeros_like(eigenvectors_mapped[i])
        for c in range(eigenvectors_mapped[i].shape[0]):
            b_vector[c] = np.random.uniform(-1, 1) * eigenvectors_mapped[i][c]
        b_vectors.append(b_vector)


# delete directory if it exists
directory = 'eigenvectors'
if os.path.exists(directory):
    shutil.rmtree(directory)

# create directory
os.makedirs(directory)

# save eigenvectors to files
for i, b_vector in enumerate(b_vectors):
    b_vector = b_vector.reshape(grid_size_x, grid_size_y)
    x = A_inverse @ b_vector.flatten()
    x = x.reshape(grid_size_x, grid_size_y)
    np.savetxt(directory + '/b_' + str(i) + '.dat', b_vector)
    np.savetxt(directory + '/x_' + str(i) + '.dat', x)
    print('Saved b_vector_' + str(i) + '.dat')
print('Done!')
