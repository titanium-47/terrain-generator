import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.grad_utils import grad_map, second_derivatives_map, laplacian_map  # Imported second_derivatives_map

# Load the heightmap
heightmap = np.load('height_maps/ditches15.npy')

# Convert heightmap to torch.Tensor
heightmap_tensor = torch.from_numpy(heightmap)

# Compute the gradient map
gradient_map = grad_map(heightmap_tensor.unsqueeze(0)).squeeze(0).numpy()

# Compute the second derivatives
second_derivatives = second_derivatives_map(heightmap_tensor.unsqueeze(0)).squeeze(0).numpy()

# Extract the gradient components
gradient_x = gradient_map[..., 0]
gradient_y = gradient_map[..., 1]

# Extract the second derivative components
second_x = second_derivatives[..., 0]
second_y = second_derivatives[..., 1]

laplacian = laplacian_map(heightmap_tensor.unsqueeze(0)).squeeze(0).numpy()

# Create a grid of coordinates
X, Y = np.meshgrid(np.arange(heightmap.shape[0]), np.arange(heightmap.shape[1]))


# Compute the squared gradient magnitudes
grad_magnitude_squared = np.sqrt(np.sqrt(gradient_x**2 + gradient_y**2))

# Plot the heightmap
plt.imshow(heightmap, cmap='terrain', origin='lower')  # Set origin to 'lower'
plt.colorbar()

# Plot the gradient vector field
plt.quiver(X, Y, gradient_x, gradient_y, color='white')

# plt.show()

# Plot the second derivative vector field
plt.figure()
plt.imshow(heightmap, cmap='terrain', origin='lower')
plt.colorbar()
plt.quiver(X, Y, second_x, second_y, color='red')
plt.title('Second Derivative Vector Field')

# plt.show()

# Plot the Laplacian map
plt.figure()
plt.imshow(laplacian, cmap='plasma', origin='lower')
plt.colorbar()
plt.title('Laplacian Map')

plt.show()
