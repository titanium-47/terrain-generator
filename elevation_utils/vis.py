import numpy as np
import matplotlib.pyplot as plt
from elev import crop_heightmap, rotate_matrix

# Load the elevation map from .npy file
elevation_map = np.load('obstacle_map2.npy')  # Replace with your file path

# Get map dimensions
rows, cols = elevation_map.shape
pixel_size = 0.1  # meters per pixel

# Calculate physical dimensions
width = cols * pixel_size
height = rows * pixel_size

# Create plot with proper scaling
plt.figure(figsize=(10, 8))
img = plt.imshow(elevation_map,
                 extent=[0, width, 0, height],
                 cmap='terrain',
                 origin='lower',  # Change to 'upper' if data starts from top
                 aspect='auto')  # Use 'equal' for equal aspect ratio

# Add labels and colorbar
plt.xlabel('X Position (meters)', fontsize=12)
plt.ylabel('Y Position (meters)', fontsize=12)
plt.title('High-Resolution Elevation Map', fontsize=14)
cbar = plt.colorbar(img)
cbar.set_label('Elevation (meters)', fontsize=12)

# rotated = rotate_matrix(elevation_map, np.pi/6)
# plt.figure(figsize=(10, 8))
# img = plt.imshow(rotated,
#                  extent=[0, 2, 0, 2],
#                  cmap='terrain',
#                  origin='lower',  # Change to 'upper' if data starts from top
#                  aspect='auto')

cropped_map = crop_heightmap(elevation_map, -2, -3, 90, width=20)
print(cropped_map)
plt.figure(figsize=(10, 8))
img = plt.imshow(cropped_map,
                 extent=[0, 2, 0, 2],
                 cmap='terrain',
                 origin='lower',  # Change to 'upper' if data starts from top
                 aspect='auto')
plt.xlabel('X Position (meters)', fontsize=12)
plt.ylabel('Y Position (meters)', fontsize=12)
plt.title('High-Resolution Elevation Map', fontsize=14)
cbar = plt.colorbar(img)
cbar.set_label('Elevation (meters)', fontsize=12)

plt.show()