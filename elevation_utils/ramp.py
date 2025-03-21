import numpy as np

# Parameters
slope_length = 0.63  # Length of each slope in meters
flat_length = 0.25   # Length of the flat part in meters
total_ramp_length = 2 * slope_length + flat_length  # Total length of the ramp
width = 0.5          # Width of the ramp in meters
resolution = 0.1     # Resolution in meters per cell

# Calculate x positions along the length, ensuring coverage up to the next resolution step beyond total_ramp_length
x_positions = np.arange(0, total_ramp_length + resolution, resolution)

# Compute heights for each x position
heights = []
for x in x_positions:
    if x < slope_length:
        # Ascending slope
        height = (x / slope_length) * 0.25
    elif x <= slope_length + flat_length:
        # Flat part
        height = 0.25
    elif x <= total_ramp_length:
        # Descending slope
        distance_along_slope = x - (slope_length + flat_length)
        height = 0.25 - (distance_along_slope / slope_length) * 0.25
    else:
        # Beyond the ramp
        height = 0.0
    heights.append(height)

# Number of columns based on width and resolution
num_columns = int(width / resolution)

# Create the 2D heightmap by replicating each height across all columns
heightmap = np.tile(np.array(heights)[:, np.newaxis], (1, num_columns))

heightmap = np.pad(heightmap, pad_width=5, mode='constant', constant_values=0)

np.save('ramp.npy', heightmap)