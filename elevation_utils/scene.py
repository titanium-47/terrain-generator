import numpy as np
from generate_heightmap import generate_heightmap

def generate_points(num_points, x_range, y_range, min_dist, existing_arrays=None, max_attempts=10000):
    """
    Generates an array of points with minimum distance between each other and existing points.
    
    Parameters:
    num_points (int): Number of points to generate.
    x_range (tuple): (x_min, x_max) range for x-coordinate.
    y_range (tuple): (y_min, y_max) range for y-coordinate.
    min_dist (float): Minimum distance required between any two points.
    existing_arrays (list of np.ndarray): Existing points to avoid.
    max_attempts (int): Maximum number of attempts to place each point.
    
    Returns:
    np.ndarray: Array of shape (num_points, 2) with generated points.
    """
    points = []
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    while len(points) < num_points:
        for _ in range(max_attempts):
            # Generate a candidate point
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            candidate = np.array([x, y])
            
            # Collect all existing points from existing_arrays and current points
            existing_points_list = []
            if existing_arrays is not None:
                for arr in existing_arrays:
                    existing_points_list.append(arr)
            if len(points) > 0:
                existing_points_list.append(np.array(points))
            
            # Check distances against all existing points
            if existing_points_list:
                all_existing = np.concatenate(existing_points_list, axis=0)
                distances = np.linalg.norm(all_existing - candidate, axis=1)
                min_distance = np.min(distances) if distances.size > 0 else np.inf
            else:
                min_distance = np.inf  # No existing points
            
            if min_distance > min_dist:
                points.append(candidate)
                break
        else:
            raise RuntimeError(f"Failed to place point {len(points)+1} after {max_attempts} attempts.")
    
    return np.array(points)

# Example output (commented out)
# print("Array with 50 points (0.5m apart):")
# print(array_50)
# print("\nArray with 100 points (0.2m apart from each other and array_50):")
# print(array_100)

# Generate the first array with 50 points, min distance 0.5
ramps = generate_points(250, (-98.5, 98.5), (-3.5, 3.5), 1.5)

ramp_yaws = np.random.uniform(3*np.pi/4, 5*np.pi/4, 250)

# Generate the second array with 100 points, min distance 0.2
blocks = generate_points(500, (-98.5, 98.5), (-3.5, 3.5), 0.5, existing_arrays=[ramps])

block_yaws = np.random.uniform(0, np.pi/2, 500)

ramps = np.concat([ramps, ramp_yaws[:, np.newaxis]], axis=1)

blocks = np.concat([blocks, block_yaws[:, np.newaxis]], axis=1)
# Example output (commented out to prevent execution here)
# print("Array with 50 points:")
# print(array_50)
# print("\nArray with 100 points:")
# print(array_100)

raw_map = np.load('heightmap.npy')
ramp_map = np.load('ramp.npy')
block_map = np.load('block.npy')

obstacles = []

for block in blocks:
    obstacles.append({
                'type': 'block',
                'position': (block[0],
                            block[1],
                            0),
                'orientation': block[2]
            })

for ramp in ramps:
    obstacles.append({
                'type': 'ramp',
                'position': (ramp[0],
                            ramp[1],
                            0),
                'orientation': ramp[2]
            })
    

height_map = generate_heightmap(obstacles, raw_map, block_map, ramp_map)

np.save('obstacle_map2.npy', height_map)