import numpy as np

def rotate_matrix(matrix, degrees):
    """
    Rotates a 2D numpy matrix counter-clockwise around its center by specified degrees.
    
    Parameters:
    matrix (numpy.ndarray): Input 2D matrix.
    degrees (float): Rotation angle in degrees (counter-clockwise).
    cd ..
    
    Returns:
    numpy.ndarray: Rotated matrix with the same shape as the input.
    """
    theta = np.radians(-degrees)  # Convert to radians and negate for correct direction
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, s], [-s, c]])  # Rotation matrix
    
    h, w = matrix.shape
    cy, cx = (h-1)/2.0, (w-1)/2.0  # Center coordinates
    
    # Generate grid of target coordinates
    y_out, x_out = np.indices((h, w))
    x_out_centered = x_out - cx
    y_out_centered = y_out - cy
    
    # Apply inverse rotation to find source coordinates
    x_orig_centered = rotation_matrix[0,0] * x_out_centered + rotation_matrix[0,1] * y_out_centered
    y_orig_centered = rotation_matrix[1,0] * x_out_centered + rotation_matrix[1,1] * y_out_centered
    x_orig = x_orig_centered + cx
    y_orig = y_orig_centered + cy
    
    # Bilinear interpolation
    x_floor = np.floor(x_orig).astype(int)
    y_floor = np.floor(y_orig).astype(int)
    dx = x_orig - x_floor
    dy = y_orig - y_floor
    
    # Handle out-of-bounds coordinates and clip
    x0 = np.clip(x_floor, 0, w-1)
    y0 = np.clip(y_floor, 0, h-1)
    x1 = np.clip(x_floor + 1, 0, w-1)
    y1 = np.clip(y_floor + 1, 0, h-1)
    
    # Validate coordinates (set to 0 if out of bounds)
    valid_x0 = (x_floor >= 0) & (x_floor < w)
    valid_y0 = (y_floor >= 0) & (y_floor < h)
    valid_x1 = (x_floor+1 >= 0) & (x_floor+1 < w)
    valid_y1 = (y_floor+1 >= 0) & (y_floor+1 < h)
    
    # Sample values with validity masks
    v00 = matrix[y0, x0] * (valid_x0 & valid_y0)
    v10 = matrix[y0, x1] * (valid_x1 & valid_y0)
    v01 = matrix[y1, x0] * (valid_x0 & valid_y1)
    v11 = matrix[y1, x1] * (valid_x1 & valid_y1)
    
    # Interpolation formula
    interpolated = (
        (v00 * (1 - dx) + v10 * dx) * (1 - dy) +
        (v01 * (1 - dx) + v11 * dx) * dy
    )
    
    return interpolated.astype(matrix.dtype)

if __name__ == '__main__':
    heightmap = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 2, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)

    # Rotate by 45 degrees
    rotated = rotate_matrix(heightmap, 85)

    print("Original heightmap:")
    print(heightmap)
    print("\nRotated heightmap:")
    print(rotated)