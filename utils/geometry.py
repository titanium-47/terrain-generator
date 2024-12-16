import numpy as np

def normalize(height_map: np.ndarray) -> np.ndarray:
    sum_heights = np.sum(height_map)
    num_elements = height_map.size
    avg_height = sum_heights / num_elements
    return height_map - avg_height
    

def create_geometry(
        height_map: list[list[float]], 
        len_scale: float = 1,
        height_scale: float = 1, 
        zero_height: bool = False,
        norm_height: bool = False
        ) -> list[tuple[float]]:
    
    if norm_height:
        height_map = np.array(height_map, dtype=np.float32)
        min_height = np.min(height_map)
        height_map -= min_height
        max_height = np.max(height_map)
        height_map /= max_height
        height_map *= height_scale
    else:
        height_map = np.array(height_map, dtype=np.float32) * height_scale

    if zero_height:
        height_map = normalize(height_map)
    
    points = []
    vertex_counts = []
    face_vertex_indices = []
    rows, cols = height_map.shape
    for y in range(rows):
        for x in range(cols):
            height = height_map[y, x]
            points.append(((x-cols/2)*len_scale, float(height), (y-rows/2)*len_scale))
            if x > 0 and y > 0:
                idx = len(points) - 1
                face_vertex_indices.extend([idx, idx-cols, idx-1])
                face_vertex_indices.extend([idx-cols-1, idx-1, idx-cols])
                vertex_counts.append(3)
                vertex_counts.append(3)
    return points, vertex_counts, face_vertex_indices, height_map