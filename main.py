from pxr import Usd, UsdGeom
import numpy as np
from PIL import Image

from utils.exporter import export_geometry
from utils.geometry import create_geometry

#define a simple height map
# HEIGHT_MAP = [
#     [0, 0, 0, 0, 0],
#     [0, 0, 1, 2, 0],
#     [0, 1, 2, 1, 1],
#     [0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0]
# ]

HEIGHT_MAP_NAME = "dessert"
HEIGHT_MAP_PATH = f"./preprocessed/{HEIGHT_MAP_NAME}.png"
image = Image.open(HEIGHT_MAP_PATH)
height_map = np.array(image)

h_scale = 2048/100.0 * 0.878 #3d print width

points, vertex_counts, face_vertex_indices, height_map = create_geometry(height_map, height_scale=h_scale, len_scale=1/4.0)
output_path = f"{HEIGHT_MAP_NAME}_mesh.usd"
export_geometry(points, vertex_counts, face_vertex_indices, output_path, smooth=True)

np.save(f"height_maps/{HEIGHT_MAP_NAME}.npy", height_map)