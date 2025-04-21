import numpy as np
import os

# File paths
detections_file = r"./detections.txt"  # Input file
output_file = r"./detections_with_grid.txt"  # Output file

# Pitch dimensions (Change these based on your actual video resolution)
pitch_width = 1280  # Adjust based on your video
pitch_height = 720   # Adjust based on your video
grid_rows, grid_cols = 10, 10  # 10x10 grid

# Compute grid cell sizes
box_width = pitch_width // grid_cols
box_height = pitch_height // grid_rows

def get_grid_index(x, y):
    """Return the grid cell number for a given (x, y) coordinate."""
    col = min(x // box_width, grid_cols - 1)  # Ensure we don't go out of bounds
    row = min(y // box_height, grid_rows - 1)
    return row * grid_cols + col  # Convert 2D (row, col) to 1D index

# Read detections.txt
if not os.path.exists(detections_file):
    raise FileNotFoundError(f"Detections file not found: {detections_file}")

detections_with_grid = []

with open(detections_file, 'r') as f:
    detection_data = f.readlines()

for line in detection_data:
    frame_id, obj_id, x, y, w, h, filtered_x, filtered_y, estimated_z = map(float, line.strip().split())
    
    # Compute grid index using filtered ball position
    grid_index = get_grid_index(int(filtered_x), int(filtered_y))

    # Store the updated detection with grid info
    detections_with_grid.append(f"{int(frame_id)} {int(obj_id)} {int(x)} {int(y)} {int(w)} {int(h)} {grid_index} {int(filtered_x)} {int(filtered_y)} {estimated_z:.2f}\n")

# Save to detections_with_grid.txt
with open(output_file, 'w') as f:
    f.writelines(detections_with_grid)

print(f"Updated detections with grid saved to {output_file}")
