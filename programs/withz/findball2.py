import torch
import cv2
import numpy as np
import os
from filterpy.kalman import KalmanFilter

# Ensure correct file paths
video_input_path = r"../../rawvideos/slow.mp4"  # Input video path
detections_output_file = r"./detections.txt"  # Output detections file

# Load YOLOv5 model (Ensure the model path is correct)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../../model/best.pt')
model.conf = 0.2  # Confidence threshold
model.iou = 0.45  # IoU threshold
model.to(device)  # Move model to GPU

# Initialize Kalman Filter for X, Y
kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, velocity_x, velocity_y]
kf.F = np.array([[1, 0, 1, 0],  
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])  # State transition model

kf.H = np.array([[1, 0, 0, 0],  
                 [0, 1, 0, 0]])  # Measurement function

kf.P *= 1000  # Initial uncertainty
kf.R = np.array([[10, 0], [0, 10]])  # Measurement noise
kf.Q = np.array([[1, 0, 0, 0],  
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])  # Process noise
kf.x = np.array([[0], [0], [0], [0]])  # Initial state

# Function to apply Kalman filter on (x, y)
def apply_kalman_filter(measured_x, measured_y):
    """Update Kalman Filter with new measurement and predict next position."""
    kf.predict()
    kf.update([measured_x, measured_y])
    
    # Extract scalar values from NumPy array
    predicted_x, predicted_y = int(kf.x[0, 0]), int(kf.x[1, 0])
    
    return predicted_x, predicted_y

# Function to estimate Z (depth)
def estimate_z(width, initial_distance=0, min_z=0.1, max_z=22.0):
    """
    Estimate ball's position along the Z-axis (depth).
    
    - Smaller width (`w`) → ball is further away
    - Larger width (`w`) → ball is closer to the camera
    - We scale Z based on bounding box width.
    """
    scale_factor = 150  # Adjust based on camera calibration

    # Estimate distance from the camera (inverse relation to bounding box width)
    estimated_z = initial_distance - (scale_factor / max(width, 1))

    return max(min(estimated_z, max_z), min_z)  # Ensure Z is within realistic range

# Open video file
cap = cv2.VideoCapture(video_input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_id = 0
detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB format (required for YOLO)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)  # Perform YOLOv5 inference

    # Move results back to CPU before processing
    predictions = results.xyxy[0].cpu().numpy()

    if predictions is None or len(predictions) == 0:
        frame_id += 1
        continue

    for det in predictions:
        if len(det) < 6:  # Ensure correct number of values before unpacking
            print(f"Skipping invalid detection: {det}")  # Debugging print
            continue  

        x1, y1, x2, y2, conf, cls = map(float, det)  # Convert values to float
        obj_id = int(cls)  # Convert class ID to integer
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

        center_x, center_y = x + w // 2, y + h // 2  # Compute ball center

        # Use Kalman Filter to predict and smooth position
        filtered_x, filtered_y = apply_kalman_filter(center_x, center_y)

        # Estimate Z (depth in 3D space)
        estimated_z = estimate_z(w)

        detections.append(f"{frame_id} {obj_id} {x} {y} {w} {h} {filtered_x} {filtered_y} {estimated_z:.2f}\n")

    frame_id += 1

cap.release()

# Ensure directory exists
output_dir = os.path.dirname(detections_output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save detections to file
with open(detections_output_file, 'w') as f:
    f.writelines(detections)

print(f"✅ Detections saved to {detections_output_file}")
