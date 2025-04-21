import torch
import cv2
import numpy as np
import os
from filterpy.kalman import KalmanFilter

# Ensure correct file paths
video_input_path = r"../rawvideos/v1.mp4"  # Input video path
video_name = os.path.splitext(os.path.basename(video_input_path))[0]  # Extract filename without extension
detections_output_file = os.path.join(os.path.dirname(os.path.abspath(video_input_path)), f"{video_name}.txt")  # Save in same directory
# Load YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../model/best.pt')
model.conf = 0.4  # Confidence threshold
model.iou = 0.45  # IoU threshold
model.to(device)  # Move model to GPU

# Initialize Kalman Filter
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

def apply_kalman_filter(measured_x, measured_y):
    """ Update Kalman Filter with new measurement and predict next position """
    kf.predict()
    kf.update([measured_x, measured_y])
    return int(kf.x[0, 0]), int(kf.x[1, 0])

def is_irregular(prev_x, prev_y, new_x, new_y, threshold=50):
    """ Check if movement is too large to be realistic """
    return abs(new_x - prev_x) > threshold or abs(new_y - prev_y) > threshold

# Open video file
cap = cv2.VideoCapture(video_input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_id = 0
detections = []
prev_x, prev_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)  # Perform YOLOv5 inference
    predictions = results.xyxy[0].cpu().numpy()
    
    if predictions is None or len(predictions) == 0:
        # No detection, use Kalman prediction
        filtered_x, filtered_y = apply_kalman_filter(prev_x, prev_y) if prev_x is not None else (0, 0)
        final_x, final_y = filtered_x, filtered_y
        decision = 0  # 0 means finalized by Kalman
        detections.append(f"{frame_id} -1 -1 {filtered_x} {filtered_y} -1 {final_x} {final_y} {decision}\n")
    else:
        # Select the highest confidence detection
        best_detection = max(predictions, key=lambda det: det[4])  # Highest confidence
        x1, y1, x2, y2, conf, cls = map(float, best_detection)
        obj_id = int(cls)
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        center_x, center_y = x + w // 2, y + h // 2
        filtered_x, filtered_y = apply_kalman_filter(center_x, center_y)
        
        # Check if detected movement is too irregular
        if prev_x is not None and is_irregular(prev_x, prev_y, center_x, center_y):
            final_x, final_y = filtered_x, filtered_y  # Use Kalman prediction
            decision = 0  # 0 means finalized by Kalman
        else:
            final_x, final_y = center_x, center_y  # Use YOLO detection
            decision = 1  # 1 means finalized by YOLO
        
        detections.append(f"{frame_id} {center_x} {center_y} {filtered_x} {filtered_y} {conf:.2f} {final_x} {final_y} {decision}\n")
        prev_x, prev_y = final_x, final_y  # Store last valid position
    
    frame_id += 1

cap.release()

# Save detections to file
with open(detections_output_file, 'w') as f:
    f.writelines(detections)

print(f"Detections saved to {detections_output_file}")
