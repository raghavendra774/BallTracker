import cv2
import numpy as np
import os
import torch
from filterpy.kalman import KalmanFilter

# File paths
video_input_path = r"../../rawvideos/slow.mp4"
video_filename = os.path.basename(video_input_path).split('.')[0]  # Gets filename without extension
detections_output_file = f"./{video_filename}.txt"

print(f"✅ Detections will be saved as: {detections_output_file}")

# Load YOLOv5 model (Ensure the model path is correct)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../../model/best.pt')
model.conf = 0.2  # Confidence threshold
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

kf.P *= 500  # Lower initial uncertainty
kf.R = np.array([[20, 0], [0, 20]])  # Increase measurement noise (previously 10)
kf.Q = np.array([[0.3, 0, 0, 0],  
                 [0, 0.3, 0, 0],
                 [0, 0, 0.1, 0],
                 [0, 0, 0, 0.1]])  # Reduce process noise
kf.x = np.array([[0], [0], [0], [0]])  # Initial state

def apply_kalman_filter(measured_x, measured_y):
    """ Update Kalman Filter with new measurement and predict next position """
    kf.predict()
    kf.update([measured_x, measured_y])
    
    # Extract scalar values from NumPy array
    predicted_x, predicted_y = int(kf.x[0, 0]), int(kf.x[1, 0])
    
    return predicted_x, predicted_y

# Function to manually select the stump (rectangle)
stump_selected = False
stump_rect = []

def select_stump(event, x, y, flags, param):
    """Allows user to select the stump area (rectangle)."""
    global stump_selected, stump_rect

    if event == cv2.EVENT_LBUTTONDOWN and len(stump_rect) < 2:
        stump_rect.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Stumps", img)

    if len(stump_rect) == 2:
        stump_selected = True
        cv2.rectangle(img, stump_rect[0], stump_rect[1], (255, 0, 0), 2)
        cv2.imshow("Select Stumps", img)

# Extract a reference frame from the video
cap = cv2.VideoCapture(video_input_path)
ret, frame = cap.read()
if not ret:
    raise FileNotFoundError("Error: Could not extract a frame from the video.")

print("\n🚨 Click on **two points** to select the rectangular area of the stumps 🚨")
img = frame.copy()
cv2.imshow("Select Stumps", frame)
cv2.setMouseCallback("Select Stumps", select_stump)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

if len(stump_rect) != 2:
    raise ValueError("Error: You must select a single rectangular stump.")

# Compute the center of the selected stump
stump_x = (stump_rect[0][0] + stump_rect[1][0]) // 2
stump_y = (stump_rect[0][1] + stump_rect[1][1]) // 2

print(f"\n✅ Stump Center Selected: ({stump_x}, {stump_y})")

# Open video file again to process frames
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
            continue  

        x1, y1, x2, y2, conf, cls = map(float, det)  # Convert values to float
        obj_id = int(cls)  # Convert class ID to integer
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

        center_x, center_y = x + w // 2, y + h // 2  # Compute ball center

        # Compute coordinates relative to the stump center
        relative_x = center_x - stump_x
        relative_y = center_y - stump_y

        # Apply Kalman Filter for smoothing
        filtered_x, filtered_y = apply_kalman_filter(relative_x, relative_y)

        detections.append(f"{frame_id} {obj_id} {relative_x} {relative_y} {w} {h} {filtered_x} {filtered_y}\n")

    frame_id += 1

cap.release()

# Ensure directory exists
output_dir = os.path.dirname(detections_output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save detections to file
with open(detections_output_file, 'w') as f:
    f.writelines(detections)

print(f"\n✅ Detections saved to {detections_output_file}")
