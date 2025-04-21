import cv2
import numpy as np
import os

# File paths
video_input_path = r"../rawvideos/v1.mp4"  # Input video
video_output_path = r"../results/v1.mp4"  # Output video
detections_file = r"./v1.txt"

# Ensure detections file exists
if not os.path.exists(detections_file):
    raise FileNotFoundError("Detections file is missing.")

# Open video
cap = cv2.VideoCapture(video_input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'XVID' if needed
out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

# List to store ball trajectory
ball_trajectory = []
last_detected_frame = -1  # Track last detected frame

# Read detections file
with open(detections_file, 'r') as f:
    detection_data = f.readlines()

# Organize detections by frame number
frame_detections = {}
for line in detection_data:
    parts = line.strip().split()
    if len(parts) != 9:
        print(f"Skipping malformed line: {line.strip()}")
        continue
    
    frame_num, center_x, center_y, filtered_x, filtered_y, conf, final_x, final_y, decision = map(float, parts)

    
    if final_x != 0 or final_y != 0:  # Ignore frames where final_x and final_y are both 0
        frame_detections[frame_num] = []

        frame_detections[frame_num].append((final_x, final_y))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Check if the current frame has detections
    if frame_count in frame_detections:
        for final_x, final_y in frame_detections[frame_count]:
            if last_detected_frame != -1 and (frame_count - last_detected_frame) > 15:
                # If more than 15 frames have passed since the last detection, reset trajectory
                ball_trajectory = []
            
            ball_trajectory.append((int(final_x), int(final_y)))  # Store integer coordinates
            last_detected_frame = frame_count  # Update last detected frame

        # Ensure we have at least two points to draw a trajectory
        if len(ball_trajectory) > 1:
            for i in range(1, len(ball_trajectory)):
                pt1 = ball_trajectory[i - 1]
                pt2 = ball_trajectory[i]
                if pt1 != (0, 0) and pt2 != (0, 0):  # Ignore invalid points
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green trajectory
            
            # Draw the most recent position as a red circle
            last_x, last_y = ball_trajectory[-1]
            if last_x != 0 and last_y != 0:
                cv2.circle(frame, (last_x, last_y), 7, (0, 0, 255), -1)  # Red ball
    
    out.write(frame)
    frame_count += 1

cap.release()
out.release()

print(f"Processed video saved at {video_output_path}")
