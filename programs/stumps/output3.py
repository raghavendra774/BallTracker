import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# File paths
detections_file = r"./slow.txt"  # Input detections file
video_input_path = r"../../rawvideos/slow.mp4"  # Input video
video_filename = os.path.basename(video_input_path).split('.')[0]  # Extract filename without extension

output_image_path = f"./{video_filename}.jpg"  # Output trajectory image
output_video_path = f"./{video_filename}_trajectory.mp4"  # Output trajectory video

# Function to manually select the stump (rectangle)
stump_selected = False
stump_rect = []

def select_stump(event, x, y, flags, param):
    """Allows user to select the stump area (rectangle)."""
    global stump_selected, stump_rect

    if event == cv2.EVENT_LBUTTONDOWN and len(stump_rect) < 2:
        stump_rect.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Stumps", image)

    if len(stump_rect) == 2:
        stump_selected = True
        cv2.rectangle(image, stump_rect[0], stump_rect[1], (255, 0, 0), 2)
        cv2.imshow("Select Stumps", image)

# Extract a reference frame from the video
cap = cv2.VideoCapture(video_input_path)
ret, frame = cap.read()
if not ret:
    raise FileNotFoundError("Error: Could not extract a frame from the video.")

print("\n🚨 Click on **two points** to select the rectangular area of the stumps 🚨")
image = frame.copy()
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

# Read detections.txt
if not os.path.exists(detections_file):
    raise FileNotFoundError(f"Detections file not found: {detections_file}")

object_trajectories = {}

with open(detections_file, 'r') as f:
    detection_data = f.readlines()

# Extract trajectory points (using Kalman-filtered values)
for line in detection_data:
    frame_id, obj_id, relative_x, relative_y, w, h, filtered_x, filtered_y = map(int, line.strip().split())

    # Convert Kalman filtered coordinates to absolute image coordinates based on the stump
    mapped_x = stump_x + filtered_x  # Use Kalman-filtered X
    mapped_y = stump_y + filtered_y  # Use Kalman-filtered Y

    if obj_id not in object_trajectories:
        object_trajectories[obj_id] = []
    object_trajectories[obj_id].append((frame_id, mapped_x, mapped_y))

# ======================= Process Video ==========================
print("\n🎥 Generating video with smooth moving trajectory...")

cap = cv2.VideoCapture(video_input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
image_with_trajectory = frame.copy()  # To store final trajectory image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    for obj_id, points in object_trajectories.items():
        visible_points = [p for p in points if p[0] <= frame_count]  # Show only points up to this frame

        if len(visible_points) > 1:
            for i in range(1, len(visible_points)):
                cv2.line(frame, (visible_points[i - 1][1], visible_points[i - 1][2]),
                         (visible_points[i][1], visible_points[i][2]), (0, 255, 0), 3)  # Green trajectory

                cv2.line(image_with_trajectory, (visible_points[i - 1][1], visible_points[i - 1][2]),
                         (visible_points[i][1], visible_points[i][2]), (0, 255, 0), 3)  # Save on final image

            cv2.circle(frame, (visible_points[-1][1], visible_points[-1][2]), 7, (0, 0, 255), -1)  # Red ball
            cv2.circle(image_with_trajectory, (visible_points[-1][1], visible_points[-1][2]), 7, (0, 0, 255), -1)  # Save on final image

    out.write(frame)
    frame_count += 1

cap.release()
out.release()

# ===================== Save Final Trajectory Image =====================
cv2.imwrite(output_image_path, image_with_trajectory)

print(f"✅ Processed video saved at {output_video_path}")
print(f"✅ Ball trajectory image saved at {output_image_path}")

# Display the final image
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(image_with_trajectory, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Final Ball Trajectory Image")
plt.show()
