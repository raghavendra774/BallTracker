
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Initialize lists to store x, y, z coordinates
# x = []
# y = []
# z = []

# # Read the coordinates from the .txt file
# with open('detections.txt', 'r') as file:
#     for line in file:
        
#         values = line.split()
#         if len(values) == 9:  
#             x.append(float(values[6]))
#             y.append(float(values[7]))
#             z.append(float(values[0]))


# x = np.array(x)
# y = np.array(y)
# z = np.array(z)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# ax.scatter(x, y, z, c='r', marker='o')  


# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')


# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Initialize lists to store x, y, z coordinates
x = []
y = []
z = []

# Read the coordinates from the .txt file
with open('detections.txt', 'r') as file:
    for line in file:
        # Split each line by spaces and convert the values to float
        values = line.split()
        if len(values) == 9:  # Ensure there are 3 values per line (x, y, z)
            x.append(float(values[6]))
            y.append(float(values[7]))
            z.append(float(values[0]))



# Convert lists to numpy arrays for convenience
# x = np.array(x)
# y = np.array(y)
# z = np.array(z)

# streach_z = z*200

# # Create a figure and an axes object for 3D plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the points as a line in 3D space
# ax.plot(x, y, streach_z, c='b')  # Line plot connecting the points
# ax.set_zlim([min(z)-10, max(z)+10])
# # Set labels for the axes
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# # Show the plot
# plt.show()

# fig = go.Figure()

# fig.add_trace(go.Scatter3d(
#      z=z, x=x, y=y,
#     mode='lines',
#     marker=dict(size=5, color='red'),
   
# ))

# # Set aspect ratio relative to the data ranges
# fig.update_layout(
#     scene=dict(
#         aspectmode="manual",
#         aspectratio=dict(x=0.5, y=0.5, z=2)  # Adjust the relative scaling of axes
#     )
# )

# fig.show()

fig = go.Figure()

# Plot the points
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers+lines',  # Show both markers and lines
    marker=dict(size=5, color='red', opacity=0.8),  # Customize marker
    line=dict(color='blue', width=3)  # Customize line between points
))

# Set aspect ratio and ensure all axes are shown correctly
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[min(x) - 1, max(x) + 1]),  # Adjust range for better visibility
        yaxis=dict(range=[min(y) - 1, max(y) + 1]),  # Adjust range for better visibility
        zaxis=dict(range=[min(z) - 1, max(z) + 1]),  # Adjust range for better visibility
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)  # Equal scaling for all axes to ensure proper orientation
    ),
    title="3D Scatter Plot with Proper Aspect Ratios",
    scene_camera=dict(
        eye=dict(x=1.5, y=1.5, z=3)  # Adjust the view from different angles
    )
)

# Show the plot
fig.show()
