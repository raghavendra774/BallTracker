import pandas as pd
import plotly.graph_objects as go
import panel as pn

# Enable Panel Extensions (needed for Jupyter & interactive use)
pn.extension()

# 📂 **Step 1: Load detections.txt**
detections_file = "./detections.txt"

# Read the detections file (Ensure it has frame_id, filtered_x, filtered_y, estimated_z)
columns = ["frame_id", "obj_id", "x", "y", "w", "h", "filtered_x", "filtered_y", "estimated_z"]
df = pd.read_csv(detections_file, sep=" ", names=columns)

# 📌 **Step 2: Sort Data by Frame ID (for smooth visualization)**
df = df.sort_values(by="frame_id")

# 📌 **Step 3: Create 3D Scatter Plot (Trajectory)**
fig = go.Figure()

# Add 3D trajectory line
fig.add_trace(go.Scatter3d(
    x=df["filtered_x"], 
    y=df["filtered_y"], 
    z=df["frame_id"] * 30, 
    mode="lines+markers",
    marker=dict(size=5, color=df["estimated_z"], colorscale="Viridis"),
    line=dict(width=4)
))

# 📌 **Step 4: Configure 3D Plot Appearance**
fig.update_layout(
    title="3D Ball Trajectory",
    scene=dict(
        xaxis_title="X (Horizontal)",
        yaxis_title="Y (Vertical)",
        zaxis_title="Z (Depth/Distance from Camera)"
    )
)

# 📌 **Step 5: Display Interactive 3D Plot**
bokeh_panel = pn.pane.Plotly(fig, width=800, height=600)

# ✅ Opens directly in browser
pn.serve(bokeh_panel, show=True)
