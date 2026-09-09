"""Create the interactive 3D body/world-frame figure for the blog."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from radial_sphere.geometry import fibonacci_sphere


OUT = Path(__file__).parent / "assets" / "body-world-coordinate-frames.html"
AXIS_COLORS = ("#D92D20", "#079455", "#175CD3")


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return the body-to-world rotation Rz(yaw) Ry(pitch) Rx(roll)."""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def add_arrow(fig, origin, vector, color, label, frame_name, width=8):
    """Add a line, cone arrowhead, and large 3D label."""
    origin = np.asarray(origin, dtype=float)
    vector = np.asarray(vector, dtype=float)
    end = origin + vector
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], end[0]],
            y=[origin[1], end[1]],
            z=[origin[2], end[2]],
            mode="lines",
            line=dict(color=color, width=width),
            hovertemplate=f"{frame_name} {label}<extra></extra>",
            showlegend=False,
        )
    )
    unit = vector / np.linalg.norm(vector)
    fig.add_trace(
        go.Cone(
            x=[end[0]], y=[end[1]], z=[end[2]],
            u=[unit[0]], v=[unit[1]], w=[unit[2]],
            anchor="tip", sizemode="absolute", sizeref=4.0,
            colorscale=[[0, color], [1, color]], showscale=False,
            hoverinfo="skip",
        )
    )
    label_pos = end + 3.0 * unit
    fig.add_trace(
        go.Scatter3d(
            x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]],
            mode="text", text=[label], textfont=dict(color=color, size=20),
            hoverinfo="skip", showlegend=False,
        )
    )


fig = go.Figure()
center = np.array([6.0, 3.0, 15.5])
core_radius = 15.0  # cm
R = rotation_matrix(yaw_deg=34, pitch_deg=-18, roll_deg=24)

# Ground plane and a light world-frame grid.
grid_min, grid_max = -45.0, 48.0
for value in np.arange(-40, 51, 10):
    fig.add_trace(go.Scatter3d(
        x=[grid_min, grid_max], y=[value, value], z=[0, 0], mode="lines",
        line=dict(color="#D0D5DD", width=2), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[value, value], y=[grid_min, grid_max], z=[0, 0], mode="lines",
        line=dict(color="#D0D5DD", width=2), hoverinfo="skip", showlegend=False,
    ))

# Semi-transparent core sphere.
longitude = np.linspace(0, 2 * np.pi, 70)
latitude = np.linspace(0, np.pi, 38)
SX = center[0] + core_radius * np.outer(np.cos(longitude), np.sin(latitude))
SY = center[1] + core_radius * np.outer(np.sin(longitude), np.sin(latitude))
SZ = center[2] + core_radius * np.outer(np.ones_like(longitude), np.cos(latitude))
fig.add_trace(go.Surface(
    x=SX, y=SY, z=SZ,
    surfacecolor=np.ones_like(SX),
    colorscale=[[0, "#98A2B3"], [1, "#98A2B3"]],
    opacity=0.46, showscale=False, hoverinfo="skip", name="rigid core",
))

# All 60 bars rotate with the core because their directions are fixed in B.
body_dirs = fibonacci_sphere(60)
world_dirs = body_dirs @ R.T
rod_x, rod_y, rod_z = [], [], []
tip_x, tip_y, tip_z = [], [], []
for direction in world_dirs:
    base = center + core_radius * direction
    tip = center + (core_radius + 7.0) * direction
    rod_x.extend([base[0], tip[0], None])
    rod_y.extend([base[1], tip[1], None])
    rod_z.extend([base[2], tip[2], None])
    tip_x.append(tip[0])
    tip_y.append(tip[1])
    tip_z.append(tip[2])
fig.add_trace(go.Scatter3d(
    x=rod_x, y=rod_y, z=rod_z, mode="lines",
    line=dict(color="#667085", width=4), opacity=0.72,
    name="body-fixed bars", hoverinfo="skip",
))
fig.add_trace(go.Scatter3d(
    x=tip_x, y=tip_y, z=tip_z, mode="markers",
    marker=dict(size=3.5, color="#344054"), opacity=0.85,
    name="bar feet", hoverinfo="skip",
))

# W is fixed to the ground. B is attached to the ball centre.
world_origin = np.array([-35.0, -27.0, 0.4])
world_axes = np.eye(3)
body_axes_world = R  # columns are x_B, y_B, z_B expressed in W
for axis_index, (color, axis_name) in enumerate(zip(AXIS_COLORS, ("x", "y", "z"))):
    add_arrow(
        fig, world_origin, 20.0 * world_axes[:, axis_index], color,
        f"{axis_name}<sub>W</sub>", "world frame", width=9,
    )
    add_arrow(
        fig, center, 25.0 * body_axes_world[:, axis_index], color,
        f"{axis_name}<sub>B</sub>", "body frame", width=10,
    )

# Highlight an actual bar that is easy to see from the initial camera angle.
target_world = np.array([0.65, -0.50, 0.57])
target_world /= np.linalg.norm(target_world)
selected = int(np.argmax(world_dirs @ target_world))
u_body = body_dirs[selected]
u_world = world_dirs[selected]
highlight_end = center + 29.0 * u_world
fig.add_trace(go.Scatter3d(
    x=[center[0], highlight_end[0]],
    y=[center[1], highlight_end[1]],
    z=[center[2], highlight_end[2]],
    mode="lines", line=dict(color="#C11574", width=12),
    name=f"highlighted bar {selected}",
    hovertemplate=(
        f"bar {selected}<br>"
        f"u<sub>i</sub><sup>B</sup> = [{u_body[0]:+.2f}, {u_body[1]:+.2f}, {u_body[2]:+.2f}]<br>"
        f"u<sub>i</sub><sup>W</sup> = [{u_world[0]:+.2f}, {u_world[1]:+.2f}, {u_world[2]:+.2f}]"
        "<extra></extra>"
    ),
))
fig.add_trace(go.Cone(
    x=[highlight_end[0]], y=[highlight_end[1]], z=[highlight_end[2]],
    u=[u_world[0]], v=[u_world[1]], w=[u_world[2]],
    anchor="tip", sizemode="absolute", sizeref=4.8,
    colorscale=[[0, "#C11574"], [1, "#C11574"]], showscale=False,
    hoverinfo="skip",
))

# Resolve the highlighted direction into world-frame x, y, and z components.
# Here d^W = x_W and d_perp^W = y_W, so these are also the task projections.
component_scale = 29.0
point_x = center + component_scale * np.array([u_world[0], 0.0, 0.0])
point_xy = center + component_scale * np.array([u_world[0], u_world[1], 0.0])
projection_segments = [
    (center, point_x, AXIS_COLORS[0],
     f"u<sub>i</sub><sup>∥</sup> = u<sub>i,x</sub><sup>W</sup> = {u_world[0]:+.2f}"),
    (point_x, point_xy, AXIS_COLORS[1],
     f"u<sub>i</sub><sup>⊥</sup> = u<sub>i,y</sub><sup>W</sup> = {u_world[1]:+.2f}"),
    (point_xy, highlight_end, AXIS_COLORS[2],
     f"u<sub>i</sub><sup>z</sup> = u<sub>i,z</sub><sup>W</sup> = {u_world[2]:+.2f}"),
]
for segment_start, segment_end, color, label in projection_segments:
    midpoint = 0.5 * (segment_start + segment_end)
    fig.add_trace(go.Scatter3d(
        x=[segment_start[0], segment_end[0]],
        y=[segment_start[1], segment_end[1]],
        z=[segment_start[2], segment_end[2]],
        mode="lines", line=dict(color=color, width=7, dash="dash"),
        hovertemplate=label + "<extra></extra>", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[midpoint[0]], y=[midpoint[1]], z=[midpoint[2] + 2.2],
        mode="text", text=[label], textfont=dict(size=16, color=color),
        hoverinfo="skip", showlegend=False,
    ))
fig.add_trace(go.Scatter3d(
    x=[point_x[0], point_xy[0]], y=[point_x[1], point_xy[1]],
    z=[point_x[2], point_xy[2]], mode="markers",
    marker=dict(size=4.5, color=[AXIS_COLORS[0], AXIS_COLORS[1]],
                symbol="circle-open", line=dict(width=2)),
    hoverinfo="skip", showlegend=False,
))
fig.add_trace(go.Scatter3d(
    x=[highlight_end[0] + 2.0], y=[highlight_end[1] - 1.0], z=[highlight_end[2] + 2.0],
    mode="text", text=["same physical bar"],
    textfont=dict(size=18, color="#C11574"), hoverinfo="skip", showlegend=False,
))

fig.update_layout(
    height=720,
    autosize=True,
    title=dict(
        text="Rotating a bar from body coordinates into world coordinates",
        x=0.5, xanchor="center", font=dict(size=24, color="#101828"),
    ),
    scene=dict(
        domain=dict(x=[0.0, 0.70], y=[0.04, 1.0]),
        xaxis=dict(visible=False, range=[-48, 52]),
        yaxis=dict(visible=False, range=[-48, 52]),
        zaxis=dict(visible=False, range=[0, 50]),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=-2.05, z=1.18), center=dict(x=0, y=0, z=-0.08)),
        bgcolor="white",
    ),
    margin=dict(l=0, r=10, t=85, b=30),
    paper_bgcolor="white",
    legend=dict(
        x=0.01, y=0.01, orientation="h", bgcolor="rgba(255,255,255,0.88)",
        font=dict(size=13),
    ),
    annotations=[
        dict(
            x=0.01, y=0.98, xref="paper", yref="paper", showarrow=False,
            text="<b>World frame W</b><br>fixed to the ground",
            xanchor="left", yanchor="top", align="left",
            font=dict(size=17, color="#344054"),
            bgcolor="rgba(255,255,255,0.90)", bordercolor="#D0D5DD", borderpad=6,
        ),
        dict(
            x=0.69, y=0.98, xref="paper", yref="paper", showarrow=False,
            text="<b>Body frame B</b><br>attached to the core",
            xanchor="right", yanchor="top", align="right",
            font=dict(size=17, color="#344054"),
            bgcolor="rgba(255,255,255,0.90)", bordercolor="#D0D5DD", borderpad=6,
        ),
        dict(
            x=0.725, y=0.91, xref="paper", yref="paper", showarrow=False,
            xanchor="left", yanchor="top", align="left",
            text=(
                "<b>1. Body → world</b><br><br>"
                f"u<sub>i</sub><sup>B</sup> = [{u_body[0]:+.2f}, {u_body[1]:+.2f}, {u_body[2]:+.2f}]<sup>T</sup><br>"
                "u<sub>i</sub><sup>W</sup> = R<sub>WB</sub>(q) u<sub>i</sub><sup>B</sup><br>"
                f"u<sub>i</sub><sup>W</sup> = [{u_world[0]:+.2f}, {u_world[1]:+.2f}, {u_world[2]:+.2f}]<sup>T</sup><br><br>"
                "R<sub>WB</sub> maps coordinates from B to W."
            ),
            font=dict(size=16, color="#101828"),
            bgcolor="#F9FAFB", bordercolor="#98A2B3", borderpad=10,
        ),
        dict(
            x=0.725, y=0.52, xref="paper", yref="paper", showarrow=False,
            xanchor="left", yanchor="top", align="left",
            text=(
                "<b>2. Project in W</b><br><br>"
                "d<sup>W</sup> = [d<sub>x</sub>, d<sub>y</sub>, 0]<sup>T</sup><br>"
                "d<sub>⊥</sub><sup>W</sup> = [-d<sub>y</sub>, d<sub>x</sub>, 0]<sup>T</sup><br><br>"
                "u<sub>i</sub><sup>∥</sup> = (u<sub>i</sub><sup>W</sup>)<sup>T</sup>d<sup>W</sup><br>"
                "u<sub>i</sub><sup>⊥</sup> = (u<sub>i</sub><sup>W</sup>)<sup>T</sup>d<sub>⊥</sub><sup>W</sup><br>"
                "u<sub>i</sub><sup>z</sup> = (u<sub>i</sub><sup>W</sup>)<sup>T</sup>z<sup>W</sup>"
            ),
            font=dict(size=16, color="#101828"),
            bgcolor="#F9FAFB", bordercolor="#98A2B3", borderpad=10,
        ),
        dict(
            x=0.725, y=0.13, xref="paper", yref="paper", showarrow=False,
            xanchor="left", yanchor="top", align="left",
            text=(
                "<b>Shown here:</b> d<sup>W</sup> = x<sub>W</sub> and "
                "d<sub>⊥</sub><sup>W</sup> = y<sub>W</sub>.<br>"
                "The dashed red, green, and blue segments are the three projections."
            ),
            font=dict(size=15, color="#344054"),
            bgcolor="white", bordercolor="#D0D5DD", borderpad=8,
        ),
    ],
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(
    OUT,
    include_plotlyjs="directory",
    full_html=True,
    config={"displaylogo": False, "responsive": True, "scrollZoom": True},
)
print(OUT)
