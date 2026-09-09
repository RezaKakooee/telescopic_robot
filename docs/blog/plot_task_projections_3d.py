"""Create an interactive 3D explanation of RoboBall task projections."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from radial_sphere.geometry import fibonacci_sphere


OUT = Path(__file__).parent / "assets" / "task-projections-3d.html"
INK = "#101828"
GUIDE = "#667085"
FORWARD = "#079455"
SIDEWAYS = "#175CD3"
VERTICAL = "#7F56D9"
BAR = "#C11574"


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def add_arrow(fig, start, vector, color, label, width=7, label_offset=None):
    start = np.asarray(start, dtype=float)
    vector = np.asarray(vector, dtype=float)
    end = start + vector
    unit = vector / np.linalg.norm(vector)
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode="lines", line=dict(color=color, width=width),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Cone(
        x=[end[0]], y=[end[1]], z=[end[2]],
        u=[unit[0]], v=[unit[1]], w=[unit[2]],
        anchor="tip", sizemode="absolute", sizeref=0.13,
        colorscale=[[0, color], [1, color]], showscale=False, hoverinfo="skip",
    ))
    if label_offset is None:
        label_offset = 0.13 * unit
    label_pos = end + np.asarray(label_offset)
    fig.add_trace(go.Scatter3d(
        x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]],
        mode="text", text=[label], textfont=dict(size=18, color=color),
        hoverinfo="skip", showlegend=False,
    ))


def add_component(fig, start, end, color, label, label_offset):
    start = np.asarray(start)
    end = np.asarray(end)
    midpoint = 0.5 * (start + end) + np.asarray(label_offset)
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode="lines", line=dict(color=color, width=8, dash="dash"),
        hovertemplate=label + "<extra></extra>", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[midpoint[0]], y=[midpoint[1]], z=[midpoint[2]],
        mode="text", text=[label], textfont=dict(size=17, color=color),
        hoverinfo="skip", showlegend=False,
    ))


fig = go.Figure()
origin = np.zeros(3)
radius = 1.0

# Choose a non-axis-aligned horizontal task direction so dx and dy are visible.
task_angle = np.deg2rad(32.0)
d_hat = np.array([np.cos(task_angle), np.sin(task_angle), 0.0])
d_perp = np.array([-np.sin(task_angle), np.cos(task_angle), 0.0])
z_hat = np.array([0.0, 0.0, 1.0])

# Rotate the 60 body-fixed directions into W for the gray reference bars.
R_WB = rotation_matrix(yaw_deg=34, pitch_deg=-18, roll_deg=24)
body_dirs = fibonacci_sphere(60)
world_dirs = body_dirs @ R_WB.T

# Construct the exact illustrative direction used in the tutorial.
u_world = -0.70 * d_hat - 0.20 * d_perp - np.sqrt(0.47) * z_hat
u_world /= np.linalg.norm(u_world)
u_body = R_WB.T @ u_world
u_parallel = float(u_world @ d_hat)
u_sideways = float(u_world @ d_perp)
u_vertical = float(u_world @ z_hat)

# Ground plane and world x-y grid.
for value in np.arange(-1.6, 1.61, 0.4):
    fig.add_trace(go.Scatter3d(
        x=[-1.6, 1.6], y=[value, value], z=[-1.04, -1.04], mode="lines",
        line=dict(color="#D0D5DD", width=2), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[value, value], y=[-1.6, 1.6], z=[-1.04, -1.04], mode="lines",
        line=dict(color="#D0D5DD", width=2), hoverinfo="skip", showlegend=False,
    ))

# Transparent core and all body-fixed bars in their current world orientation.
longitude = np.linspace(0, 2 * np.pi, 64)
latitude = np.linspace(0, np.pi, 34)
SX = radius * np.outer(np.cos(longitude), np.sin(latitude))
SY = radius * np.outer(np.sin(longitude), np.sin(latitude))
SZ = radius * np.outer(np.ones_like(longitude), np.cos(latitude))
fig.add_trace(go.Surface(
    x=SX, y=SY, z=SZ, surfacecolor=np.ones_like(SX),
    colorscale=[[0, "#98A2B3"], [1, "#98A2B3"]],
    opacity=0.28, showscale=False, hoverinfo="skip", name="RoboBall core",
))
rod_x, rod_y, rod_z = [], [], []
tip_x, tip_y, tip_z = [], [], []
for direction in world_dirs:
    base = radius * direction
    tip = 1.14 * direction
    rod_x.extend([base[0], tip[0], None])
    rod_y.extend([base[1], tip[1], None])
    rod_z.extend([base[2], tip[2], None])
    tip_x.append(tip[0])
    tip_y.append(tip[1])
    tip_z.append(tip[2])
fig.add_trace(go.Scatter3d(
    x=rod_x, y=rod_y, z=rod_z, mode="lines",
    line=dict(color="#98A2B3", width=3), opacity=0.45,
    name="60 rotated bars", hoverinfo="skip",
))
fig.add_trace(go.Scatter3d(
    x=tip_x, y=tip_y, z=tip_z, mode="markers",
    marker=dict(size=2.7, color="#667085"), opacity=0.60,
    name="bar feet", hoverinfo="skip",
))

# Thin world axes show what dx and dy refer to.
add_arrow(fig, origin, np.array([1.38, 0, 0]), GUIDE,
          "x<sub>W</sub>", width=4, label_offset=[0.10, 0, 0])
add_arrow(fig, origin, np.array([0, 1.38, 0]), GUIDE,
          "y<sub>W</sub>", width=4, label_offset=[0, 0.10, 0])

# Task basis: forward, sideways, and vertical.
add_arrow(
    fig, origin, 1.52 * d_hat, FORWARD,
    "d&#770;<sup>W</sup> = [d<sub>x</sub>, d<sub>y</sub>, 0]",
    width=9, label_offset=[0.14, 0.04, 0.05],
)
add_arrow(
    fig, origin, 1.52 * d_perp, SIDEWAYS,
    "d&#770;<sub>⊥</sub><sup>W</sup> = [-d<sub>y</sub>, d<sub>x</sub>, 0]",
    width=9, label_offset=[-0.18, 0.02, 0.05],
)
add_arrow(
    fig, origin, 1.52 * z_hat, VERTICAL,
    "z&#770;<sup>W</sup>", width=9, label_offset=[0.05, 0.02, 0.12],
)

# The magenta arrow is one physical bar after body-to-world rotation.
bar_length = 1.48
bar_end = bar_length * u_world
add_arrow(
    fig, origin, bar_end, BAR,
    "u&#770;<sub>i</sub><sup>W</sup>", width=12,
    label_offset=0.13 * u_world + np.array([0.02, 0.02, -0.05]),
)

# Head-to-tail construction of the three task-frame projections.
point_parallel = bar_length * u_parallel * d_hat
point_horizontal = point_parallel + bar_length * u_sideways * d_perp
add_component(
    fig, origin, point_parallel, FORWARD,
    f"u<sub>i</sub><sup>∥</sup> = {u_parallel:+.2f}", [0.00, 0.00, 0.10],
)
add_component(
    fig, point_parallel, point_horizontal, SIDEWAYS,
    f"u<sub>i</sub><sup>⊥</sup> = {u_sideways:+.2f}", [0.00, 0.00, 0.10],
)
add_component(
    fig, point_horizontal, bar_end, VERTICAL,
    f"u<sub>i</sub><sup>z</sup> = {u_vertical:+.2f}", [0.08, 0.04, 0.00],
)
fig.add_trace(go.Scatter3d(
    x=[point_parallel[0], point_horizontal[0], bar_end[0]],
    y=[point_parallel[1], point_horizontal[1], bar_end[1]],
    z=[point_parallel[2], point_horizontal[2], bar_end[2]],
    mode="markers", marker=dict(size=5, color=[FORWARD, SIDEWAYS, VERTICAL],
                                symbol="circle-open", line=dict(width=2)),
    hoverinfo="skip", showlegend=False,
))

fig.update_layout(
    height=720,
    autosize=True,
    title=dict(
        text="Project one world-frame bar direction onto the task axes",
        x=0.5, xanchor="center", font=dict(size=24, color=INK),
    ),
    scene=dict(
        domain=dict(x=[0.0, 0.69], y=[0.03, 1.0]),
        xaxis=dict(visible=False, range=[-1.75, 1.75]),
        yaxis=dict(visible=False, range=[-1.75, 1.75]),
        zaxis=dict(visible=False, range=[-1.25, 1.75]),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=-2.05, z=1.30),
                    center=dict(x=0, y=0, z=-0.05)),
        bgcolor="white",
    ),
    margin=dict(l=0, r=10, t=80, b=25),
    paper_bgcolor="white",
    legend=dict(
        x=0.01, y=0.01, orientation="h", bgcolor="rgba(255,255,255,0.88)",
        font=dict(size=13),
    ),
    annotations=[
        dict(
            x=0.715, y=0.91, xref="paper", yref="paper", showarrow=False,
            xanchor="left", yanchor="top", align="left",
            text=(
                "<b>1. Rotate B → W</b><br><br>"
                "u&#770;<sub>i</sub><sup>W</sup> = R<sub>WB</sub>(q) "
                "u&#770;<sub>i</sub><sup>B</sup><br><br>"
                f"u&#770;<sub>i</sub><sup>B</sup> = "
                f"[{u_body[0]:+.2f}, {u_body[1]:+.2f}, {u_body[2]:+.2f}]<sup>T</sup><br>"
                f"u&#770;<sub>i</sub><sup>W</sup> = "
                f"[{u_world[0]:+.2f}, {u_world[1]:+.2f}, {u_world[2]:+.2f}]<sup>T</sup>"
            ),
            font=dict(size=16, color=INK),
            bgcolor="#F9FAFB", bordercolor="#98A2B3", borderpad=10,
        ),
        dict(
            x=0.715, y=0.54, xref="paper", yref="paper", showarrow=False,
            xanchor="left", yanchor="top", align="left",
            text=(
                "<b>2. Project in W</b><br><br>"
                "u<sub>i</sub><sup>∥</sup> = "
                "(u&#770;<sub>i</sub><sup>W</sup>)<sup>T</sup>d&#770;<sup>W</sup><br>"
                "= u<sub>i,x</sub><sup>W</sup>d<sub>x</sub> + "
                "u<sub>i,y</sub><sup>W</sup>d<sub>y</sub><br><br>"
                "u<sub>i</sub><sup>⊥</sup> = "
                "(u&#770;<sub>i</sub><sup>W</sup>)<sup>T</sup>d&#770;<sub>⊥</sub><sup>W</sup><br>"
                "= u<sub>i,x</sub><sup>W</sup>(-d<sub>y</sub>) + "
                "u<sub>i,y</sub><sup>W</sup>d<sub>x</sub><br><br>"
                "u<sub>i</sub><sup>z</sup> = "
                "(u&#770;<sub>i</sub><sup>W</sup>)<sup>T</sup>z&#770;<sup>W</sup> "
                "= u<sub>i,z</sub><sup>W</sup>"
            ),
            font=dict(size=16, color=INK),
            bgcolor="#F9FAFB", bordercolor="#98A2B3", borderpad=10,
        ),
        dict(
            x=0.715, y=0.10, xref="paper", yref="paper", showarrow=False,
            xanchor="left", yanchor="top", align="left",
            text=(
                f"<b>This bar:</b>  u<sub>i</sub><sup>∥</sup>={u_parallel:+.2f}, "
                f"u<sub>i</sub><sup>⊥</sup>={u_sideways:+.2f}, "
                f"u<sub>i</sub><sup>z</sup>={u_vertical:+.2f}<br>"
                "Negative ∥ means behind; negative z means below."
            ),
            font=dict(size=15, color=INK),
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
