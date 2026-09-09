"""Plot how one contacting RoboBall bar generates torque."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch


OUT = Path(__file__).with_name("assets") / "single-rod-contact-torque.png"

INK = "#101828"
CORE = "#667085"
CORE_EDGE = "#1D2939"
SLEEVE = "#344054"
INNER = "#F2F4F7"
FOOT = "#D92D8A"
FOOT_EDGE = "#851651"
R_COLOR = "#7A5AF8"
FN_COLOR = "#067647"
FT_COLOR = "#B54708"
F_COLOR = "#D92D20"
TORQUE_COLOR = "#155EEF"
ACT_COLOR = "#6941C6"
PUSH_COLOR = "#9E165F"


fig, ax = plt.subplots(figsize=(13, 8.2), facecolor="white")
ax.set_aspect("equal")
ax.set_xlim(-0.48, 0.48)
ax.set_ylim(-0.14, 0.58)
ax.axis("off")

# Geometry in an x-z side view. The foot centre sits one foot radius above the
# ground; C is the actual contact point beneath it.
O = np.array([0.0, 0.28])
foot_centre = np.array([-0.17, 0.018])
C = np.array([-0.17, 0.0])
core_radius = 0.15
foot_radius = 0.018
u = (foot_centre - O) / np.linalg.norm(foot_centre - O)
surface = O + core_radius * u

# Ground and core.
ax.fill_between([-0.48, 0.48], -0.14, 0, color="#EAECF0", zorder=0)
ax.plot([-0.48, 0.48], [0, 0], color="#475467", lw=3.0, zorder=1)
ax.text(0.40, -0.115, "ground", fontsize=13, color="#475467", ha="center")
ax.add_patch(Circle(O, core_radius, fc=CORE, ec=CORE_EDGE, lw=2.5, zorder=2))
ax.scatter(*O, s=35, color=INK, zorder=12)
ax.text(O[0] + 0.012, O[1] + 0.010, "$O$", fontsize=15, fontweight="bold",
        color="white", zorder=13)

# One telescoping bar and its rounded foot.
bar_end = foot_centre - 0.012 * u
ax.plot([surface[0], bar_end[0]], [surface[1], bar_end[1]], color=SLEEVE,
        lw=18, solid_capstyle="round", zorder=4)
ax.plot([surface[0], bar_end[0]], [surface[1], bar_end[1]], color=INNER,
        lw=9, solid_capstyle="round", zorder=5)
ax.add_patch(Circle(foot_centre, foot_radius, fc=FOOT, ec=FOOT_EDGE,
                    lw=2.0, zorder=7))
ax.scatter(*C, s=28, color=INK, zorder=13)

# Position vector from the centre to the contact point.
ax.annotate(
    "",
    xy=C,
    xytext=O,
    arrowprops=dict(arrowstyle="-|>", color=R_COLOR, lw=3.0,
                    linestyle=(0, (4, 3)), shrinkA=7, shrinkB=6),
    zorder=10,
)
ax.annotate(
    r"position vector $\mathbf{r}_i$",
    xy=(-0.055, 0.28),
    xytext=(-0.44, 0.44),
    fontsize=13.5,
    fontweight="bold",
    color=R_COLOR,
    arrowprops=dict(arrowstyle="->", color=R_COLOR, lw=1.8),
    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=R_COLOR, alpha=0.97),
    zorder=15,
)

# Contact-force decomposition. Arrow lengths are illustrative, not a force
# scale. The dashed sides complete the vector-addition parallelogram.
# Rod angle is 58.7 deg. To produce clockwise torque (rolling forward to the right),
# the resultant ground reaction F_i must pass to the LEFT of O.
# This means F_i is steeper than r_i (angle 73.0 deg vs 58.7 deg), placing F_i
# on the LEFT of the position vector r_i with angle gamma = 14.3 deg.
Fn = np.array([0.0, 0.170])
Ft = np.array([0.052, 0.0])
F = Fn + Ft


def vector(start, delta, color, lw=3.2, zorder=14):
    ax.annotate(
        "",
        xy=start + delta,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        shrinkA=0, shrinkB=0, mutation_scale=18),
        zorder=zorder,
    )


# The actuator extends the bar along its own axis and transmits an outward push
# to the foot. Placed on the right flank so it is cleanly separated from the reaction force on the left.
norm_right = np.array([-u[1], u[0]])
act_start = surface + 0.024 * norm_right + 0.015 * u
act_delta = 0.090 * u
vector(act_start, act_delta, ACT_COLOR, lw=3.2, zorder=15)
ax.annotate(
    r"actuator force $\mathbf{F}_{\mathrm{act}}$",
    xy=act_start + 0.4 * act_delta,
    xytext=(0.04, 0.075),
    fontsize=13.5,
    fontweight="bold",
    color=ACT_COLOR,
    arrowprops=dict(arrowstyle="->", color=ACT_COLOR, lw=1.8),
    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=ACT_COLOR, alpha=0.97),
    zorder=17,
)

vector(C, Fn, FN_COLOR)
vector(C, Ft, FT_COLOR)
vector(C, F, F_COLOR, lw=3.8, zorder=16)

# Dashed line of action of F extending past the vector to the left of O
line_end = C + 2.05 * F
ax.plot([C[0] + F[0], line_end[0]], [C[1] + F[1], line_end[1]],
        color=F_COLOR, lw=1.8, ls="--", alpha=0.90, zorder=11)

# Perpendicular lever arm line d from O to line of action
u_F = F / np.linalg.norm(F)
CO = O - C
proj_len = np.dot(CO, u_F)
P_lever = C + proj_len * u_F
ax.plot([O[0], P_lever[0]], [O[1], P_lever[1]], color="#F9F5FF", lw=2.2, ls=":", zorder=13)
ax.scatter(*P_lever, s=28, color="#F9F5FF", ec=INK, lw=1.0, zorder=14)
ax.text((O[0] + P_lever[0])/2 - 0.018, (O[1] + P_lever[1])/2 + 0.008, "$d$",
        color="#F9F5FF", fontsize=14.5, fontweight="bold", zorder=15)

# Parallelogram dashed lines
ax.plot([C[0], C[0] + F[0]], [C[1] + Fn[1], C[1] + Fn[1]],
        color="#98A2B3", lw=1.4, ls="--", zorder=8)
ax.plot([C[0] + Ft[0], C[0] + Ft[0]], [C[1], C[1] + Fn[1]],
        color="#98A2B3", lw=1.4, ls="--", zorder=8)

# Newton's third-law partner: the robot pushes the ground opposite to the
# reaction force drawn above. It is shortened only to fit inside the diagram.
vector(C, -0.46 * F, PUSH_COLOR, lw=3.4, zorder=16)
ax.text(
    C[0] - 0.06,
    -0.105,
    r"robot pushes ground: $\mathbf{F}_{R\to G}=-\mathbf{F}_i$",
    color=PUSH_COLOR,
    fontsize=12.5,
    fontweight="bold",
    ha="center",
    va="top",
)

ax.text(C[0] - 0.022, C[1] + 0.105, r"normal $\mathbf{F}_{n,i}$",
        color=FN_COLOR, fontsize=13.5, fontweight="bold", ha="right", va="center")
ax.text(C[0] + Ft[0] + 0.016, C[1] + 0.020,
        r"friction $\mathbf{F}_{t,i}$", color=FT_COLOR, fontsize=13.5,
        fontweight="bold", ha="left", va="bottom")

# Resultant force callout badge
ax.annotate(
    r"resultant $\mathbf{F}_i$",
    xy=C + F,
    xytext=(-0.44, 0.28),
    fontsize=13.5,
    fontweight="bold",
    color=F_COLOR,
    ha="left",
    va="center",
    arrowprops=dict(arrowstyle="->", color=F_COLOR, lw=1.8, shrinkB=4),
    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=F_COLOR, alpha=0.97),
    zorder=17,
)
ax.annotate(
    "contact point $C$",
    xy=C,
    xytext=(-0.44, 0.040),
    fontsize=12.5,
    fontweight="bold",
    color=INK,
    arrowprops=dict(arrowstyle="->", color=INK, lw=1.6, shrinkB=4),
    bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#98A2B3"),
    zorder=15,
)

# Angle gamma annotation between r_i line (from C toward O at 58.7 deg) and F_i (at 73.0 deg)
ang_CO = np.rad2deg(np.arctan2(O[1] - C[1], O[0] - C[0]))  # 58.7 deg
ang_F = np.rad2deg(np.arctan2(F[1], F[0]))                 # 73.0 deg

arc_r = 0.088
gamma_arc = Arc(C, 2 * arc_r, 2 * arc_r, angle=0, theta1=ang_CO, theta2=ang_F,
                color="#D92D20", lw=2.6, zorder=18)
ax.add_patch(gamma_arc)

target_ang = np.deg2rad(66.5)
arc_pt = C + arc_r * np.array([np.cos(target_ang), np.sin(target_ang)])

ax.annotate(
    r"$\gamma$",
    xy=arc_pt,
    xytext=(-0.075, 0.225),
    fontsize=15.0,
    fontweight="bold",
    color="#D92D20",
    ha="center",
    va="center",
    arrowprops=dict(arrowstyle="->", color="#D92D20", lw=1.8,
                    mutation_scale=13, shrinkB=3),
    bbox=dict(boxstyle="circle,pad=0.20", fc="white", ec="#D92D20", lw=1.5, alpha=0.98),
    zorder=20,
)

# A clockwise curved arrow shows the rotational effect for the illustrated
# force direction.
rotation = FancyArrowPatch(
    (0.035, 0.445),
    (0.145, 0.315),
    connectionstyle="arc3,rad=-0.48",
    arrowstyle="Simple,tail_width=1.6,head_width=10,head_length=11",
    color=TORQUE_COLOR,
    lw=1.2,
    zorder=12,
)
ax.add_patch(rotation)
ax.text(0.165, 0.425, "resulting\nrotation", color=TORQUE_COLOR,
        fontsize=14, fontweight="bold", ha="left", va="center")

# Small world-frame axes.
axis_origin = np.array([0.37, 0.005])
vector(axis_origin, np.array([0.075, 0.0]), "#D92D20", lw=2.0)
vector(axis_origin, np.array([0.0, 0.075]), "#155EEF", lw=2.0)
ax.text(axis_origin[0] + 0.085, axis_origin[1], "$x_W$", color="#D92D20",
        fontsize=13, va="center")
ax.text(axis_origin[0], axis_origin[1] + 0.085, "$z_W$", color="#155EEF",
        fontsize=13, ha="center")

card_x = 0.325
ax.text(
    card_x,
    0.285,
    r"$\mathbf{F}_i=\mathbf{F}_{n,i}+\mathbf{F}_{t,i}$"
    "\n"
    r"$\mathbf{F}_{R\to G}=-\mathbf{F}_i$"
    "\n"
    r"$\boldsymbol{\tau}_i=\mathbf{r}_i\times\mathbf{F}_i$"
    "\n"
    r"$\|\boldsymbol{\tau}_i\| = \|\mathbf{r}_i\|\,\|\mathbf{F}_i\|\sin\gamma$",
    fontsize=15.0,
    color=INK,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.42", fc="#F9FAFB", ec=TORQUE_COLOR, lw=1.8),
)
ax.text(
    card_x,
    0.165,
    "The force line misses $O$,\nso it creates a turning effect.",
    fontsize=12.5,
    color="#344054",
    ha="center",
    va="top",
)

ax.set_title("How one contacting bar creates torque", fontsize=22,
             fontweight="bold", color=INK, pad=18)
fig.text(
    0.5,
    0.025,
    "The arrow lengths illustrate vector direction and addition; they are not drawn to a force scale.",
    ha="center",
    fontsize=12.5,
    color="#475467",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved to", OUT)
