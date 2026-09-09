"""Visualize how task-frame projections select RoboBall's driving rods."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge

from radial_sphere.geometry import fibonacci_sphere


OUT = Path(__file__).with_name("assets") / "rod-task-projections.png"
INK = "#101828"
CORE = "#D0D5DD"
CORE_EDGE = "#344054"
FORWARD = "#067647"
LATERAL = "#155EEF"
VERTICAL = "#7F56D9"
ROD = "#D92D20"
GUIDE = "#667085"


def setup_robot_panel(ax, title):
    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.65)
    ax.set_ylim(-1.55, 1.55)
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", color=INK, pad=12)
    ax.add_patch(Circle((0, 0), 1.0, facecolor=CORE, edgecolor=CORE_EDGE, lw=2))
    ax.scatter([0], [0], s=32, color=INK, zorder=8)
    ax.text(-0.08, 0.08, r"$O$", color=INK, fontsize=20, fontweight="bold")


fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), facecolor="white")
fig.subplots_adjust(left=0.035, right=0.98, top=0.82, bottom=0.16, wspace=0.25)
fig.suptitle(
    r"Example: $\hat{\mathbf{d}}=[1,0]$ and "
    r"$\mathbf{u}_i^W\approx[-0.70,-0.20,-0.69]$",
    fontsize=18,
    fontweight="bold",
    color=INK,
    y=0.96,
)

# Top view: longitudinal and lateral projections.
ax = axes[0]
setup_robot_panel(ax, "1. Project onto the horizontal task axes")
ax.annotate("", xy=(1.48, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=FORWARD, lw=3))
ax.annotate(r"$\hat{\mathbf{d}}=[d_x,d_y]$" "\n" r"$=[1,0]$",
            xy=(1.32, 0), xytext=(1.47, 1.02),
            ha="right", color=FORWARD, fontsize=20, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=FORWARD, lw=1.8,
                            connectionstyle="arc3,rad=0.18"))
ax.annotate("", xy=(0, 1.40), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=LATERAL, lw=2.6))
ax.annotate(r"$\hat{\mathbf{d}}_{\perp}=[-d_y,d_x]$" "\n" r"$=[0,1]$",
            xy=(0, 1.28), xytext=(-1.48, 1.06), ha="left", va="bottom",
            color=LATERAL, fontsize=18, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=LATERAL, lw=1.8,
                            connectionstyle="arc3,rad=-0.18"))
ax.text(-1.28, 0.10, "REAR", ha="center", color=GUIDE, fontsize=18, fontweight="bold")
ax.text(1.27, -0.20, "FRONT", ha="center", color=GUIDE, fontsize=18, fontweight="bold")

tip_xy = np.array([-0.70, -0.20])
ax.annotate("", xy=tip_xy, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=ROD, lw=3), zorder=9)
ax.plot([tip_xy[0], tip_xy[0]], [0, tip_xy[1]], ls="--", color=GUIDE, lw=1.7)
ax.annotate("", xy=(tip_xy[0], 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="<->", color=FORWARD, lw=2.2))
ax.annotate(r"$u_i^{\parallel}=-0.70$", xy=(-0.35, 0), xytext=(-1.48, 0.58),
            ha="left", color=FORWARD, fontsize=20, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=FORWARD),
            arrowprops=dict(arrowstyle="->", color=FORWARD, lw=1.8,
                            connectionstyle="arc3,rad=-0.14"))
ax.annotate("", xy=tip_xy, xytext=(tip_xy[0], 0),
            arrowprops=dict(arrowstyle="<->", color=LATERAL, lw=2.2))
ax.annotate(r"$u_i^{\perp}=-0.20$", xy=(-0.70, -0.10), xytext=(-1.48, -1.08),
            ha="left", color=LATERAL, fontsize=20, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=LATERAL),
            arrowprops=dict(arrowstyle="->", color=LATERAL, lw=1.8,
                            connectionstyle="arc3,rad=0.15"))
ax.annotate(r"$(u_{i,x}^W,u_{i,y}^W)$", xy=(-0.35, -0.10), xytext=(0.28, -1.36),
            ha="center", color=ROD, fontsize=18, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ROD, lw=1.8,
                            connectionstyle="arc3,rad=-0.18"))

# Side view: vertical projection.
ax = axes[1]
setup_robot_panel(ax, "2. Read the vertical component")
ax.add_patch(Wedge((0, 0), 1.0, 180, 270, facecolor="#FDB022", alpha=0.24, zorder=2))
ax.annotate("", xy=(1.48, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=FORWARD, lw=3))
ax.annotate("", xy=(0, 1.40), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=VERTICAL, lw=2.6))
ax.text(0.10, 1.36, r"world $z$", color=VERTICAL, fontsize=20, fontweight="bold")
ax.text(-1.28, 0.10, "REAR", ha="center", color=GUIDE, fontsize=18, fontweight="bold")
ax.text(1.27, -0.20, "FRONT", ha="center", color=GUIDE, fontsize=18, fontweight="bold")

tip_xz = np.array([-0.70, -0.69])
ax.annotate("", xy=tip_xz, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=ROD, lw=3), zorder=9)
ax.plot([tip_xz[0], tip_xz[0]], [0, tip_xz[1]], ls="--", color=GUIDE, lw=1.7)
ax.annotate("", xy=tip_xz, xytext=(tip_xz[0], 0),
            arrowprops=dict(arrowstyle="<->", color=VERTICAL, lw=2.2))
ax.annotate(r"$u_i^z=-0.69$", xy=(-0.70, -0.35), xytext=(-1.48, -1.08),
            ha="left", color=VERTICAL, fontsize=20, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=VERTICAL),
            arrowprops=dict(arrowstyle="->", color=VERTICAL, lw=1.8,
                            connectionstyle="arc3,rad=-0.15"))
ax.annotate("rear + below\ngets a high\ndrive score", xy=(-0.42, -0.48),
            xytext=(0.52, -1.07), ha="center", va="top", color="#93370D",
            fontsize=18, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#93370D", lw=1.8,
                            connectionstyle="arc3,rad=0.18"))

# Exact score for the 60 Fibonacci directions at the identity orientation.
ax = axes[2]
dirs = fibonacci_sphere(60)
u_long = dirs[:, 0]
u_lat = dirs[:, 1]
u_z = dirs[:, 2]
rear = np.clip((-u_long - 0.10) / 0.90, 0.0, 1.0)
down = np.clip(1.0 - np.abs(u_z + 0.35) / 0.85, 0.0, 1.0)
centre = np.clip(1.0 - 1.8 * u_lat**2, 0.0, 1.0)
wave = np.clip(rear**1.1 * down * 1.6 * centre, 0.0, 1.0)
wave[u_long > -0.05] = 0.0
wave[u_z > 0.10] = 0.0

ax.set_aspect("equal")
ax.set_xlim(-1.08, 1.08)
ax.set_ylim(-1.08, 1.08)
ax.set_title("3. Apply the calculation to all 60 bars", fontsize=15,
             fontweight="bold", color=INK, pad=12)
ax.add_patch(Circle((0, 0), 1.0, facecolor="#F9FAFB", edgecolor=CORE_EDGE, lw=1.8))
ax.axvline(0, color="#98A2B3", lw=1.1)
ax.axhline(0, color="#98A2B3", lw=1.1)
inactive = wave <= 1e-6
ax.scatter(u_long[inactive], u_z[inactive], s=38, color="#98A2B3",
           edgecolor="white", linewidth=0.7, zorder=4)
ax.scatter(u_long[~inactive], u_z[~inactive],
           s=55 + 190 * wave[~inactive], c=wave[~inactive],
           cmap="YlOrRd", vmin=0, vmax=1, edgecolor=INK,
           linewidth=0.6, zorder=5)
ax.set_xlabel(r"$u_i^{\parallel}$: rear $(-)$ to front $(+)$",
              color=INK, fontsize=18, fontweight="bold")
ax.set_ylabel(r"$u_i^z$: below $(-)$ to above $(+)$",
              color=INK, fontsize=18, fontweight="bold")
ax.tick_params(colors=GUIDE, labelsize=14)
ax.grid(ls=":", color="#D0D5DD", zorder=0)
cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="YlOrRd"),
                    ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"drive weight $w_i$", fontsize=18, color=INK)
cbar.ax.tick_params(labelsize=14)

fig.text(
    0.5,
    0.055,
    r"Green: $u_i^{\parallel}=-0.70$   |   Blue: $u_i^{\perp}=-0.20$   |   Purple: $u_i^z=-0.69$",
    ha="center",
    fontsize=18,
    fontweight="bold",
    color=INK,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT)
