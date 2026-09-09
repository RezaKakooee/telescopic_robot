"""Visualize selected Fibonacci-sphere indices used in the RoboBall blog."""

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d


OUT = Path(__file__).with_name("assets") / "fibonacci-selected-indices.png"
N = 60
REQUESTED = np.array(
    [0, 1, 2, 5, 10, 15, 20, 25, 30, 31, 32, 35, 40, 45, 50, 55, 58, 59, 60]
)
VALID = REQUESTED[REQUESTED < N]


def angles(indices):
    indices = np.asarray(indices, dtype=float)
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / N)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * (indices + 0.5)
    return phi, theta


all_i = np.arange(N)
all_phi, all_theta = angles(all_i)
phi, theta = angles(VALID)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

fig = plt.figure(figsize=(17, 9), facecolor="white")
grid = fig.add_gridspec(2, 2, width_ratios=(1.08, 1.0), hspace=0.33, wspace=0.18)
ax_phi = fig.add_subplot(grid[0, 0])
ax_theta = fig.add_subplot(grid[1, 0])
ax_sphere = fig.add_subplot(grid[:, 1], projection="3d")

colors = plt.cm.viridis(VALID / (N - 1))


def decorate_index_plot(ax):
    ax.axvspan(59.5, 60.5, color="#FEE4E2", alpha=0.9, zorder=0)
    ax.axvline(59.5, color="#D92D20", ls="--", lw=1.6)
    ax.set_xlim(-1.5, 61.5)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.grid(True, color="#D0D5DD", lw=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)


ax_phi.plot(all_i, all_phi, color="#98A2B3", lw=2.0, label="all valid indices")
ax_phi.scatter(VALID, phi, c=colors, s=66, edgecolor="#101828", lw=0.7, zorder=3)
decorate_index_plot(ax_phi)
ax_phi.set_title(r"Polar angle: $\phi_i=\arccos\!\left(1-\frac{2(i+0.5)}{N}\right)$")
ax_phi.set_ylabel(r"$\phi_i$ (radians)")
ax_phi.set_ylim(-0.05, np.pi + 0.18)
ax_phi.set_yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
ax_phi.set_yticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])

ax_theta.plot(all_i, all_theta, color="#98A2B3", lw=2.0)
ax_theta.scatter(VALID, theta, c=colors, s=66, edgecolor="#101828", lw=0.7, zorder=3)
decorate_index_plot(ax_theta)
ax_theta.set_title(r"Azimuth sequence: $\theta_i=\pi(1+\sqrt{5})(i+0.5)$")
ax_theta.set_xlabel("index $i$")
ax_theta.set_ylabel(r"$\theta_i$ (radians, before wrapping)")

# Label only the requested values on the two exact angle plots. Alternating
# offsets keep adjacent indices 30, 31, and 32 readable.
for k, (idx, p, t) in enumerate(zip(VALID, phi, theta)):
    offset = (0, 11 if k % 2 == 0 else -17)
    va = "bottom" if offset[1] > 0 else "top"
    ax_phi.annotate(str(idx), (idx, p), xytext=offset, textcoords="offset points",
                    ha="center", va=va, fontsize=9.5, fontweight="bold")
    ax_theta.annotate(str(idx), (idx, t), xytext=offset, textcoords="offset points",
                      ha="center", va=va, fontsize=9.5, fontweight="bold")

# Show how the same angle pairs become directions on the unit sphere.
u = np.linspace(0, 2 * np.pi, 70)
v = np.linspace(0, np.pi, 40)
sx = np.outer(np.cos(u), np.sin(v))
sy = np.outer(np.sin(u), np.sin(v))
sz = np.outer(np.ones_like(u), np.cos(v))
ax_sphere.plot_surface(sx, sy, sz, color="#D0D5DD", alpha=0.18, linewidth=0)
ax_sphere.plot_wireframe(sx, sy, sz, rstride=7, cstride=7, color="#98A2B3",
                         alpha=0.25, linewidth=0.55)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
ax_sphere.scatter(x, y, z, c=colors, s=74, edgecolor="#101828", lw=0.8,
                  depthshade=False)


ax_sphere.set_title("Selected indices placed on the unit sphere\n"
                    r"($\theta_i$ wraps every $2\pi$)", pad=20)
ax_sphere.set_xlabel("$x$")
ax_sphere.set_ylabel("$y$")
ax_sphere.set_zlabel("$z$")
ax_sphere.set_xticks([])
ax_sphere.set_yticks([])
ax_sphere.set_zticks([])
ax_sphere.set_box_aspect((1, 1, 1))
ax_sphere.set_xlim(-1.15, 1.15)
ax_sphere.set_ylim(-1.15, 1.15)
ax_sphere.set_zlim(-1.15, 1.15)
ax_sphere.view_init(elev=22, azim=38)

fig.suptitle("How selected Fibonacci-sphere indices become bar directions",
             fontsize=22, fontweight="bold", y=0.98)
fig.text(
    0.5,
    0.015,
    "N = 60. Labels identify the requested indices; colour progresses from i = 0 to i = 59.",
    ha="center",
    fontsize=12.5,
    color="#344054",
)


# Place index annotations directly beside their corresponding points on the unit sphere.
fig.canvas.draw()
offsets = {
    0:  (-7, 6, "right", "bottom"),
    1:  (7, 5, "left", "bottom"),
    2:  (8, 0, "left", "center"),
    5:  (-8, 0, "right", "center"),
    10: (-8, 0, "right", "center"),
    15: (-8, 0, "right", "center"),
    20: (-8, 0, "right", "center"),
    25: (-8, -4, "right", "top"),
    30: (8, 0, "left", "center"),
    31: (-8, 0, "right", "center"),
    32: (7, 5, "left", "bottom"),
    35: (6, 6, "left", "bottom"),
    40: (7, 4, "left", "bottom"),
    45: (7, -4, "left", "top"),
    50: (8, -1, "left", "center"),
    55: (-8, 0, "right", "center"),
    58: (-8, 0, "right", "center"),
    59: (8, -2, "left", "center"),
}

for idx, xx, yy, zz in zip(VALID, x, y, z):
    px, py, _ = proj3d.proj_transform(xx, yy, zz, ax_sphere.get_proj())
    dx, dy, ha, va = offsets.get(int(idx), (6, 6, "left", "bottom"))
    ax_sphere.annotate(
        str(idx),
        xy=(px, py),
        xycoords="data",
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=10.5,
        fontweight="bold",
        color="#101828",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        annotation_clip=False,
    )
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
plt.close(fig)
print(OUT)
