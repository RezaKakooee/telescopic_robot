"""Show how one body-fixed RoboBall rod changes role as the core rolls."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).with_name("assets") / "same-rod-changing-role.png"

INK = "#101828"
CORE = "#667085"
CORE_EDGE = "#1D2939"
ROD = "#98A2B3"
FOOT = "#344054"
HIGHLIGHT = "#C11574"
HIGHLIGHT_EDGE = "#851651"
XB = "#D92D20"
ZB = "#155EEF"
WORLD = "#067647"
ROTATE = "#175CD3"

CORE_R = 0.72
FOOT_R = 0.065
ROD_REACH = 1.02
CENTRE = np.array([0.0, 1.085])


def arrow(ax, start, end, color, width=2.4, style="-|>", zorder=12):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=width,
                        mutation_scale=16, shrinkA=0, shrinkB=0),
        zorder=zorder,
    )


def direction(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def draw_card(ax, center_x, center_y, role, world_vector, rod_midpoint, arrow_rad):
    card_w = 0.98
    card_h = 0.64
    x0 = center_x - card_w / 2
    y0 = center_y - card_h / 2

    # Outer unified card
    card = FancyBboxPatch(
        (x0, y0), card_w, card_h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        fc="white", ec="#D0D5DD", lw=1.5, zorder=18,
    )
    ax.add_patch(card)

    # Line 1: same rod i
    ax.text(center_x, center_y + 0.18, r"$\mathbf{same\ rod\ } \boldsymbol{i}$",
            color=HIGHLIGHT, fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=20)

    # Line 2: Role badge
    ax.text(
        center_x, center_y + 0.01, role,
        ha="center", va="center", color=HIGHLIGHT, fontsize=11.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.20", fc="#FDF2F8", ec=HIGHLIGHT, lw=1.3),
        zorder=20,
    )

    # Line 3: World-frame vector
    ax.text(center_x, center_y - 0.18, world_vector,
            color=INK, fontsize=11, fontweight="bold", ha="center", va="center", zorder=20)

    # Smooth pointer from card edge to active rod midpoint
    dx = rod_midpoint[0] - center_x
    dy = rod_midpoint[1] - center_y
    if abs(dx) > abs(dy):
        attach_x = center_x + np.sign(dx) * (card_w / 2 + 0.02)
        attach_y = center_y
    else:
        attach_x = center_x
        attach_y = center_y + np.sign(dy) * (card_h / 2 + 0.02)

    arrow_patch = FancyArrowPatch(
        (attach_x, attach_y),
        rod_midpoint,
        connectionstyle=f"arc3,rad={arrow_rad}",
        arrowstyle="-|>,head_width=4.5,head_length=7",
        color=HIGHLIGHT,
        lw=2.0,
        shrinkA=2, shrinkB=4,
        zorder=22,
    )
    ax.add_patch(arrow_patch)


def draw_panel(ax, body_angle, role, world_vector, rotation_text, card_x, card_y, arrow_rad):
    ax.set_aspect("equal")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-0.10, 2.25)
    ax.axis("off")

    # Fixed ground
    ax.fill_between([-1.75, 1.75], -0.10, 0, color="#EAECF0", zorder=0)
    ax.plot([-1.75, 1.75], [0, 0], color="#475467", lw=2.4, zorder=1)

    # Travel direction
    arrow(ax, np.array([0.18, 2.05]), np.array([1.25, 2.05]), WORLD, width=2.6)
    ax.text(0.71, 2.11, r"forward $+x_W$", color=WORLD, fontsize=12.5,
            fontweight="bold", ha="center")

    # Background context rods: distributed around the core, preserving a clear zone around -z_B (270 deg)
    base_body_angles = np.deg2rad([25, 60, 95, 135, 170, 205, 335])
    for b_ang in base_body_angles:
        world_ang = b_ang + body_angle
        d = direction(world_ang)
        start = CENTRE + CORE_R * d
        end = CENTRE + 0.86 * d
        ax.plot([start[0], end[0]], [start[1], end[1]], color=ROD,
                lw=3.0, solid_capstyle="round", zorder=3)
        ax.add_patch(Circle(end, 0.025, fc=FOOT, ec=INK, lw=0.6, zorder=4))

    # Core
    ax.add_patch(Circle(CENTRE, CORE_R, fc=CORE, ec=CORE_EDGE, lw=2.4, zorder=5))
    ax.scatter(*CENTRE, s=30, color=INK, zorder=14)
    ax.text(CENTRE[0] + 0.04, CENTRE[1] + 0.035, "$O$", color="white",
            fontsize=13.5, fontweight="bold", zorder=15)

    # Body axes rotate with core
    x_body = direction(body_angle)
    z_body = direction(body_angle + np.pi / 2)
    arrow(ax, CENTRE, CENTRE + 0.46 * x_body, XB, width=2.3, zorder=10)
    arrow(ax, CENTRE, CENTRE + 0.46 * z_body, ZB, width=2.3, zorder=10)
    xb_head = CENTRE + 0.54 * x_body
    zb_head = CENTRE + 0.54 * z_body
    ax.text(*xb_head, "$x_B$", color=XB, fontsize=12.5, fontweight="bold",
            ha="center", va="center", zorder=16)
    ax.text(*zb_head, "$z_B$", color=ZB, fontsize=12.5, fontweight="bold",
            ha="center", va="center", zorder=16)

    # Active rod tracked along -z_B
    rod_direction = -z_body
    rod_start = CENTRE + CORE_R * rod_direction
    foot_centre = CENTRE + ROD_REACH * rod_direction
    rod_end = foot_centre - FOOT_R * rod_direction
    ax.plot([rod_start[0], rod_end[0]], [rod_start[1], rod_end[1]],
            color=HIGHLIGHT, lw=8.5, solid_capstyle="round", zorder=8)
    ax.plot([rod_start[0], rod_end[0]], [rod_start[1], rod_end[1]],
            color="white", lw=3.2, solid_capstyle="round", zorder=9)
    ax.add_patch(Circle(foot_centre, FOOT_R, fc=HIGHLIGHT, ec=HIGHLIGHT_EDGE,
                        lw=2.0, zorder=11))

    rod_midpoint = 0.5 * (rod_start + rod_end)

    # Clockwise rolling cue
    rotation = FancyArrowPatch(
        (0.10, 1.90),
        (0.72, 1.45),
        connectionstyle="arc3,rad=-0.44",
        arrowstyle="Simple,tail_width=1.2,head_width=8,head_length=9",
        color=ROTATE,
        lw=1.0,
        zorder=13,
    )
    ax.add_patch(rotation)
    ax.text(0.86, 1.74, rotation_text, color=ROTATE, fontsize=11.5,
            fontweight="bold", ha="center")

    # Unified rod callout card
    draw_card(ax, card_x, card_y, role, world_vector, rod_midpoint, arrow_rad)


fig, axes = plt.subplots(1, 3, figsize=(18, 7.4), facecolor="white")
fig.subplots_adjust(left=0.03, right=0.98, bottom=0.14, top=0.80, wspace=0.08)

draw_panel(
    axes[0],
    body_angle=0.0,
    role="UNDERNEATH",
    world_vector=r"$(u_{i,x}^W, u_{i,z}^W) = (0, -1)$",
    rotation_text="start",
    card_x=-0.92,
    card_y=0.42,
    arrow_rad=-0.12,
)
draw_panel(
    axes[1],
    body_angle=-np.pi / 2,
    role="REAR",
    world_vector=r"$(u_{i,x}^W, u_{i,z}^W) = (-1, 0)$",
    rotation_text=r"$90^\circ$ clockwise",
    card_x=-1.15,
    card_y=0.45,
    arrow_rad=-0.15,
)
draw_panel(
    axes[2],
    body_angle=-3 * np.pi / 2,
    role="FRONT",
    world_vector=r"$(u_{i,x}^W, u_{i,z}^W) = (1, 0)$",
    rotation_text=r"$270^\circ$ clockwise",
    card_x=1.15,
    card_y=0.45,
    arrow_rad=0.15,
)

for ax, heading in zip(
    axes,
    ["1. Underneath the core", "2. Behind the core", "3. In front of the core"],
):
    ax.set_title(heading, fontsize=17, fontweight="bold", color=INK, pad=12)

fig.suptitle("The same body-fixed rod changes its world-frame role",
             fontsize=23, fontweight="bold", color=INK, y=0.96)
fig.text(
    0.5,
    0.895,
    "RoboBall moves right and rotates clockwise; the unshown top position occurs between rear and front.",
    ha="center",
    fontsize=13,
    color="#475467",
)
fig.text(
    0.5,
    0.045,
    r"The body-frame direction $\hat{\mathbf{u}}_i^B$ is constant, but "
    r"$\hat{\mathbf{u}}_i^W=R_{WB}(q)\hat{\mathbf{u}}_i^B$ changes as the core rotates.",
    ha="center",
    fontsize=14,
    color=INK,
    bbox=dict(boxstyle="round,pad=0.35", fc="#F9FAFB", ec="#98A2B3"),
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print("Regenerated", OUT)
