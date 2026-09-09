"""Generate the rod-extension geometry figure used in the RoboBall blog."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


OUT = Path(__file__).with_name("assets") / "rod-extension-geometry.png"

RC = 0.150
L0 = 0.010
EI = 0.160
FOOT_R = 0.013

INK = "#101828"
CORE = "#667085"
CORE_EDGE = "#101828"
HUB = "#6BA7BD"
SLEEVE = "#344054"
MIDDLE = "#98A2B3"
INNER = "#F9FAFB"
RC_COLOR = "#155EEF"
L0_COLOR = "#B54708"
EI_COLOR = "#C11574"
FOOT_LABEL_COLOR = "#C11574"
FOOT_COLOR = "#D92D8A"
FOOT_EDGE = "#851651"
SURFACE_COLOR = "#F79009"
NESTED_ARROW_COLOR = "#155EEF"
GREEN = "#067647"
RED = "#D92D20"


def dim_arrow(ax, start, end, y, color, label, label_y=None):
    """Draw a high-contrast double-headed dimension arrow and witness lines."""
    ax.plot([start, start], [y - 0.011, y + 0.011], color=color, lw=2.2, zorder=8)
    ax.plot([end, end], [y - 0.011, y + 0.011], color=color, lw=2.2, zorder=8)
    ax.annotate(
        "",
        xy=(end, y),
        xytext=(start, y),
        arrowprops=dict(arrowstyle="<->", color=color, lw=2.7),
        zorder=9,
    )
    ax.text(
        (start + end) / 2,
        y - 0.015 if label_y is None else label_y,
        label,
        ha="center",
        va="top",
        color=color,
        fontsize=12.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=1.1, alpha=0.97),
        zorder=10,
    )


def base_panel(ax, title):
    ax.set_aspect("equal")
    ax.set_xlim(-0.19, 0.37)
    ax.set_ylim(-0.205, 0.205)
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", color=INK, pad=12)
    ax.add_patch(Circle((0, 0), RC, fc=CORE, ec=CORE_EDGE, lw=2.2, zorder=1))
    ax.add_patch(Circle((0, 0), 0.072, fc=HUB, ec="none", alpha=0.68, zorder=2))
    ax.scatter([0], [0], s=24, color=INK, zorder=8)
    ax.text(-0.008, -0.017, r"$O$", ha="right", va="top", color="white", fontsize=12, zorder=9)
    ax.text(
        0,
        -0.043,
        "clear central hub\n$r_{\mathrm{hub}} = 7.2\,\mathrm{cm}$",
        ha="center",
        va="top",
        fontsize=10.2,
        color="white",
        fontweight="bold",
        zorder=9,
    )


def foot(ax, x, annotate_center=True):
    ax.add_patch(Circle((x, 0), FOOT_R, fc=FOOT_COLOR, ec=FOOT_EDGE, lw=1.8, zorder=8))
    ax.scatter([x], [0], s=18, color="white", zorder=10)
    if not annotate_center:
        return
    ax.annotate(
        "foot centre",
        xy=(x, 0),
        xytext=(x + 0.035, 0.036),
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        arrowprops=dict(arrowstyle="->", color=FOOT_LABEL_COLOR, lw=2.0, shrinkA=4, shrinkB=5),
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=FOOT_LABEL_COLOR, alpha=0.96),
        zorder=12,
    )


fig, axes = plt.subplots(1, 2, figsize=(16, 8.2), facecolor="#252A33")
fig.subplots_adjust(left=0.035, right=0.975, top=0.76, bottom=0.16, wspace=0.10)

for ax in axes:
    ax.set_facecolor("white")

fig.suptitle(
    "One RoboBall bar works like a telescoping radio antenna",
    fontsize=25,
    fontweight="bold",
    color=INK,
    y=0.955,
)

# Legend.
legend_handles = [
    Rectangle((0, 0), 1, 1, fc=SLEEVE, ec=INK, label="fixed outer sleeve"),
    Rectangle((0, 0), 1, 1, fc=MIDDLE, ec="#475467", label="middle stage"),
    Rectangle((0, 0), 1, 1, fc=INNER, ec=RC_COLOR, label="inner stage"),
    Rectangle((0, 0), 1, 1, fc=FOOT_COLOR, ec=FOOT_EDGE, label="rubber foot"),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.875),
    ncol=4,
    frameon=True,
    facecolor="white",
    edgecolor="#98A2B3",
    fontsize=11.5,
)

# Retracted panel.
ax = axes[0]
base_panel(ax, r"Retracted:  $e_i=0$")
ax.add_patch(FancyBboxPatch((0.072, -0.0132), 0.084, 0.0264,
                            boxstyle="round,pad=0.001,rounding_size=0.013",
                            fc=SLEEVE, ec=INK, lw=1.5, zorder=4))
ax.add_patch(FancyBboxPatch((0.075, -0.0102), 0.083, 0.0204,
                            boxstyle="round,pad=0.001,rounding_size=0.010",
                            fc=MIDDLE, ec="#475467", lw=1.3, zorder=5))
ax.add_patch(FancyBboxPatch((0.078, -0.0076), 0.080, 0.0152,
                            boxstyle="round,pad=0.001,rounding_size=0.008",
                            fc=INNER, ec=RC_COLOR, lw=1.5, zorder=6))
foot(ax, RC + L0)
ax.annotate(
    "all three tubes are nested",
    xy=(0.118, 0.006),
    xytext=(0.105, 0.095),
    ha="center",
    color=NESTED_ARROW_COLOR,
    fontsize=11,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=NESTED_ARROW_COLOR, lw=1.8),
    bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#98A2B3"),
    zorder=11,
)
dim_arrow(ax, 0, RC, -0.142, RC_COLOR, r"$r_c=15.0\,\mathrm{cm}$")
dim_arrow(
    ax,
    RC,
    RC + L0,
    -0.034,
    L0_COLOR,
    r"$\ell_0=1.0\,\mathrm{cm}$" + "\nsurface to foot centre",
    -0.049,
)
ax.annotate(
    "core surface",
    xy=(-0.106, 0.106),
    xytext=(-0.055, 0.082),
    ha="center",
    va="center",
    color=SURFACE_COLOR,
    fontsize=11,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=SURFACE_COLOR, lw=2.2),
    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=SURFACE_COLOR, alpha=0.97),
    zorder=11,
)

# Extended panel.
ax = axes[1]
base_panel(ax, r"Fully extended:  $e_i=16.0\,\mathrm{cm}$")
foot_x = RC + L0 + EI
ax.add_patch(FancyBboxPatch((0.072, -0.0132), 0.088, 0.0264,
                            boxstyle="round,pad=0.001,rounding_size=0.013",
                            fc=SLEEVE, ec=INK, lw=1.5, zorder=4))
ax.add_patch(FancyBboxPatch((0.152, -0.0102), 0.085, 0.0204,
                            boxstyle="round,pad=0.001,rounding_size=0.010",
                            fc=MIDDLE, ec="#475467", lw=1.3, zorder=5))
ax.add_patch(FancyBboxPatch((0.232, -0.0076), 0.076, 0.0152,
                            boxstyle="round,pad=0.001,rounding_size=0.008",
                            fc=INNER, ec=RC_COLOR, lw=1.6, zorder=6))
foot(ax, foot_x, annotate_center=False)

# Direction and foot-position vector.
ax.annotate("", xy=(foot_x, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#7A5AF8", lw=2.5, ls="--"), zorder=7)
ax.text(0.068, 0.019, r"$\mathbf{p}_i^B$", color="#53389E", fontsize=15,
        fontweight="bold", zorder=10)
ax.text(0.202, -0.046, r"rod direction  $\hat{\mathbf{u}}_i^B$", ha="right", va="center",
        color=RED, fontsize=11.5, fontweight="bold", zorder=10)
ax.annotate("", xy=(0.305, -0.046), xytext=(0.207, -0.046),
            arrowprops=dict(arrowstyle="->", color=RED, lw=2.2), zorder=10)

dim_arrow(ax, RC, foot_x, -0.105, EI_COLOR, r"$\ell_0+e_i=17.0\,\mathrm{cm}$")

# Mechanical detail callouts.
for x in (0.156, 0.236):
    ax.plot([x, x], [-0.017, 0.017], color=GREEN, lw=2.0, zorder=8)
ax.annotate("0.41 cm\noverlap", xy=(0.156, 0.014), xytext=(0.168, 0.112),
            ha="center", color=GREEN, fontsize=9.5, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6), zorder=10)
ax.annotate("0.41 cm\noverlap", xy=(0.236, 0.014), xytext=(0.256, 0.116),
            ha="center", color=GREEN, fontsize=9.5, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6), zorder=10)
ax.annotate(
    r"$r_f=1.3\,\mathrm{cm}$",
    xy=(foot_x, FOOT_R),
    xytext=(foot_x - 0.018, 0.055),
    ha="center",
    color=GREEN,
    fontsize=10.5,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8),
    zorder=10,
)
ax.annotate("outer radial surface", xy=(foot_x + FOOT_R, 0), xytext=(0.355, 0.028),
            ha="left", color=GREEN, fontsize=10.2, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.7), zorder=10)

# Equation and note.
fig.text(
    0.5,
    0.09,
    r"$q_{i,\mathrm{middle}}=0.5e_i,\qquad q_{i,\mathrm{inner}}=e_i,"
    r"\qquad \mathbf{p}_i^B=(r_c+\ell_0+e_i)\hat{\mathbf{u}}_i^B$",
    ha="center",
    va="center",
    fontsize=16,
    color=INK,
    bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="#98A2B3", lw=1.2),
)
fig.text(
    0.5,
    0.035,
    "Cutaway drawn to scale from the current multi-stage MJCF geometry; "
    "tube thicknesses and full-stroke overlaps are included.",
    ha="center",
    va="center",
    fontsize=10.5,
    color="#344054",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT)
