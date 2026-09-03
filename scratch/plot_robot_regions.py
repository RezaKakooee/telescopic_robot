import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 10))

# Draw the core
core = plt.Circle((0, 0), 0.15, color='gray', alpha=0.5, label='Core')
ax.add_artist(core)

# Define 8 regions
regions = {
    "Top (North)": (0, 1),
    "Bottom (South)": (0, -1),
    "Left (West)": (1, 0),  # +Y is Left in our previous convention
    "Right (East)": (-1, 0),
    "Top-Left (NW)": (0.707, 0.707),
    "Top-Right (NE)": (-0.707, 0.707),
    "Bottom-Left (SW)": (0.707, -0.707),
    "Bottom-Right (SE)": (-0.707, -0.707)
}

# Draw conceptual rods and labels
for name, (dy, dz) in regions.items():
    # Rod line
    start_y = 0.15 * dy
    start_z = 0.15 * dz
    end_y = 0.35 * dy
    end_z = 0.35 * dz
    ax.plot([start_y, end_y], [start_z, end_z], 'b-', linewidth=3)
    
    # Text label
    text_y = 0.45 * dy
    text_z = 0.45 * dz
    
    # Adjust alignment based on position
    ha = 'center'
    va = 'center'
    if dy > 0.1: ha = 'left'
    elif dy < -0.1: ha = 'right'
    if dz > 0.1: va = 'bottom'
    elif dz < -0.1: va = 'top'
    
    ax.text(text_y, text_z, name, ha=ha, va=va, fontsize=12, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Draw Walls
ax.axvline(x=0.20, color='r', linestyle='--', label='Left Wall (+Y)')
ax.axvline(x=-0.20, color='r', linestyle='--', label='Right Wall (-Y)')

ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
ax.set_aspect('equal')
ax.set_xlabel('Y Axis (Left/Right)')
ax.set_ylabel('Z Axis (Up/Down)')
ax.set_title('Robot Rod Naming Convention (Y-Z View)')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')

# Save to artifacts
plt.savefig('/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/robot_regions.png', dpi=150, bbox_inches='tight')
print("Plot saved.")
