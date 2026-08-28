from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

img_paths = [
    "docs/project_journey/assets/standing_jump_1_stand.png",
    "docs/project_journey/assets/standing_jump_2_crouch.png",
    "docs/project_journey/assets/standing_jump_3_takeoff.png",
    "docs/project_journey/assets/standing_jump_4_apex.png",
    "docs/project_journey/assets/standing_jump_5_landing.png",
]

labels = [
    "1. Stationary Stand (z=0.21m)",
    "2. Deep Crouch Preload (z=0.16m)",
    "3. Explosive Takeoff (+3.14 m/s)",
    "4. Airborne Flight Apex (+45.0 cm)",
    "5. Compliant Landing (z=0.19m)",
]

# Load only the 3D close-up half (left half: width 0 to 640)
crops = []
for p in img_paths:
    img = Image.open(p)
    w, h = img.size
    # Left viewport crop: width w // 2
    left_view = img.crop((0, 0, w // 2, h))
    crops.append(left_view)

# Create a horizontal strip
cw, ch = crops[0].size
strip = Image.new("RGB", (cw * len(crops), ch), (255, 255, 255))

for idx, crop in enumerate(crops):
    strip.paste(crop, (idx * cw, 0))

draw = ImageDraw.Draw(strip)

# Add title badges
for idx, label in enumerate(labels):
    bx = idx * cw + 15
    by = 15
    draw.rectangle([bx, by, bx + len(label)*10 + 20, by + 32], fill=(20, 24, 30, 220))
    draw.text((bx + 10, by + 8), label, fill=(255, 255, 255))

strip.save("docs/project_journey/assets/standing_jump_progression_grid.png")
strip.save("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/standing_jump_progression_grid.png")
print("Saved 5-stage progression grid to docs/project_journey/assets/standing_jump_progression_grid.png")
