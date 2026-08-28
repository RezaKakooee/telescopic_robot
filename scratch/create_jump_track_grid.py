from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

img_paths = [
    "docs/project_journey/assets/forward_jump_track_1_stand.png",
    "docs/project_journey/assets/forward_jump_track_2_jump1_apex.png",
    "docs/project_journey/assets/forward_jump_track_3_sprint.png",
    "docs/project_journey/assets/forward_jump_track_4_hurdle_apex.png",
    "docs/project_journey/assets/forward_jump_track_5_landing.png",
]

labels = [
    "1. Stand at Start Line (x=0.0m)",
    "2. Standing Forward Jump (Peak z=0.55m)",
    "3. High-Speed Sprint (vx=1.8 m/s)",
    "4. Airborne Hurdle Leap (z=0.45m, +1.1m in air!)",
    "5. Touchdown & Rollout (x=2.3m)",
]

crops = []
for p in img_paths:
    img = Image.open(p)
    crops.append(img)

# 5-panel horizontal strip
w, h = crops[0].size
strip = Image.new("RGB", (w * 5, h), (20, 24, 30))

for i, img in enumerate(crops):
    strip.paste(img, (w * i, 0))

draw = ImageDraw.Draw(strip)

for idx, label in enumerate(labels):
    bx, by = w * idx + 10, 8
    draw.rectangle([bx, by, bx + len(label)*8 + 16, by + 28], fill=(15, 18, 22, 230))
    draw.text((bx + 8, by + 6), label, fill=(255, 255, 255))

strip.save("docs/project_journey/assets/forward_jump_track_progression.png")
strip.save("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/forward_jump_track_progression.png")
print("Saved 5-panel strip -> docs/project_journey/assets/forward_jump_track_progression.png")
