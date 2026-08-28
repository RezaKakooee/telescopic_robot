from pathlib import Path
from PIL import Image, ImageDraw

img_paths = [
    "docs/project_journey/assets/forward_jump_1_crouch.png",
    "docs/project_journey/assets/forward_jump_2_apex1.png",
    "docs/project_journey/assets/forward_jump_3_sprint.png",
    "docs/project_journey/assets/forward_jump_4_hurdle_apex.png",
    "docs/project_journey/assets/forward_jump_5_landing.png",
]

labels = [
    "1. Deep Crouch Preload (z=0.16m)",
    "2. Standing Forward Launch (+35cm)",
    "3. High-Speed Sprint (vx=2.3 m/s)",
    "4. Explosive Hurdle Leap (1.1m Arc)",
    "5. Compliant Landing & Rollout",
]

crops = []
for p in img_paths:
    img = Image.open(p)
    w, h = img.size
    left_view = img.crop((0, 0, w // 2, h))
    crops.append(left_view)

cw, ch = crops[0].size
strip = Image.new("RGB", (cw * len(crops), ch), (255, 255, 255))

for idx, crop in enumerate(crops):
    strip.paste(crop, (idx * cw, 0))

draw = ImageDraw.Draw(strip)

for idx, label in enumerate(labels):
    bx = idx * cw + 15
    by = 15
    draw.rectangle([bx, by, bx + len(label)*10 + 20, by + 32], fill=(20, 24, 30, 220))
    draw.text((bx + 10, by + 8), label, fill=(255, 255, 255))

strip.save("docs/project_journey/assets/forward_jump_progression_grid.png")
strip.save("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c/forward_jump_progression_grid.png")
print("Saved 5-stage forward jump progression grid to docs/project_journey/assets/forward_jump_progression_grid.png")
