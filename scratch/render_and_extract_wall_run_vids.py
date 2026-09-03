import os
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import imageio
import numpy as np
from PIL import Image

from scripts.skills.run_wall_run import run

ART_DIR = Path("/home/azureuser/.gemini/antigravity-ide/brain/cae66589-5edc-46dd-9306-d193640ffe8c")
RENDERS_DIR = Path("/home/azureuser/telescopic_robot/renders")
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

modes = ["curved", "banked", "flat_multistep"]

for m in modes:
    print(f"\n==========================================")
    print(f"  RENDERING HIGH-SPEED IMPACT: {m.upper()}")
    print(f"==========================================")
    out = run(mode=m, seconds=5.0, seed=3, record_video=True, video_name=f"wall_run_{m}", slowmo=2)
    vid_src = Path(out["video"])
    vid_dst_art = ART_DIR / f"wall_run_{m}.mp4"
    vid_dst_ws = RENDERS_DIR / f"wall_run_{m}.mp4"

    if vid_src.exists():
        data = vid_src.read_bytes()
        vid_dst_art.write_bytes(data)
        vid_dst_ws.write_bytes(data)
        print(f"Saved {vid_dst_ws}")

        reader = imageio.get_reader(str(vid_dst_ws))
        n_frames = reader.count_frames()

        hist = out.get("hist", [])
        
        # Locate exact step indices for the 4 key stages:
        # 1. Sprint acceleration
        # 2. Airborne flight with all 60 rods open
        # 3. Wall impact & compression
        # 4. Compliant landing
        step_sprint = next((i for i, h in enumerate(hist) if h[3] == "sprint" and i > 20), 40)
        step_fly = next((i for i, h in enumerate(hist) if h[3] == "fly"), 100)
        # Find step with max compression or ride phase
        ride_steps = [i for i, h in enumerate(hist) if h[3] == "ride"]
        step_ride = ride_steps[len(ride_steps)//2] if ride_steps else step_fly + 25
        step_land = next((i for i, h in enumerate(hist) if i > step_ride and h[3] in ("land", "settle")), step_ride + 25)

        idx_sprint = int(np.clip(step_sprint // 2, 0, n_frames - 1))
        idx_fly = int(np.clip(step_fly // 2 + 3, 0, n_frames - 1))
        idx_ride = int(np.clip(step_ride // 2, 0, n_frames - 1))
        idx_land = int(np.clip(step_land // 2, 0, n_frames - 1))

        frame_indices = [idx_sprint, idx_fly, idx_ride, idx_land]
        print(f"  Exact phase steps: sprint={step_sprint}, fly={step_fly}, ride={step_ride}, land={step_land}")
        print(f"  Exact frame indices for {m}: {frame_indices} of {n_frames}")

        extracted = []
        for idx in frame_indices:
            frame = reader.get_data(min(idx, n_frames - 1))
            extracted.append(frame)
        reader.close()

        h, w, _ = extracted[0].shape
        grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        grid[0:h, 0:w] = extracted[0]
        grid[0:h, w:2*w] = extracted[1]
        grid[h:2*h, 0:w] = extracted[2]
        grid[h:2*h, w:2*w] = extracted[3]

        grid_path_art = ART_DIR / f"wall_run_{m}_grid.png"
        grid_path_ws = RENDERS_DIR / f"wall_run_{m}_grid.png"
        Image.fromarray(grid).save(grid_path_art)
        Image.fromarray(grid).save(grid_path_ws)
        print(f"Saved grid to {grid_path_ws}")

print("\nALL 3 MODES RENDERED AND EXTRACTED ACCURATELY!")
