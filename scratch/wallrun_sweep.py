import os; os.environ["MUJOCO_GL"]="egl"
import scripts.skills.run_wall_run as W
print(f"{'damp':>5} {'cush_mm':>8} {'ang':>4} | {'contact s':>10} {'along m':>8} "
      f"{'min gap':>8} {'squash mm':>10} {'peak z':>7}")
FULL = 0.173 + 0.30
for damp, cm, ang in ((0.10,0.050,22),(0.10,0.020,22),(0.06,0.012,22),
                      (0.04,0.008,22),(0.06,0.012,30),(0.04,0.008,30)):
    r = W.run(seconds=12, record_video=False, damp=damp, cushion_max=cm,
              approach_angle=ang)
    lo, hi = r["contact_z"]
    print(f"{damp:5.2f} {cm*1000:8.0f} {ang:4.0f} | {r['contact_s']:10.2f} "
          f"{r['contact_len']:8.2f} {r['min_gap']:8.3f} {(FULL-r['min_gap'])*1000:10.0f} "
          f"{r['peak_z']:7.2f}", flush=True)
