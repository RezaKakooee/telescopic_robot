import os; os.environ["MUJOCO_GL"]="egl"
import scripts.skills.run_wall_run as W
FULL=0.173+0.30
print(f"{'up':>4} {'spd':>4} {'ang':>4} | {'cont s':>7} {'along':>6} {'squash':>7} "
      f"{'wall z':>12} {'peak':>5} {'exit':>5} {'turns':>6} {'endz':>5}")
for up, spd, ang in ((2.4,3.4,22),(2.6,3.4,22),(2.8,3.4,22),
                     (2.6,4.0,22),(2.6,3.4,28),(2.8,4.0,26)):
    r = W.run(seconds=14, record_video=False, launch_up=up, speed=spd, approach_angle=ang)
    lo,hi = r["contact_z"]
    print(f"{up:4.1f} {spd:4.1f} {ang:4.0f} | {r['contact_s']:7.2f} {r['contact_len']:6.2f} "
          f"{(FULL-r['min_gap'])*1000:7.0f} {lo:5.2f}-{hi:5.2f} {r['peak_z']:5.2f} "
          f"{r['exit_v']:5.2f} {r['turns']:6.2f} {r['end_z']:5.2f}", flush=True)
