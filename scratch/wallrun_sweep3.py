import os; os.environ["MUJOCO_GL"]="egl"
import scripts.skills.run_wall_run as W
FULL = 0.173 + 0.30
print(f"{'give':>5} {'span':>5} {'up':>4} {'spd':>4} | {'cont s':>7} {'along':>6} "
      f"{'squash':>7} {'peak':>5} {'exit':>5} {'endz':>5}")
for give, span, up, spd in ((0.18,0.26,2.4,3.4),(0.22,0.30,2.4,3.4),
                            (0.18,0.26,2.9,3.4),(0.22,0.30,2.9,3.4),
                            (0.22,0.30,2.9,4.2),(0.26,0.32,2.9,3.4)):
    r = W.run(seconds=12, record_video=False, give=give, squash_span=span,
              launch_up=up, speed=spd)
    print(f"{give*100:5.0f} {span*100:5.0f} {up:4.1f} {spd:4.1f} | {r['contact_s']:7.2f} "
          f"{r['contact_len']:6.2f} {(FULL-r['min_gap'])*1000:7.0f} {r['peak_z']:5.2f} "
          f"{r['exit_v']:5.2f} {r['end_z']:5.2f}", flush=True)
