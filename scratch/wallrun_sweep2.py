import os; os.environ["MUJOCO_GL"]="egl"
import scripts.skills.run_wall_run as W
FULL = 0.173 + 0.30
print(f"{'give_mm':>8} {'span_mm':>8} {'spd':>5} | {'contact s':>10} {'along m':>8} "
      f"{'squash mm':>10} {'peak z':>7} {'wall z':>14}")
for give, span, spd in ((0.00,0.16,3.4),(0.06,0.16,3.4),(0.12,0.22,3.4),
                        (0.18,0.26,3.4),(0.12,0.22,4.4),(0.18,0.26,4.4)):
    r = W.run(seconds=12, record_video=False, give=give, squash_span=span, speed=spd)
    lo, hi = r["contact_z"]
    print(f"{give*1000:8.0f} {span*1000:8.0f} {spd:5.1f} | {r['contact_s']:10.2f} "
          f"{r['contact_len']:8.2f} {(FULL-r['min_gap'])*1000:10.0f} {r['peak_z']:7.2f} "
          f"{lo:6.2f}-{hi:5.2f}", flush=True)
