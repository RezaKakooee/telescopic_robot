import os; os.environ["MUJOCO_GL"]="egl"
import scripts.skills.run_wall_run as W
print(f"{'push':>5} | " + " ".join(f"{'run'+str(i):>22}" for i in (1,2,3)) + f" {'squash':>7} {'endz':>5}")
for pf in (0.65, 0.70, 0.75):
    r = W.run(seconds=24, record_video=False, repeats=3, push_frac=pf)
    cells=[]
    for k in range(3):
        if k < len(r["runs"]):
            g=r["runs"][k]
            cells.append(f"{g['secs']:5.2f}s {g['along']:4.2f}m z{g['z'][0]:.2f}-{g['z'][1]:.2f}")
        else:
            cells.append(f"{'missed':>22}")
    print(f"{pf:5.2f} | " + " ".join(f"{c:>22}" for c in cells) +
          f" {r['squash']*1000:7.0f} {r['end_z']:5.2f}", flush=True)
