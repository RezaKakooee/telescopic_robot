import os; os.environ["MUJOCO_GL"]="egl"
import scripts.skills.run_motordrome_wall_of_death as R
print(f"{'close':>6} {'dspeed':>7} {'brake':>6} | {'max drop':>9} {'time down':>10} {'z_end':>6} {'v_end':>6} {'parked':>7}")
for cr, ds, bg in ((0.09,0.90,0.9),(0.14,0.85,1.2),(0.20,0.80,1.5),(0.09,0.80,1.5)):
    r = R.run(seconds=110, record_video=False, descend_after=45,
              close_rate=cr, descend_speed=ds, brake_gain=bg)
    print(f"{cr:6.2f} {ds:7.2f} {bg:6.2f} | {r['max_drop']:9.2f} {r['down_time']:10.1f} "
          f"{r['z_end']:6.2f} {r['v_end']:6.2f} {str(r['parked']):>7}", flush=True)
