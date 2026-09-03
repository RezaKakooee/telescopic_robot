# Backflip skill

This document is the implementation and verification reference for the radial
sphere's backflip. The complete maneuver is driven by
`scripts/skills/run_somersault.py`; the pure rod-target primitive lives in
`skills/flip.py`.

## Definition: what counts as a backflip

The rendered side camera looks from world `-Y`, with world `+X` moving to the
right of the image. Under that convention:

- rightward airborne travel is `dx > 0`;
- counter-clockwise pitch is negative rotation and negative `omega_y`;
- a backflip must satisfy both at the same time.

Rightward travel with clockwise rotation is a **frontflip**, even if its total
rotation is 360 degrees. The regression test checks the sign, not just the
magnitude.

## Why the maneuver uses a rebound

A normal rightward run gives the ball clockwise rolling momentum. Preserving
that momentum through a jump produces a frontflip. The backflip therefore
uses a counter-torque preload followed by a separate rebound launch:

```text
approach
  -> compact
  -> counter-plant
  -> preload-tuck
  -> rebound-launch
  -> aerial-tuck
  -> flare
  -> settle
```

1. **Approach** — drive along world `+X` for 40 control steps with
   `back_gain=5.0`.
2. **Compact** — retract all rods for 4 control steps. This lowers rotational
   inertia and stores the full actuator stroke.
3. **Counter-plant** — command a short reverse contact wave with
   `back_gain=6.0`. This creates negative, counter-clockwise pitch momentum
   and a small preload hop.
4. **Preload tuck** — keep all rods retracted until the preload hop touches
   down.
5. **Rebound launch** — on that touchdown, fire the ground-facing rod fan
   with `launch_power=1.0` and `launch_torque=4.0`. The rear bias produces
   positive-X launch velocity while preserving the negative pitch momentum.
6. **Aerial tuck** — retract all rods during the main flight to increase the
   pitch rate.
7. **Flare** — near 350 degrees, extend every rod to 90% stroke. This brakes
   rotation and creates the landing cage.
8. **Settle** — after touchdown, use `stop()` to brake into a low stance.

The preload hop is preparation. It is deliberately excluded from the
reported backflip rotation and air time.

## Robot and simulation configuration

The skill uses `configs/rl/somersault.yaml`:

| Setting | Value |
|---|---:|
| Core radius | 0.15 m |
| Telescoping rods | 60 |
| Maximum rod stroke | 0.30 m |
| Actuator force limit | 150 N |
| Sliding friction | 1.4 |
| Physics timestep | 0.002 s |
| Physics substeps per control step | 5 |
| Effective control step | 0.01 s |

The effective control step is calculated as
`model.opt.timestep * env.action_repeat`. Do not replace it with a hard-coded
substep count.

## Measurement and acceptance criteria

All aerial metrics start at the main rebound liftoff and stop at its first
descending robot-floor contact. Floor contact must involve a robot geometry;
scenery touching the floor does not count.

Pitch is measured by unwrapping the body's forward-axis angle. The result is
cross-checked by integrating the signed `omega_y` rate over the same flight
window. This prevents a tilted tumble or angle wrap from being accepted as a
backflip.

The test in `tests/test_skills.py::test_somersault` requires:

| Check | Required |
|---|---:|
| Signed airborne pitch | -375 to -340 degrees (CCW) |
| Independent pitch-rate integral | at most -280 degrees |
| Airborne X displacement | at least +0.15 m |
| Takeoff X velocity | greater than +0.10 m/s |
| Peak core height | at least 0.65 m |
| Main-flight air time | at least 0.50 s |
| Touchdown pitch rate | at most 5.0 rad/s magnitude |
| Core impacts | 0 |
| Final linear speed | at most 0.20 m/s |
| Final angular speed | at most 0.35 rad/s |

### Current deterministic result

Using seed 42 and the checked-in configuration:

| Metric | Measured |
|---|---:|
| Signed airborne pitch | **-350.6 degrees CCW** |
| Pitch-rate integral | **-319.8 degrees** |
| Airborne X displacement | **+0.267 m** |
| Peak core height | **1.796 m** |
| Main-flight air time | **1.10 s** |
| Touchdown vertical speed | **-4.34 m/s** |
| Touchdown pitch rate | **-1.55 rad/s** |
| Core impacts | **0** |
| Settled linear speed | **0.029 m/s** |

## Run and verify

From the repository root, using the RoboVerse environment:

```bash
# Render the default backflip video (2x slow motion)
MUJOCO_GL=egl PYTHONPATH=. \
  /home/azureuser/miniconda3/envs/roboverse/bin/python \
  scripts/skills/run_somersault.py

# Run without rendering
MUJOCO_GL=egl PYTHONPATH=. \
  /home/azureuser/miniconda3/envs/roboverse/bin/python \
  scripts/skills/run_somersault.py --no-video

# Run the complete skills regression suite
MUJOCO_GL=egl PYTHONPATH=. \
  /home/azureuser/miniconda3/envs/roboverse/bin/python tests/test_skills.py
```

Videos are written to:

```text
storage_local/<run-id>/renders/somersault_backflip.mp4
```

The video overlay reports signed airborne rotation and labels it `CCW` or
`CW`. A valid default backflip must show a negative value and `CCW`.

## Python interfaces

The low-level primitive returns rod targets for one phase:

```python
from skills import execute_skill

targets = execute_skill(
    "backflip",
    quat,
    dirs_body,
    max_extend,
    phase="launch",        # runner label: rebound-launch
    direction="backward",
    launch_power=1.0,
    launch_torque=4.0,
)
```

`skills/flip.py` is intentionally stateless. It does not detect preload
liftoff, preload touchdown, main liftoff, or final touchdown. Use
`scripts/skills/run_somersault.py::run` when you need the complete verified
maneuver:

```python
from scripts.skills.run_somersault import run

result = run(direction="backward", record_video=False)
assert result["rotation_direction"] == "ccw"
assert result["air_dx"] > 0
```

The registry aliases `backflip`, `somersault`, and `flip` to the same
low-level primitive. `frontflip` is also available, but it is a different
direction preset and must not be used as evidence that the backflip passes.

## Files

| File | Responsibility |
|---|---|
| `skills/flip.py` | Pure per-phase rod targets and direction presets |
| `scripts/skills/run_somersault.py` | Contact-driven state machine, signed telemetry, video |
| `configs/rl/somersault.yaml` | Full-stroke robot and contact configuration |
| `radial_sphere/scenario.py` | Open backflip arena and unobstructed goal placement |
| `tests/test_skills.py` | Signed direction, travel, landing, and settling assertions |

## Retuning warnings

The contact sequence is physics-dependent. Re-run the signed regression test
after changing any of the following:

- rod stroke, mass, actuator force, damping, or armature;
- floor friction or contact solver parameters;
- physics timestep or action repeat;
- approach, compact, counter-plant, rebound, tuck, or flare parameters.

Never validate with absolute rotation alone. At minimum, require signed
quaternion rotation, signed pitch-rate integral, positive airborne X travel,
touchdown, and zero core impact.
