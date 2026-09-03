# Wall of Death: what was wrong, and what works now

## 1. Result

The robot spirals up the drome bowl and holds a high orbit. Measured over 75 s:

| quantity | value |
|---|---|
| laps in 150 s | 40.3 |
| peak height | 2.11 m |
| sustained height | 1.77 m, never below 1.65 m from 10 s to 150 s |
| sustained speed | 5.0 m/s |
| bowl depth | 2.69 m |
| ride line tops out at | 1.87 m |

The ride settles by about 10 s and stays. Before this work the robot reached
1.30-1.40 m once and fell back to the apron.

Run it:

```bash
python scripts/skills/run_motordrome_wall_of_death.py --seconds 75 --video
```

## 2. Why the old arena could never work

The old arena was a vertical cylinder. A vertical wall cannot hold this robot,
and no amount of speed or friction changes that.

On a vertical wall the only thing that can hold a body up is friction. For a
**rolling** ball the friction that holds it also applies a torque, so the ball
spins up and rolls straight down the wall. This was measured, not assumed:

| wall radius | speed used | result |
|---|---|---|
| 0.70 m | 1.3x and 2.0x the friction limit | slid down |
| 0.90 m | same | slid down |
| 1.10 m | same | slid down |
| 1.40 m | same | slid down |
| 1.80 m | same | slid down |

Every case fell about 1.12 m in 0.57 s. Radius made no difference. A uniformly
extended ball, used as a plain wheel with no control at all, also slid down.

Two further things were measured and ruled out along the way:

* Pressing many rods against the wall does hold the robot, but it turns the
  ball into a skid. Sliding friction then costs about 20 m/s^2 of drag and the
  speed collapses to zero within a second.
* Driving the wave up the wall instead of along it does not help either. Pitch
  values from 0 to 30 were tried; all fell.

## 3. What the arena is now

A drome bowl: flat floor in the middle, then boards that bank over as they go
out, with the vertical wall left standing above the rim.

The bank is not a guess. A body circling at radius `r` and speed `v` needs
`v^2 / r` of inward acceleration, and a bank of angle `b` supplies
`g * tan(b)` of it with no friction at all. Set them equal:

    v^2 = g * r * tan(b)

So the bank that carries a chosen speed at every radius is `tan(b) = v^2/(g r)`,
and the bowl is built to follow it. That is what makes the whole climb
continuous instead of one ledge the robot has to survive. Near the floor edge
the curve is far too steep to drive onto, so the bank is ramped in linearly and
the gentler of the two wins.

Settings live in `configs/rl/motordrome.yaml`:

| setting | value | meaning |
|---|---|---|
| `floor_radius` | 1.10 m | flat pad, used to build the first speed |
| `wall_radius` | 3.50 m | rim, where the vertical wall begins |
| `bowl_ride_end` | 3.10 m | end of the ride line; the boards curl up past it |
| `bowl_ride_speed` | 4.20 m/s | speed the bank carries where the ramp meets it |
| `bowl_rim_speed` | 5.00 m/s | speed the bank carries at the end of the ride line |
| `bowl_ramp_rate` | 1.50 | how fast the bank comes in off the flat floor |
| `bowl_lip_bank` | 78 deg | boards curl to this by the rim, to meet the wall |
| `bowl_facets` | 32 | planks around the barrel; see below |
| `bowl_segments` | 26 | conical rings, crowded into the curl |
| `wall_friction` | 1.80 | boards against rubber feet |
| `robot.max_extend` | 0.30 m | long stroke: 5.45 m/s, against 4.55 at 0.26 |

### The join to the wall, and the plank count

The boards used to stop at their riding angle and the wall started at 90
degrees, leaving a hard corner. Now the last stretch past `bowl_ride_end`
curls up to 78 degrees so the two meet almost tangentially. The curl sits
outside the ride line on purpose: a robot pushed out there meets boards far
too steep for its speed and loses everything.

Two things went wrong while smoothing it, both measured over 150 s runs.

**More planks make a rougher ride, not a smoother one.** Every plank overlaps
its neighbours, and a foot in the overlap picks up two contacts at once:

| planks | holds | laps |
|---|---|---|
| 32 | 1.77 m | 40.1 |
| 48 | 0.68 m | 11.8 |

Thinning the overlap recovers only part of it. Narrow rings cost the same way,
so the rings are crowded into the curl, where nothing rides, and left wide
across the middle, where the robot does.

**The commanded radius has to stop at the ride line.** Left free it pushed the
robot into the curl, which threw it off. `Bowl.ride_limit` caps both the
commanded radius and the throttle.

An earlier entry ramp of 6.0 was a 42-degree step at the floor edge. The robot
drove into it and stopped dead. 1.50 spreads the same bank over a metre.

## 4. What the controller does

Three ideas, in `skills/wall_of_death.py`.

**1. The gait is aimed at the surface, not at the floor.** `move` builds its
push wave on how far each rod points *down*. `surface_drive` builds the same
wave on how far each rod points *into the surface*, whatever that surface is.
Given the floor normal it reproduces `move` exactly, to the last bit.

**2. The surface is measured, never guessed.** `surface_frame` reads MuJoCo's
contact normals each step. No zone test for floor, bowl or wall appears
anywhere in the controller, so the joins need no special case.

This was also where the worst bug lived. The first version took the direction
from the core to the contact point. Those two are only the same for a plain
sphere. This robot stands on rods, so a ball resting on a rod tilted 45 degrees
reports a 45-degree slope on flat ground. That error fed straight into the
gait, and until it was fixed the robot could not build speed at all.

**3. The climb is a spiral, and the throttle follows the radius.**
`advance_radius` widens the commanded circle only while the measured speed can
hold the next one out, at 0.10 m of radius per second. `Bowl.target_speed` then
asks the gait for the speed that circle can hold, and no more.

That last point was the other big failure. At full throttle the robot reached
5.93 m/s at radius 1.21, where holding the circle needs 29 m/s^2. It cannot
make more than about 4.5. So it ran wide, hit the steep boards at an angle it
could not ride, and lost everything. Speed is not free: on a bank it has to be
paid for with radius.

## 5. Also fixed on the way

The speed curve was calibrated only at 0.16 m stroke. Asking for 1.2 m/s on the
0.30 m build returned the amplitude that cruises at 1.2 m/s on the short build,
which on the long one is nearer 2.9 m/s. A second curve was measured at 0.30 m
and `gain_for_speed` now interpolates by stroke. The two do not scale: the
ratio runs from 4.0x at the lowest amplitude to 1.95x at the highest.

## 6. Files

| file | what changed |
|---|---|
| `skills/wall_of_death.py` | new: `Bowl`, `advance_radius`, `surface_frame`, `wall_of_death` |
| `skills/locomotion.py` | new `surface_drive`; stroke-aware speed curve |
| `radial_sphere/scenario.py` | bowl profile from the ride condition |
| `radial_sphere/mujoco_mjcf.py` | bowl built from a radius/height profile; configurable friction |
| `configs/rl/motordrome.yaml` | new: the arena and the long-stroke build |
| `scripts/skills/run_motordrome_wall_of_death.py` | rewritten on the skill |
| `tests/test_skills.py` | test 17: sustained ride, not a peak |

## 7. Known limits

* The ride settles at about a 40-degree bank, not a vertical wall. That is
  what 4.75 m/s buys at radius 3.0. Riding steeper means riding lower.
* Deeper bowls were tried. A 4.40 m rim gives a 2.50 m bowl and a 2.79 m peak,
  but the arena is then 8.8 m across and the robot needs longer to spiral out.
* Height is capped by speed. The ride point always satisfies
  `r * tan(b) = v^2 / g`, so a faster robot is the only way to go both higher
  and steeper.
