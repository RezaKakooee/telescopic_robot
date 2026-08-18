# Handoff — Telescopic Robot Maze RL

Date: 2026-08-14

## Current state

Level-3 maze training uses one fixed wall layout with two endpoint variants:

- `configs/rl/maze_level3_fixed_goal.yaml`: fixed goal, random valid start.
- `configs/rl/maze_level3_fixed_start.yaml`: fixed start, random valid goal.

Success was corrected after video review. It now requires an actual MuJoCo
contact between the green goal and the robot's colliding `core_geom`/`foot_*`
geometries. Geodesic distance is only progress shaping. The old `goal_eps:
0.45` proximity termination was removed, the goal is a fixed collider,
`info["success"]` reports contact, and evaluation uses that field. Near-goal
steering also continues toward the target instead of stopping at 0.45 m.

## Checkpoints

- Fixed goal / random start:
  `storage_local/radial__20260814_1238__local_1543734__train_rl/checkpoints/final.zip`
- Fixed start / random goal:
  `storage_local/radial__20260814_1242__local_1607268__train_rl/checkpoints/final.zip`

Each run also has its required `vecnormalize.pkl` at the run root.

## Correct strict-contact evaluation

Three deterministic rendered episodes per checkpoint:

- Fixed goal / random start: **2/3 contacts (66.7%)**
  - `storage_local/radial__20260814_2142__local_4102116__eval_rl/renders/`
- Fixed start / random goal: **1/3 contact (33.3%)**
  - `storage_local/radial__20260814_2142__local_4102527__eval_rl/renders/`

Earlier reported 100%/20% results are invalid for the user's definition: they
measured proximity, not touch. The checkpoints have not yet been retrained
after the contact-success correction.

## Recommended next step

Resume the stronger fixed-goal checkpoint under the corrected contact reward,
then evaluate at least 10 deterministic rendered episodes. Example:

```bash
cd /home/azureuser/telescopic_robot
./ops/local_train.sh train_rl configs/rl/maze_level3_fixed_goal.yaml \
  --kind maze \
  --resume storage_local/radial__20260814_1238__local_1543734__train_rl
```

No radial train/eval jobs are currently running. The worktree is intentionally
dirty with the level-3 maze, local runner, endpoint presets, camera, and contact
changes; do not discard unrelated modifications. Physics smoke tests confirmed
that 0.44 m proximity fails, real contact succeeds, and randomized fixed goal
geometry relocates correctly.
