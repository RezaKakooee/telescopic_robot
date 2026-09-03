# Project status — what was done and where we are

*Last update: 2026-08-13. Written as a handover so work can continue on any
server without the old chat history.*

## What this project is

A telescopic sphere robot ("RadialSphere"): a ball with 60 bars that extend
and retract to make it roll. Physics: RoboVerse `metasim` + MuJoCo (CPU).
Control is two-level: RL picks the direction ("go 45° right"), a scripted
controller moves the 60 bars. See `docs/ENV_OVERVIEW.md` for the env details.

## Story so far (chronological)

1. **Modular refactor** — one package `radial_sphere/` with config-driven env.
2. **Visible telescoping** — bars rebuilt as sleeve/rod/foot units, per-bar
   colors, full-travel controller. Fixed a MuJoCo contact bug (tiny feet sank
   into the floor → explicit `solref/solimp` + `priority=1`).
3. **Scenarios** — `path` (sine), `roundtrip` (out and back toward the
   camera), `goal` (random target), `obstacle` (random pillars, 2 always
   block the straight line), `maze` (level 1: ~28.5 m serpentine corridor of
   thin fixed iron walls).
4. **RL steering layer** — `SteeringEnv`: action = direction (+drive) in the
   goal frame, held 10 low-level steps; obs = velocity/goal features +
   3 nearest pillars + 16 lidar rays. PPO (SB3) + VecNormalize.
5. **Reward fix for mazes** — distance to goal is now **geodesic** (through
   the walls' free space, Dijkstra field), not straight-line.

### Recent Skills & Handoffs:
- [Horizontal Wall Run Parkour Skill (Section 14 in Project Journey)](project_journey/02_skill_library_and_the_skill_course.md#14-horizontal-wall-run-parkour-wall-run--inertia-ride)
- [Verified composed stairs skill](stairs_skill.md)
- [Cylinder / Motordrome Wall of Death Handoff](cylinder_wall_of_death_handoff.md)
- [Imitation & RL Fine-tuning Report](imitation_and_rl_finetuning_report.md)
6. **Cameras** — `chase` (follows behind), `bird` (top-down, follows),
   `bird_fixed` (static, sees the whole maze; default for maze).
7. **Project layout** — mirrors `~/ant_swarm`: `configs/rl/config.yaml`
   (OmegaConf + Hydra, `-cn` variants, `key=value` overrides,
   `RADIAL_SPHERE_CONFIG` env var), `scripts/{heuristic,rl,env}/`,
   `ops/sb_train.sh` (+ `sb_train_gpu.sh`), shared run id
   (`radial_sphere/run_id.py`) across run dir / wandb / .out log,
   loguru logging with per-run `train.log`, wandb inside the run dir.

## Results

| task | heuristic (scripted) | PPO steering |
|---|---|---|
| path / roundtrip | 100 % success | (not needed) |
| goal (random target) | 100 % | 100 % after 100k decisions (easy) |
| obstacle (pillars) | **0 %** — stalls at a pillar | **solved**, reward ~13 |
| maze level 1 (corridor) | **0 %** — pushes into the wall under the goal | **solved**: reward 33.9 of ~34.4 max, goal in ~1800 of 4000 steps, eval 100 % |

## Where things are (key runs under storage_local/)

- Maze RL model: `20260813_1132__20431881__train_rl/`
  (`checkpoints/final.zip` + `vecnormalize.pkl` — both needed for eval)
- Maze RL video: `20260813_1351__local__eval_rl__maze/renders/`
- Heuristic stuck-in-maze video: `20260813_1129__local__heuristic_agent__maze/renders/`
- Obstacle RL model: `radial_sphere__20260812_2305__20381975__rl_train__obstacle/`
- SLURM job logs: `storage_local/sci_out/<run id>.out`
- wandb project: `telescopic_robot` (run name = run id)

## How to run

    # scripted baseline (fails in the maze — expected)
    python scripts/heuristic/heuristic_agent.py --kind maze

    # train (SLURM; log → storage_local/sci_out/<run id>.out)
    sbatch ops/sb_train.sh train_rl "" --kind maze --steps 150000
    # GPU variant (rtx4090): sbatch ops/sb_train_gpu.sh ... rl.device=cuda

    # evaluate a trained run with videos
    python scripts/rl/eval_rl.py --run storage_local/<train run dir> --kind maze

    # any config value can be overridden:  key=value  (Hydra dotlist)
    python scripts/rl/train_rl.py --kind maze rl.total_steps=3e5 rl.n_envs=8

## New server setup

    git clone <repo>
    conda create -n roboverse python=3.10 && conda activate roboverse
    pip install -r requirements.txt
    export MUJOCO_GL=egl        # or osmesa for headless video rendering
    wandb login                 # optional

## Known facts and traps

- The camera makes stepping ~50x slower (renders 1280×720 inside
  `get_states`). Training must run with `enable_camera=False` (the training
  script does this).
- Old scicore login node: 10 GB per-user memory cap. Training there: only
  `--n-envs 1`. Use SLURM for real runs.
- metasim's `ObsSaver` buffers frames in RAM (OOM risk) — we use our
  streaming `radial_sphere.render.VideoRecorder` instead.
- A SLURM timeout once killed the final save; checkpoints now save their
  VecNormalize stats too (`save_vecnormalize=True`), so any checkpoint is
  evaluable.

## Hardware & Sim2Real Mechanisms

- Comprehensive hardware feasibility, 3D CAD cutaways, and Sim2Real engineering breakdown available in:
  [`docs/telescoping_mechanisms_sim2real.md`](./telescoping_mechanisms_sim2real.md)
  (Covers cascade cable-driven telescoping, nested lead-screws, pneumatic cylinders, anti-rotation spline keyways, and bushing overlap dynamics).

## Sim-Ready Architecture & 22-Skill Showcase (2026-09-03)

1. **Continuous Overlap Multi-Stage Telescoping**:
   - Replaced single-stage rod pokes with 3 selectable physical architectures: `single_stage`, `multi_stage` (Option 1 cascade concentric), and `zip_chain` (Option 2 tangential push-chain spool).
   - Enforced continuous positive overlap across 100% of the $16\text{ cm}$ stroke with $+4.2\text{ mm}$ sleeve-to-stage1 and $+4.5\text{ mm}$ stage1-to-piston overlap, completely eliminating floating feet or gaps while maintaining an open $7.4\text{ cm}$ protected central hub for avionics and LiPo battery.

2. **DeepMind MuJoCo Menagerie Sim-Ready Standard**:
   - Integrated full 243-channel on-board sensory suite: central 6-axis IMU (`imu_acc`, `imu_gyro`, `framequat`), 60 joint position encoders (`jointpos`), 60 velocity sensors (`jointvel`), 60 actuator force feedback monitors (`actuatorfrc`), and 60 foot contact touch sensors (`touch`).
   - Calibrated Shore 70A vulcanized polyurethane rubber footpads with 4D contact cone (`condim="4"`: normal, sliding, torsional, rolling friction).
   - Dynamic Dyneema cable compliance via `<equality>` damping (`solref="0.004 1.0"`, `solimp="0.95 0.99 0.001"`).

3. **Explosive Jump Dynamics Calibration**:
   - Calibrated PTFE guide bushing friction ($0.35\text{ Ns/m}$ viscous drag, $0.08\text{ N}$ Coulomb frictionloss) and reflected rotor armature ($0.002\text{ kg}$).
   - Set motor burst envelope to $120.0\text{ N}$ with crisp PD gains ($K_p = 1200\text{ N/m}$, $K_v = 22\text{ Ns/m}$).
   - Result: Vertical leap increased from $0.38\text{ m} \to 0.830\text{ m}$ ($v_z = 3.85\text{ m/s}$), and forward leap increased to $0.714\text{ m}$, cleanly vaulting obstacles with $>24\text{ cm}$ clearance.

4. **Complete 22-Skill Showcase Suite**:
   - Rendered and verified all 22 individual skills + the grand finale 19-discipline continuous parkour course (`00_continuous_skill_course_parkour.mp4`), stored in:
     - Local Workspace: `storage_local/20260903_2334__local__all_skills_showcase/renders/`
     - Artifacts Directory: `cae66589-5edc-46dd-9306-d193640ffe8c/all_skills_showcase/`
