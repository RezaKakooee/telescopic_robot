"""Hierarchical Skill-Arbitration Environment.

The RL policy acts as a high-level skill selector:
- Observes the local elevation/occupancy map patch + global navigation waypoints + robot state.
- Selects either skills_rl primitives or skills/ controllers via rl.skill_backend.
- Macro actions select a skill; hybrid actions also supply continuous parameters.
- The skills backend executes a jump's phases within one policy decision.
"""
from __future__ import annotations

import numpy as np

from ._gym import gym, spaces
from .map_perception import LocalMapPatchExtractor, MonotonicWaypointTracker, WaypointTracker
from .mujoco_env import MujocoRadialSphereEnv
import skills_rl as S


class SkillArbitrationEnv(gym.Env):
    """Hierarchical RL environment for skill arbitration in complex, real-world-like terrains."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        config=None,
        scenario=None,
        action_mode: str = "hybrid",  # "hybrid" (Option 1) or "macro" (Option 2)
        grid_size: int = 16,
        patch_span: float = 4.0,
        decision_every: int = 10,
        max_steps: int = 1500,
        randomize: bool = False,
        seed: int | None = None,
        training: bool = False,
        **env_kwargs,
    ):
        super().__init__()
        low_level_kwargs = dict(env_kwargs)
        if "max_steps" in low_level_kwargs:
            del low_level_kwargs["max_steps"]

        self.env = MujocoRadialSphereEnv(config, max_steps=1000000, scenario=scenario, **low_level_kwargs)
        self.cfg = self.env.cfg
        rl = getattr(self.cfg, "rl", None)
        self.training_start_probability = float(getattr(rl, "training_start_probability", 0.0)) if training else 0.0
        self.training_start_points = np.asarray(
            getattr(rl, "training_start_points", []), dtype=float).reshape(-1, 2)
        if not 0.0 <= self.training_start_probability <= 1.0:
            raise ValueError("training_start_probability must be between 0 and 1")
        if self.training_start_probability and (not len(self.training_start_points)
                                                 or self.env.scenario.kind != "playground"):
            raise ValueError("Training starts require playground start points")

        # Action mode switch: config knob overrides default if provided
        configured_mode = getattr(rl, "action_mode", None)
        self.action_mode = str(configured_mode if configured_mode is not None else action_mode).lower()
        if self.action_mode not in ("hybrid", "macro"):
            raise ValueError(f"Unknown action_mode '{self.action_mode}', expected 'hybrid' or 'macro'")
        self.skill_backend = str(getattr(rl, "skill_backend", "skills_rl")).lower()
        if self.skill_backend not in ("skills_rl", "skills"):
            raise ValueError(f"Unknown skill_backend '{self.skill_backend}', expected 'skills_rl' or 'skills'")
        self.handcrafted = None
        if self.skill_backend == "skills":
            from . import handcrafted_skill_backend
            self.handcrafted = handcrafted_skill_backend
        self.skill_names = self.handcrafted.SKILL_NAMES if self.handcrafted else S.SKILL_NAMES

        self.grid_size = int(getattr(rl, "grid_size", grid_size))
        self.patch_span = float(getattr(rl, "patch_span", patch_span))
        self.k = int(getattr(rl, "decision_every", decision_every))
        if self.k < 1:
            raise ValueError("decision_every must be positive")
        self.max_steps = int(max_steps if max_steps is not None else getattr(rl, "max_steps", 1500))

        self.scenario = self.env.scenario
        self.patch_extractor = LocalMapPatchExtractor(self.scenario)
        if self.scenario.kind == "playground" and hasattr(self.env, "model"):
            self.patch_extractor.rasterize_physics(
                self.env.model, self.env.data, self.env._terrain_ray_groups)
        tracker_cls = MonotonicWaypointTracker if getattr(self.scenario, "monotonic_path", False) else WaypointTracker
        self.waypoint_tracker = tracker_cls(self.scenario.path_pts, self.scenario.goal)

        # 1. Action Space Setup
        if self.handcrafted:
            self.action_space = self.handcrafted.action_space(self.action_mode)
        elif self.action_mode == "hybrid":
            self.action_space = S.action_space()
        else:
            # Option 2: 7 continuous logits for the 7 macro primitives
            self.action_space = spaces.Box(-1.0, 1.0, shape=(len(S.PRIMITIVES),), dtype=np.float32)

        # 2. Observation Space Setup:
        # Local patch (2 * grid_size * grid_size) + guidance (6) + proprioception (12)
        self.patch_dim = 2 * self.grid_size * self.grid_size
        self.guidance_dim = 6
        self.proprio_dim = 12
        self.obs_dim = self.patch_dim + self.guidance_dim + self.proprio_dim

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.control_steps = 0
        self.on_control_step = None  # optional video callback, also called inside jump options
        self.prev_path_dist = 0.0
        self.last_skill_idx = self.skill_names.index("stop") if self.handcrafted else 0
        self.last_skill_name = "stop" if self.handcrafted else "stance"
        # Match the actuator model, not the visual/mechanical rod layout.
        from skills_rl.calibration import profile_for_config
        self.profile = profile_for_config(self.cfg)

        # Configurable reward shaping and exploration options
        self.stage_rewards = bool(getattr(rl, "stage_rewards", False))
        self.forward_incentive = bool(getattr(rl, "forward_incentive", False))
        self.anti_stagnation = bool(getattr(rl, "anti_stagnation", False))
        self.stagnant_steps = 0
        self.no_progress_limit = int(getattr(rl, "no_progress_steps", 100 if self.anti_stagnation else 0))
        self.progress_min_delta = float(getattr(rl, "progress_min_delta", 0.10))
        self.progress_anchor = 0.0
        self.no_progress_steps = 0
        self.cleared_milestones: set[str] = set()
        # Reward for pushing into an obstacle face (hurdle, box side, pipe wall, ring).
        # Charged once per macro step in which a hit happened; zero keeps old behaviour.
        self.obstacle_hit_penalty = float(getattr(rl, "obstacle_hit_penalty", 0.0))
        self.episode_hits = 0

    def _get_obs(self) -> np.ndarray:
        pos = self.env.data.qpos[:3].copy()
        vel = self.env.data.qvel[:3].copy()
        ang_vel = self.env.data.qvel[3:6].copy()
        quat = self.env.data.qpos[3:7].copy()

        # 1. Local elevation & occupancy patch (flattened)
        patch = self.patch_extractor.get_patch(pos, grid_size=self.grid_size, patch_span=self.patch_span)
        flat_patch = patch.reshape(-1)

        # 2. Global guidance vector (6,)
        guidance = self.waypoint_tracker.get_guidance(pos)

        # 3. Proprioceptive vector (12,)
        # Compute roll and pitch from quaternion
        w, x, y, z = quat
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        cf = self.env.get_rod_contact_forces() if hasattr(self.env, "get_rod_contact_forces") else np.zeros(60)
        mean_cf = float(np.mean(cf)) if cf is not None else 0.0
        n_contacts = float(np.sum(cf > 1.0)) if cf is not None else 0.0

        proprio = np.array([
            vel[0], vel[1], vel[2],
            ang_vel[0], ang_vel[1], ang_vel[2],
            float(pos[2]),
            roll, pitch,
            float(self.last_skill_idx) / len(self.skill_names),
            mean_cf * 0.02,
            n_contacts * 0.05,
        ], dtype=np.float32)

        return np.concatenate([flat_patch, guidance, proprio]).astype(np.float32)

    def _robot_state(self):
        """Read current pose so world-directed primitives follow the rolling core."""
        return S.RobotState(
            quat=self.env.data.qpos[3:7].copy(),
            dirs_body=self.env.dirs_body,
            max_extend=self.env.max_extend,
            lin_vel=self.env.data.qvel[:3].copy(),
            core_z=float(self.env.data.qpos[2]),
            core_vz=float(self.env.data.qvel[2]),
            contact_forces=self.env.get_rod_contact_forces() if hasattr(self.env, "get_rod_contact_forces") else None,
            terrain_clearances=self.env.get_terrain_clearances() if hasattr(self.env, "get_terrain_clearances") else None,
            profile=self.profile,
        )

    def _outside_playground(self, pos: np.ndarray) -> bool:
        """Check the core XY against the closed playground wall outline.

        Ray crossings include the inner walls, so jumping into the courtyard
        also leaves the course. Height is deliberately ignored: jumps within
        the corridor are legal, but clearing a boundary is an episode failure.
        Other scenarios do not necessarily have a closed wall outline.
        """
        if self.scenario.kind not in ("playground", "robotics_playground", "proving_ground"):
            return False
        walls = np.asarray(self.scenario.walls)
        x1, y1, x2, y2 = walls.T
        x, y = pos[:2]
        crosses_y = (y1 > y) != (y2 > y)
        crossing_x = x1 + np.divide(
            (y - y1) * (x2 - x1), y2 - y1,
            out=np.zeros(len(walls), dtype=float), where=crosses_y,
        )
        return np.count_nonzero(crosses_y & (x < crossing_x)) % 2 == 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _obs, info = self.env.reset(seed=seed)
        if hasattr(self.waypoint_tracker, "reset"):
            self.waypoint_tracker.reset()
        training_start = None
        if self.training_start_probability and self.np_random.random() < self.training_start_probability:
            import mujoco

            index = int(self.np_random.integers(len(self.training_start_points)))
            xy = self.training_start_points[index]
            if self._outside_playground(xy):
                raise ValueError(f"Training start {xy.tolist()} is outside the playground")
            ix, iy = self.patch_extractor._world_to_grid(*xy)
            ground = float(self.patch_extractor.global_elevation[iy, ix])
            self.env.data.qpos[:3] = [*xy, ground + self.env.sphere_radius + self.env.base_ext + .02]
            self.env.data.qvel[:] = 0.0
            mujoco.mj_forward(self.env.model, self.env.data)
            # Settle onto the sampled support before initializing reward history.
            for _ in range(100):
                mujoco.mj_step(self.env.model, self.env.data)
            self.env._prev_dist = self.env._nav_distance(self.env.data.qpos[:2])
            info = self.env._get_info()
            training_start = index

        self.current_step = 0
        self.control_steps = 0
        pos = self.env.data.qpos[:2]
        closest_idx, self.prev_path_dist = self.waypoint_tracker.get_path_progress(pos)
        self.progress_anchor = self.prev_path_dist
        self.no_progress_steps = 0
        self.last_skill_idx = self.skill_names.index("stop") if self.handcrafted else 0
        self.last_skill_name = "stop" if self.handcrafted else "stance"
        self.stagnant_steps = 0
        self.cleared_milestones = set()
        self.episode_hits = 0

        obs = self._get_obs()
        info["skill_name"] = self.last_skill_name
        info["skill_backend"] = self.skill_backend
        info["closest_waypoint_idx"] = closest_idx
        info["path_dist_remaining"] = self.prev_path_dist
        info["out_of_bounds"] = False
        info["stalled"] = False
        info["training_start"] = training_start
        return obs, info

    def step(self, action: np.ndarray):
        self.current_step += 1
        pos = self.env.data.qpos[:3].copy()
        guidance = self.waypoint_tracker.get_guidance(pos)
        waypoint_dir = guidance[:2]
        theta_waypoint = float(np.arctan2(waypoint_dir[1], waypoint_dir[0]))

        # Decode action according to selected action_mode
        option = None
        if self.handcrafted:
            skill_idx, skill_name, skill_params, heading = self.handcrafted.decode(
                action, self.action_mode, theta_waypoint)
            def ground_height(xy):
                ix, iy = self.patch_extractor._world_to_grid(*xy)
                return float(self.patch_extractor.global_elevation[iy, ix])
            option = self.handcrafted.SkillOption(
                self.env, skill_name, skill_params, heading, self.k, ground_height)
        elif self.action_mode == "hybrid":
            # Option 1: skills_rl decode
            skill_name, skill_params = S.decode(action)
            skill_idx = S.SKILL_NAMES.index(skill_name)
        else:
            # Option 2: Macro skills
            act_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            skill_idx = int(np.argmax(act_arr[:len(S.SKILL_NAMES)]))
            skill_name = S.SKILL_NAMES[skill_idx]

            # Populate context-aligned macro parameters
            if skill_name == "drive":
                skill_params = {"azimuth": theta_waypoint, "speed": 1.1}
            elif skill_name == "thrust":
                # A broad downward burst clears the hurdle while preserving
                # running momentum. A shallow forward-pointing burst recoils
                # backward and the old narrow sector did not reliably lift.
                skill_params = {"azimuth": theta_waypoint + np.pi, "elevation": -float(np.pi / 2.0), "power": 1.0, "spread": 0.8}
            elif skill_name == "brake":
                skill_params = {"strength": 2.2}
            elif skill_name == "tuck":
                skill_params = {"extension": 0.015}
            elif skill_name == "conform":
                skill_params = {"ride_height": 0.22, "terrain_gain": 0.8, "stiffness": 0.6}
            elif skill_name == "brace":
                skill_params = {"azimuth": theta_waypoint, "elevation": 0.0, "extension": 0.8, "spread": 0.5, "both_sides": 0.0}
            else:  # stance
                skill_params = {"height": 0.035}

        self.last_skill_idx = skill_idx
        self.last_skill_name = skill_name

        # Step native MuJoCo physics for k decision substeps
        wall_contact = False
        hit_geoms: set[str] = set()
        out_of_bounds = False
        term = trunc = False
        sub_info = {}
        executed = 0
        for substep in range(option.max_steps if option else self.k):
            rod_targets = (option.targets(substep) if option else
                           S.act(skill_name, self._robot_state(), **skill_params))
            _obs, _r, term, trunc, sub_info = self.env.step(rod_targets)
            executed += 1
            self.control_steps += 1
            if self.on_control_step is not None:
                self.on_control_step(self)
            if int(sub_info.get("wall_contact", 0)) or int(sub_info.get("n_wall_contacts", 0) > 0):
                wall_contact = True
            hit_geoms.update(sub_info.get("obstacle_hit_geoms", ()))
            out_of_bounds = self._outside_playground(self.env.data.qpos[:3])
            if term or trunc or out_of_bounds:
                break
            if option and (option.complete(executed)
                           or np.linalg.norm(self.env.data.qpos[:2] - self.scenario.goal[:2]) < .50
                           or self.env.data.qpos[2] < -.15
                           or self.control_steps >= self.max_steps * self.k):
                break

        # Preserve primitive timing; long options consume their actual control budget.
        duration = executed / self.k if option else 1

        new_pos = self.env.data.qpos[:3].copy()
        new_dist_goal = float(np.linalg.norm(new_pos[:2] - self.scenario.goal[:2]))

        # Calculate Reward
        # 1. Progress along the route, even when a leg leads away from the goal.
        closest_idx, path_dist_remaining = self.waypoint_tracker.get_path_progress(new_pos)
        # An illegal shortcut must not earn progress before its failure penalty.
        progress = 0.0 if out_of_bounds else self.prev_path_dist - path_dist_remaining
        reward = progress * 20.0
        self.prev_path_dist = path_dist_remaining
        if not out_of_bounds and self.progress_anchor - path_dist_remaining >= self.progress_min_delta:
            self.progress_anchor = path_dist_remaining
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += duration
        stalled = self.no_progress_limit > 0 and self.no_progress_steps >= self.no_progress_limit

        # 2. Forward velocity along waypoint direction
        v_xy = self.env.data.qvel[:2]
        v_fwd = float(np.dot(v_xy, waypoint_dir))
        speed = float(np.linalg.norm(v_xy))

        if self.forward_incentive:
            # Strong forward velocity reward along current waypoint heading
            reward += 0.20 * np.clip(v_fwd, -0.2, 2.0)
            if wall_contact:
                reward -= 0.02  # forgiving wall collision so agent explores corridors
        else:
            reward += 0.05 * np.clip(v_fwd, -0.5, 1.5)
            if wall_contact:
                reward -= 0.06

        # 3. Small step penalty to encourage speed
        reward -= 0.01 * duration

        # 4. Anti-stagnation penalty
        if self.anti_stagnation:
            if speed < 0.08:
                self.stagnant_steps += 1
                if self.stagnant_steps > 5:
                    reward -= 0.04 * min(self.stagnant_steps, 15)
                if skill_name == "stance" and v_fwd < 0.05 and new_dist_goal > 1.0:
                    reward -= 0.08
            else:
                self.stagnant_steps = 0

        # 5. Milestone Stage Curriculum rewards
        if self.stage_rewards and not out_of_bounds:
            x_pos, y_pos, z_pos = float(new_pos[0]), float(new_pos[1]), float(new_pos[2])
            # Leg 1: Runway & Hurdle
            if "m1_runway" not in self.cleared_milestones and x_pos >= 2.5 and y_pos < 1.0:
                self.cleared_milestones.add("m1_runway")
                reward += 10.0
            if "m2_hurdle" not in self.cleared_milestones and x_pos >= 5.5 and y_pos < 1.0:
                self.cleared_milestones.add("m2_hurdle")
                reward += 25.0
            if "m3_turn1" not in self.cleared_milestones and x_pos >= 8.0 and y_pos >= 1.0:
                self.cleared_milestones.add("m3_turn1")
                reward += 30.0

            # Leg 2: 3-Box Parkour over Valleys
            if "m4_box1" not in self.cleared_milestones and 7.8 <= x_pos <= 10.2 and 2.2 <= y_pos <= 3.4 and z_pos >= 0.28:
                self.cleared_milestones.add("m4_box1")
                reward += 40.0
            if "m5_box2" not in self.cleared_milestones and 7.8 <= x_pos <= 10.2 and 3.85 <= y_pos <= 5.05 and z_pos >= 0.28:
                self.cleared_milestones.add("m5_box2")
                reward += 50.0
            if "m6_box3" not in self.cleared_milestones and 7.8 <= x_pos <= 10.2 and 5.50 <= y_pos <= 6.70 and z_pos >= 0.28:
                self.cleared_milestones.add("m6_box3")
                reward += 60.0
            if "m7_turn2" not in self.cleared_milestones and x_pos >= 7.5 and y_pos >= 8.0:
                self.cleared_milestones.add("m7_turn2")
                reward += 40.0

            # Leg 3: Stairs & Deck
            if "m8_stairs_deck" not in self.cleared_milestones and x_pos <= 5.0 and y_pos >= 8.0:
                self.cleared_milestones.add("m8_stairs_deck")
                reward += 40.0
            if "m9_turn3" not in self.cleared_milestones and x_pos <= 1.0 and y_pos >= 9.5:
                self.cleared_milestones.add("m9_turn3")
                reward += 40.0

            # Leg 4: Rough Bed, Conduit & Goal
            if "m10_conduit" not in self.cleared_milestones and -1.2 <= x_pos <= 1.2 and y_pos >= 13.5:
                self.cleared_milestones.add("m10_conduit")
                reward += 50.0

        # Terminal conditions
        terminated = bool(term)
        truncated = bool(trunc)
        success = bool(sub_info.get("success", False))
        if getattr(self.scenario, "monotonic_path", False) and path_dist_remaining >= 1.0:
            # A tour may pass the goal position early: the low-level env's
            # goal termination does not count until the route is done.
            terminated, success = False, False

        if hit_geoms:
            self.episode_hits += 1
            reward -= self.obstacle_hit_penalty

        if out_of_bounds:
            reward -= 20.0
            terminated = True
            success = False
        elif (success or new_dist_goal < 0.50) and (
                not getattr(self.scenario, "monotonic_path", False) or path_dist_remaining < 1.0):
            # On a tour that passes the goal early, only the end of the route counts.
            reward += 100.0
            terminated = True
            success = True
        elif new_pos[2] < -0.15:
            # Trapped inside deep pit
            reward -= 20.0
            terminated = True
        elif stalled:
            reward -= 20.0
            terminated = True
        elif self.current_step >= self.max_steps or (option and self.control_steps >= self.max_steps * self.k):
            truncated = True

        obs = self._get_obs()
        info = {
            "step": self.current_step,
            "dist_to_goal": new_dist_goal,
            "closest_waypoint_idx": closest_idx,
            "path_dist_remaining": path_dist_remaining,
            "path_progress": progress,
            "success": success,
            "is_success": success,  # Stable-Baselines3 episode success metric
            "out_of_bounds": out_of_bounds,
            "stalled": bool(stalled and not success and not out_of_bounds),
            "no_progress_steps": self.no_progress_steps,
            "wall_contact": int(wall_contact),
            "obstacle_hit": int(bool(hit_geoms)),
            "obstacle_hit_geoms": sorted(hit_geoms),
            "episode_hits": self.episode_hits,
            "skill_name": skill_name,
            "action_mode": self.action_mode,
            "skill_backend": self.skill_backend,
            "skill_control_steps": executed,
            "control_steps": self.control_steps,
            "skill_phase": option.phase if option else None,
            "skill_timed_out": bool(option and option.is_jump and executed == option.max_steps
                                    and not option.complete(executed) and not (terminated or truncated)),
            "speed": float(np.linalg.norm(v_xy)),
            "core_z": float(new_pos[2]),
        }

        return obs, reward, terminated, truncated, info

    def render(self, camera_name: str | None = None):
        return self.env.render(camera_name=camera_name)

    def render_all(self) -> dict[str, np.ndarray]:
        return {"chase": self.env.render(camera_name="chase")}

    def close(self):
        self.env.close()
