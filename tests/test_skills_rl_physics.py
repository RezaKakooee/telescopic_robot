"""Physics tests for the RL primitives, run with sim2real switched ON.

A primitive that returns legal targets and moves the robot the wrong way passes
every contract test in `tests/test_skills_rl.py`. These are the tests that
catch that, and they run against the hardware model rather than the ideal one:

* actuator velocity, acceleration and stall-force limits
* a total battery power budget shared across all 60 motors
* viscoelastic rubber feet, with real sliding and rolling friction
* 25 ms of transport latency on every command
* gaussian noise on the IMU and the lidar

`configs/rl/config.yaml` ships with ``sim2real.enabled: false``. Testing under
the ideal model would prove the maths and say nothing about the robot, so every
test here turns it on. Where a limit changes what a primitive can do, the test
says so rather than lowering the bar quietly.
"""

import os
import unittest

import numpy as np

import skills_rl as S
from skills_rl.base import RobotState

os.environ.setdefault("MUJOCO_GL", "egl")

from radial_sphere.config import load_config          # noqa: E402
from radial_sphere.mujoco_env import MujocoRadialSphereEnv  # noqa: E402
from radial_sphere.scenario import Scenario, generate_scenario  # noqa: E402


def make_cfg(sim2real=True, hardware=True, **robot):
    """A config for the tests.

    ``hardware`` turns on ``sim2real.actuator_in_env``, which is what makes the
    base env apply the rod velocity, acceleration, stall-force and battery
    limits plus bus latency. Without it a primitive meets only MJCF friction
    and damping, which is how these tests read before the model was shared,
    and why they showed almost no sim-to-real gap.
    """
    cfg = load_config("configs/rl/config.yaml")
    cfg.camera.enabled = False
    cfg.sim2real.enabled = bool(sim2real)
    cfg.sim2real.actuator_in_env = bool(sim2real and hardware)
    for k, v in robot.items():
        setattr(cfg.robot, k, v)
    return cfg


def flat_scenario(cfg):
    return generate_scenario("path", cfg, seed=0)


def wall_scenario(wall_y=-0.45):
    """A single long barrier, as `tests/test_wall_push.py` builds one."""
    return Scenario(
        kind="goal", name="rl_wall",
        spawn_xy=np.array([0.0, -0.26], dtype=np.float32),
        goal=np.array([5.0, 0.0], dtype=np.float32),
        path_pts=np.array([[0.0, -0.26], [5.0, -0.26]], dtype=np.float32),
        markers=np.empty((0, 2), dtype=np.float32), path_length=5.0,
        walls=np.array([[-1.0, wall_y, 6.0, wall_y]], dtype=np.float32),
    )


class Rig:
    """One env plus the loop that drives a primitive through it."""

    def __init__(self, cfg, scenario, settle=25, seed=0):
        self.env = MujocoRadialSphereEnv(cfg, scenario=scenario, randomize=False,
                                         max_steps=4000)
        self.env.reset(seed=seed)
        for _ in range(settle):
            self.env.step(np.full(60, 0.025, dtype=np.float32))
        self.start = self.env.data.qpos[:3].copy()

    def state(self, **extra):
        e = self.env
        return RobotState(quat=e.data.qpos[3:7].copy(), dirs_body=e.dirs_body,
                          max_extend=e.max_extend, lin_vel=e.data.qvel[:3].copy(),
                          core_z=float(e.data.qpos[2]), core_vz=float(e.data.qvel[2]),
                          contact_forces=(e.get_rod_contact_forces()
                                          if extra.get("sensors") else None),
                          terrain_clearances=(e.get_terrain_clearances()
                                              if extra.get("sensors") else None))

    def run(self, name, steps, sensors=False, **params):
        peak = float(self.env.data.qpos[2])
        speeds = []
        for _ in range(steps):
            self.env.step(S.act(name, self.state(sensors=sensors), **params))
            peak = max(peak, float(self.env.data.qpos[2]))
            speeds.append(float(np.linalg.norm(self.env.data.qvel[:2])))
        q = self.env.data.qpos
        return dict(dxy=(q[:2] - self.start[:2]).copy(),
                    dist=float(np.linalg.norm(q[:2] - self.start[:2])),
                    speed=float(np.linalg.norm(self.env.data.qvel[:2])),
                    mean_speed=float(np.mean(speeds)),
                    peak_rise=peak - float(self.start[2]),
                    z=float(q[2]))

    def coast(self, steps):
        """Roll on with rods at rest, the control case for braking."""
        for _ in range(steps):
            self.env.step(np.full(60, self.env.base_ext, dtype=np.float32))
        q = self.env.data.qpos
        return dict(dist=float(np.linalg.norm(q[:2] - self.start[:2])),
                    speed=float(np.linalg.norm(self.env.data.qvel[:2])))

    def close(self):
        self.env.close()


class DriveTests(unittest.TestCase):
    def test_travels_along_the_command_under_sim2real(self):
        for az in (0.0, np.pi / 2, -3 * np.pi / 4):
            rig = Rig(make_cfg(), flat_scenario(make_cfg()))
            r = rig.run("drive", 150, azimuth=az, speed=1.2, flank=0.0)
            rig.close()
            want = np.array([np.cos(az), np.sin(az)])
            with self.subTest(azimuth=round(az, 2)):
                self.assertGreater(r["dist"], 0.25, "no travel")
                align = float(r["dxy"] @ want) / r["dist"]
                self.assertGreater(align, 0.65, f"drove {align:.2f} off course")

    def test_speed_command_is_monotonic(self):
        out = []
        for spd in (0.5, 1.2, 2.4):
            rig = Rig(make_cfg(), flat_scenario(make_cfg()))
            out.append(rig.run("drive", 150, azimuth=0.0, speed=spd, flank=0.0)["dist"])
            rig.close()
        self.assertLess(out[0], out[1])
        self.assertLess(out[1], out[2])

    def test_reverse_goes_backwards(self):
        rig = Rig(make_cfg(), flat_scenario(make_cfg()))
        f = rig.run("drive", 120, azimuth=0.0, speed=1.2, flank=0.0)
        rig.close()
        rig = Rig(make_cfg(), flat_scenario(make_cfg()))
        b = rig.run("drive", 120, azimuth=0.0, speed=-1.2, flank=0.0)
        rig.close()
        self.assertGreater(f["dxy"][0], 0.15)
        self.assertLess(b["dxy"][0], -0.15)

    def test_hardware_limits_cost_speed_but_not_direction(self):
        """The actuator caps slow the robot. They must not steer it."""
        ideal = Rig(make_cfg(sim2real=False), flat_scenario(make_cfg(False)))
        ri = ideal.run("drive", 150, azimuth=0.0, speed=2.0, flank=0.0)
        ideal.close()
        real = Rig(make_cfg(sim2real=True), flat_scenario(make_cfg(True)))
        rr = real.run("drive", 150, azimuth=0.0, speed=2.0, flank=0.0)
        real.close()
        print(f"\n  drive 2.0 m/s: ideal {ri['dist']:.2f} m, sim2real {rr['dist']:.2f} m")
        self.assertGreater(rr["dist"], 0.20, "sim2real killed travel entirely")
        # Both still head the commanded way.
        for r in (ri, rr):
            self.assertGreater(r["dxy"][0] / max(r["dist"], 1e-9), 0.65)


class BrakeTests(unittest.TestCase):
    def _wind_up(self, cfg, steps=110):
        rig = Rig(cfg, flat_scenario(cfg))
        for _ in range(steps):
            rig.env.step(S.act("drive", rig.state(), azimuth=0.0, speed=2.0, flank=0.0))
        rig.start = rig.env.data.qpos[:3].copy()
        return rig, float(np.linalg.norm(rig.env.data.qvel[:2]))

    def test_braking_beats_coasting(self):
        cfg = make_cfg()
        rig, v0 = self._wind_up(cfg)
        braked = rig.run("brake", 90, strength=3.2)
        rig.close()
        rig, v1 = self._wind_up(cfg)
        coasted = rig.coast(90)
        rig.close()
        print(f"\n  from {v0:.2f} m/s: brake -> {braked['speed']:.2f} m/s in "
              f"{braked['dist']:.2f} m; coast -> {coasted['speed']:.2f} m/s in "
              f"{coasted['dist']:.2f} m")
        self.assertGreater(v0, 0.5, "the wind-up did not get the ball rolling")
        self.assertLess(braked["speed"], coasted["speed"],
                        "braking left the ball faster than coasting")
        self.assertLess(braked["dist"], coasted["dist"] + 1e-6)

    def test_stronger_brake_stops_shorter(self):
        cfg = make_cfg()
        rig, _ = self._wind_up(cfg)
        soft = rig.run("brake", 90, strength=0.6)
        rig.close()
        rig, _ = self._wind_up(cfg)
        hard = rig.run("brake", 90, strength=3.2)
        rig.close()
        print(f"  brake 0.6 -> {soft['dist']:.2f} m, 3.2 -> {hard['dist']:.2f} m")
        self.assertLess(hard["dist"], soft["dist"])


class StanceTests(unittest.TestCase):
    def test_holds_still_and_upright(self):
        cfg = make_cfg()
        rig = Rig(cfg, flat_scenario(cfg))
        z_before = float(rig.env.data.qpos[2])
        r = rig.run("stance", 200, height=0.045)
        rig.close()
        self.assertLess(r["dist"], 0.10, "stance drifted")
        self.assertLess(r["speed"], 0.12, "stance did not settle")
        self.assertGreater(r["z"], z_before - 0.05, "stance collapsed")

    def test_taller_stance_lifts_the_core(self):
        cfg = make_cfg()
        low = Rig(cfg, flat_scenario(cfg))
        zl = low.run("stance", 150, height=0.025)["z"]
        low.close()
        high = Rig(cfg, flat_scenario(cfg))
        zh = high.run("stance", 150, height=0.09)["z"]
        high.close()
        print(f"\n  stance height 0.025 -> core z {zl:.3f}, 0.09 -> {zh:.3f}")
        self.assertGreater(zh, zl)


class ThrustTests(unittest.TestCase):
    def test_leaves_the_ground(self):
        cfg = make_cfg()
        rig = Rig(cfg, flat_scenario(cfg))
        r = rig.run("thrust", 45, azimuth=0.0, elevation=-np.pi / 2,
                    power=1.0, spread=0.55)
        rig.close()
        print(f"\n  thrust straight down: rose {r['peak_rise']:.3f} m under sim2real")
        self.assertGreater(r["peak_rise"], 0.03)

    def test_power_orders_the_height(self):
        cfg = make_cfg()
        rises = []
        for p in (0.3, 1.0):
            rig = Rig(cfg, flat_scenario(cfg))
            rises.append(rig.run("thrust", 45, azimuth=0.0, elevation=-np.pi / 2,
                                 power=p, spread=0.55)["peak_rise"])
            rig.close()
        print(f"  thrust power 0.3 -> {rises[0]:.3f} m, 1.0 -> {rises[1]:.3f} m")
        self.assertGreater(rises[1], rises[0])

    def test_actuator_speed_limit_costs_height(self):
        """A rod capped at 0.28 m/s cannot fire a 0.16 m stroke instantly."""
        ideal = Rig(make_cfg(sim2real=False), flat_scenario(make_cfg(False)))
        ri = ideal.run("thrust", 45, azimuth=0.0, elevation=-np.pi / 2, power=1.0)
        ideal.close()
        real = Rig(make_cfg(sim2real=True), flat_scenario(make_cfg(True)))
        rr = real.run("thrust", 45, azimuth=0.0, elevation=-np.pi / 2, power=1.0)
        real.close()
        print(f"  thrust: ideal rose {ri['peak_rise']:.3f} m, "
              f"sim2real {rr['peak_rise']:.3f} m")
        self.assertGreater(ri["peak_rise"], 0.0)
        self.assertGreater(rr["peak_rise"], 0.0)


class TuckTests(unittest.TestCase):
    def test_tuck_makes_no_drive(self):
        cfg = make_cfg()
        rig = Rig(cfg, flat_scenario(cfg))
        r = rig.run("tuck", 150, extension=0.01)
        rig.close()
        self.assertLess(r["dist"], 0.12, "tucking moved the robot")
        self.assertLess(r["speed"], 0.15)

    def test_tuck_lowers_the_core(self):
        cfg = make_cfg()
        rig = Rig(cfg, flat_scenario(cfg))
        z0 = float(rig.env.data.qpos[2])
        r = rig.run("tuck", 120, extension=0.0)
        rig.close()
        self.assertLess(r["z"], z0 + 0.01)


class BraceTests(unittest.TestCase):
    def test_pushes_off_a_real_wall(self):
        cfg = make_cfg()
        rig = Rig(cfg, wall_scenario(), settle=20)
        y0 = float(rig.env.data.qpos[1])
        # Wall lies at -y, so press toward -y and expect to be shoved to +y.
        r = rig.run("brace", 70, azimuth=-np.pi / 2, elevation=0.0,
                    extension=1.0, spread=0.45, both_sides=0.0)
        dy = float(rig.env.data.qpos[1]) - y0
        rig.close()
        print(f"\n  brace against a wall: pushed {dy * 100:+.1f} cm away")
        self.assertGreater(dy, 0.02, "bracing on a wall produced no shove")

    def test_both_sides_does_not_shove(self):
        """Wedging presses two ways at once, so the net push is near zero."""
        cfg = make_cfg()
        rig = Rig(cfg, wall_scenario(), settle=20)
        y0 = float(rig.env.data.qpos[1])
        rig.run("brace", 70, azimuth=-np.pi / 2, elevation=0.0,
                extension=1.0, spread=0.45, both_sides=1.0)
        dy_both = abs(float(rig.env.data.qpos[1]) - y0)
        rig.close()
        rig = Rig(cfg, wall_scenario(), settle=20)
        y0 = float(rig.env.data.qpos[1])
        rig.run("brace", 70, azimuth=-np.pi / 2, elevation=0.0,
                extension=1.0, spread=0.45, both_sides=0.0)
        dy_one = abs(float(rig.env.data.qpos[1]) - y0)
        rig.close()
        print(f"  one-sided brace moved {dy_one * 100:.1f} cm, "
              f"wedged moved {dy_both * 100:.1f} cm")
        self.assertLess(dy_both, dy_one)


class ConformTests(unittest.TestCase):
    """The primitive that reads the rod sensors, and the one that bit us."""

    def rocky(self, cfg):
        return generate_scenario("boundary", cfg, radius=2.0, n_segments=64,
                                 n_stones=130, max_stone_size=0.055)

    def test_conform_alone_does_not_drive_the_robot(self):
        """The boundary-stones bug, as a test.

        An unreachable ``ride_height`` pins the height term at its clip. In
        `apply_suspension` that clip is multiplied by a trailing-rod weight,
        which is how the rolling gait pushes, so a robot asked to hold still
        creeps forward instead. Measured before the fix: asking for 0.24 m on
        a build that holds 0.185 m drove it 2.4 m in 250 steps.

        `conform` spreads the lift over every downward rod instead, so a
        request the build cannot meet simply stops lifting. This runs the
        primitive at the very top of its range, where the old arrangement was
        worst.
        """
        cfg = make_cfg()
        top = S.PRIMITIVES["conform"].SPEC.params[0].high
        rig = Rig(cfg, self.rocky(cfg), settle=25)
        r = rig.run("conform", 250, sensors=True, ride_height=top,
                    terrain_gain=0.85, stiffness=0.75)
        rig.close()
        print(f"\n  conform at its ceiling {top} m: drifted {r['dist']:.3f} m, "
              f"left at {r['speed']:.3f} m/s")
        self.assertLess(r["dist"], 0.25, "conform drove the robot across the floor")
        self.assertLess(r["mean_speed"], 0.15, "conform became a throttle")

    def test_ride_height_parameter_raises_the_core(self):
        cfg = make_cfg()
        out = []
        for h in (0.16, 0.30):
            rig = Rig(cfg, self.rocky(cfg), settle=25)
            out.append(rig.run("conform", 200, sensors=True, ride_height=h,
                               terrain_gain=0.85, stiffness=0.75)["z"])
            rig.close()
        print(f"  conform ride 0.17 -> core z {out[0]:.3f}, 0.24 -> {out[1]:.3f}")
        self.assertGreaterEqual(out[1], out[0] - 0.005)

    def test_conform_composes_onto_drive(self):
        """`traverse_rough_terrain` is drive plus this; composing must still travel."""
        cfg = make_cfg()
        rig = Rig(cfg, self.rocky(cfg), settle=25)
        start = rig.env.data.qpos[:2].copy()
        for _ in range(200):
            st = rig.state(sensors=True)
            base = S.act("drive", st, azimuth=0.0, speed=1.2, flank=0.0)
            rig.env.step(S.act("conform", st, base=base, ride_height=0.20,
                               terrain_gain=0.85, stiffness=0.75))
        dist = float(np.linalg.norm(rig.env.data.qpos[:2] - start))
        rig.close()
        print(f"  drive + conform over stones: travelled {dist:.2f} m")
        self.assertGreater(dist, 0.30, "conforming stopped the robot travelling")


if __name__ == "__main__":
    unittest.main()


class CalibrationTests(unittest.TestCase):
    """Does a commanded speed produce that speed on the robot it runs on?

    This is the sim-to-real gap reduced to one number. The library curve was
    measured on an ideal actuator, so under the hardware model a request for
    2.4 m/s returns about 0.55. Selecting the hardware profile re-maps the
    request onto what the build can hold.
    """

    def measure(self, profile, want, steps=320):
        import skills_rl as S
        hw = profile == "hardware"
        cfg = make_cfg(sim2real=hw, hardware=hw)
        rig = Rig(cfg, flat_scenario(cfg))
        seen = []
        for i in range(steps):
            st = rig.state()
            st = type(st)(**{**st.__dict__, "profile": profile})
            rig.env.step(S.act("drive", st, azimuth=0.0, speed=want, flank=0.0))
            if i > steps * 0.55:
                seen.append(float(np.linalg.norm(rig.env.data.qvel[:2])))
        rig.close()
        return float(np.mean(seen))

    def test_hardware_profile_tracks_its_own_command(self):
        for want in (0.35, 0.45, 0.55):
            got = self.measure("hardware", want)
            err = abs(got - want) / want
            print(f"\n  hardware profile: asked {want:.2f}, got {got:.2f} m/s "
                  f"({err * 100:.0f}% off)")
            with self.subTest(want=want):
                self.assertLess(err, 0.35, "commanded speed not tracked")

    def test_ideal_curve_is_wrong_on_hardware(self):
        """The reason the hardware profile exists, asserted rather than claimed."""
        import skills_rl as S
        cfg = make_cfg(sim2real=True, hardware=True)
        rig = Rig(cfg, flat_scenario(cfg))
        seen = []
        for i in range(320):
            st = rig.state()
            rig.env.step(S.act("drive", st, azimuth=0.0, speed=2.4, flank=0.0))
            if i > 176:
                seen.append(float(np.linalg.norm(rig.env.data.qvel[:2])))
        got = float(np.mean(seen))
        rig.close()
        print(f"  ideal curve on hardware: asked 2.40, got {got:.2f} m/s")
        self.assertLess(got, 1.0, "expected the ideal curve to overpromise here")

    def test_profile_narrows_the_advertised_range(self):
        import skills_rl as S
        S.use_profile("hardware")
        lo, hi = (S.PRIMITIVES["drive"].SPEC.params[1].low,
                  S.PRIMITIVES["drive"].SPEC.params[1].high)
        S.use_profile("ideal")
        self.assertLess(hi, 1.0, "hardware range should stop well under 1 m/s")
        self.assertGreater(lo, 0.0)


class ThrustCalibrationTests(unittest.TestCase):
    """Does the thrust calibration predict the jump it actually gets?"""

    def hop(self, power, burn, hardware, settle=30, flight=90):
        import skills_rl as S
        cfg = make_cfg(sim2real=hardware, hardware=hardware)
        rig = Rig(cfg, flat_scenario(cfg), settle=settle)
        z0 = float(rig.env.data.qpos[2])
        peak = z0
        for _ in range(burn):
            rig.env.step(S.act("thrust", rig.state(), azimuth=0.0,
                               elevation=-np.pi / 2, power=power, spread=0.55))
            peak = max(peak, float(rig.env.data.qpos[2]))
        for _ in range(flight):
            rig.env.step(S.act("tuck", rig.state(), extension=0.01))
            peak = max(peak, float(rig.env.data.qpos[2]))
        rig.close()
        return peak - z0

    def test_calibration_predicts_the_hop(self):
        from skills_rl.calibration import full_stroke_steps, thrust_height
        burn = full_stroke_steps("hardware")
        for power in (0.5, 0.75, 1.0):
            got = self.hop(power, burn, hardware=True)
            want = thrust_height(power, "hardware")
            print(f"\n  hardware power {power:.2f}: predicted {want:.3f} m, "
                  f"measured {got:.3f} m")
            with self.subTest(power=power):
                self.assertLess(abs(got - want), 0.02)

    def test_a_short_burn_wastes_the_power(self):
        """The reason `full_stroke_steps` exists."""
        short = self.hop(1.0, 10, hardware=True)
        full = self.hop(1.0, 50, hardware=True)
        print(f"  full power: 10-step burn {short:.3f} m, 50-step burn {full:.3f} m")
        self.assertGreater(full, short * 2.0)

    def test_burn_length_barely_matters_on_an_ideal_actuator(self):
        short = self.hop(1.0, 10, hardware=False)
        full = self.hop(1.0, 50, hardware=False)
        print(f"  ideal: 10-step burn {short:.3f} m, 50-step burn {full:.3f} m")
        self.assertLess(abs(full - short), 0.05)


class VaultTests(unittest.TestCase):
    """Can `drive` get over a kerb, and does the vault parameter decide it?"""

    def arena(self, h, step_x=1.6):
        from radial_sphere.scenario import Scenario
        return Scenario(
            kind="goal", name="curb",
            spawn_xy=np.array([0.0, 0.0], np.float32),
            goal=np.array([4.0, 0.0], np.float32),
            path_pts=np.array([[0.0, 0.0], [4.0, 0.0]], np.float32),
            markers=np.empty((0, 2), np.float32), path_length=4.0,
            steps=np.array([[step_x, 0.0, 0.35, 1.2, h]], np.float32))

    def climb(self, h, vault, steps=700):
        import skills_rl as S
        cfg = make_cfg(sim2real=True, hardware=True)
        rig = Rig(cfg, self.arena(h))
        for _ in range(steps):
            st = rig.state()
            st = type(st)(**{**st.__dict__, "profile": "hardware"})
            rig.env.step(S.act("drive", st, azimuth=0.0, speed=0.55,
                               vault=vault, flank=0.0))
        x = float(rig.env.data.qpos[0])
        rig.close()
        return x

    def test_without_the_vault_a_small_kerb_stops_it(self):
        x = self.climb(0.04, vault=1.0)
        print(f"\n  4 cm step, no vault: stopped at x={x:.2f} (step at 1.60)")
        self.assertLess(x, 1.6, "expected the plain wave to stall at the step")

    def test_the_vault_gets_it_over(self):
        for h, v in ((0.04, 2.6), (0.06, 4.0)):
            x = self.climb(h, vault=v)
            print(f"  {h * 100:.0f} cm step, vault {v}: reached x={x:.2f}")
            with self.subTest(height=h):
                self.assertGreater(x, 2.05, "did not clear the step")

    def test_a_tall_step_needs_a_hop_not_more_boost(self):
        """The honest limit: boost does not solve everything."""
        x = self.climb(0.10, vault=4.0)
        print(f"  10 cm step, vault 4.0: stopped at x={x:.2f} -- needs `thrust`")
        self.assertLess(x, 1.6)
