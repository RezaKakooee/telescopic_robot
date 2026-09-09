"""Plot a real MuJoCo step response for one RoboBall bar."""

from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from radial_sphere.config import load_config
from radial_sphere.mujoco_env import MujocoRadialSphereEnv


OUT = Path(__file__).parent / "assets" / "pd-step-response.png"
TARGET_CM = 12.0
STEP_TIME = 0.08
DURATION = 0.60


def simulate(kv: float):
    cfg = load_config()
    cfg.robot.kv = kv
    env = MujocoRadialSphereEnv(
        config=cfg,
        randomize=False,
        max_steps=1000,
        render_mode=None,
    )
    env.reset(seed=0)

    # At reset the body frame and world frame are aligned. Choose the bar that
    # points most strongly upward so its foot does not contact the floor.
    bar = int(np.argmax(env.dirs_body[:, 2]))
    joint_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, f"slide_{bar}"
    )
    qpos_adr = int(env.model.jnt_qposadr[joint_id])
    qvel_adr = int(env.model.jnt_dofadr[joint_id])

    targets = np.full(env.n_bars, env.base_ext, dtype=float)
    start_time = float(env.data.time)
    times, commands, positions, velocities = [], [], [], []

    while float(env.data.time) - start_time <= DURATION:
        elapsed = float(env.data.time) - start_time
        target = TARGET_CM / 100.0 if elapsed >= STEP_TIME else env.base_ext
        targets[bar] = target
        env.data.ctrl[:] = targets
        mujoco.mj_step(env.model, env.data)

        times.append(elapsed)
        commands.append(target * 100.0)
        positions.append(float(env.data.qpos[qpos_adr]) * 100.0)
        velocities.append(float(env.data.qvel[qvel_adr]) * 100.0)

    kp = float(cfg.robot.kp)
    env.close()
    return {
        "time": np.asarray(times),
        "target": np.asarray(commands),
        "position": np.asarray(positions),
        "velocity": np.asarray(velocities),
        "bar": bar,
        "kp": kp,
        "kv": kv,
    }


def settling_time(result, tolerance_cm=0.2):
    after = result["time"] >= STEP_TIME
    indices = np.flatnonzero(after)
    error = np.abs(result["position"] - TARGET_CM)
    for index in indices:
        if np.all(error[index:] <= tolerance_cm):
            return float(result["time"][index] - STEP_TIME)
    return None


pd = simulate(kv=22.0)
p_only = simulate(kv=0.0)
pd_settle = settling_time(pd)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})
fig, (ax_pos, ax_vel) = plt.subplots(
    2, 1, figsize=(11, 8), sharex=True,
    gridspec_kw={"height_ratios": [1.35, 1.0]},
)
fig.suptitle(
    "Derivative damping helps a bar reach its target without continued oscillation",
    fontsize=20, fontweight="bold", y=0.98,
)

ax_pos.plot(pd["time"], pd["target"], "--", color="#101828", lw=2.4,
            label="target extension")
ax_pos.plot(p_only["time"], p_only["position"], color="#F79009", lw=2.2,
            label=r"P only: $K_p=900$, $K_v=0$")
ax_pos.plot(pd["time"], pd["position"], color="#175CD3", lw=2.8,
            label=r"PD: $K_p=900$, $K_v=22$")
ax_pos.axhspan(TARGET_CM - 0.2, TARGET_CM + 0.2, color="#12B76A", alpha=0.10,
               label=r"within $\pm0.2$ cm of target")
ax_pos.axvline(STEP_TIME, color="#98A2B3", ls=":", lw=1.8)
ax_pos.text(STEP_TIME + 0.01, 3.2, "target changes", color="#475467",
            fontsize=13, fontweight="bold")
ax_pos.set_ylabel("extension (cm)")
ax_pos.set_title("Target tracking")
ax_pos.grid(alpha=0.25)
ax_pos.legend(loc="lower right", frameon=True)

if pd_settle is not None:
    ax_pos.annotate(
        f"PD settles in {pd_settle:.3f} s",
        xy=(STEP_TIME + pd_settle, TARGET_CM),
        xytext=(0.39, 16.7),
        fontsize=14, color="#175CD3", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#D92D20", lw=2.8),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#175CD3"),
    )

ax_vel.plot(p_only["time"], p_only["velocity"], color="#F79009", lw=2.2,
            label="P only")
ax_vel.plot(pd["time"], pd["velocity"], color="#175CD3", lw=2.8,
            label="PD")
ax_vel.axhline(0.0, color="#98A2B3", lw=1.2)
ax_vel.axvline(STEP_TIME, color="#98A2B3", ls=":", lw=1.8)
ax_vel.set_xlabel("time (s)")
ax_vel.set_ylabel("bar speed (cm/s)")
ax_vel.set_title(r"The derivative term $-K_v\dot e_i$ opposes fast motion")
ax_vel.grid(alpha=0.25)
ax_vel.legend(loc="upper right")

fig.text(
    0.5, 0.01,
    f"MuJoCo response of upward-facing bar {pd['bar']}; target steps from "
    f"{pd['target'][0]:.1f} cm to {TARGET_CM:.1f} cm.",
    ha="center", fontsize=12.5, color="#475467",
)
fig.tight_layout(rect=[0.04, 0.045, 0.98, 0.95])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT)
