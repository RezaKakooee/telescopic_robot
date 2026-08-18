"""MJCF XML generation for the radial-sphere robot.

    xml, dirs = build_robot_mjcf(n_bars=60)

RoboVerse provides the floor, lighting and skybox via the scenario, so we
only emit the <worldbody>'s single core body + actuators.  The handler adds a
freejoint when fix_base_link=False, so we deliberately do NOT include one here.

Each bar is a telescopic unit built from three geoms:

    sleeve  — fixed guide tube on the core, ending in a near-flush port at the
              sphere surface (``SLEEVE_STUB``);
    rod     — the sliding inner capsule, long enough that its rear end never
              leaves the sleeve at full extension (it always looks connected);
    foot    — a rounded cap on the rod tip that makes ground contact.

At zero extension a bar sinks fully into the ball — only its coloured foot
hugs the surface — and at full extension the rod reaches out almost a whole
ball radius, so the open/close travel is maximally visible.  Each bar has its
own hue so it can be tracked through that cycle in videos.

Only the core sphere and the feet collide; sleeves and rods are visual-only,
which keeps contacts on rounded, well-conditioned geometry.  The feet carry
explicit contact parameters with ``priority="1"``: MuJoCo's default contact
stiffness scales with the touching body's effective mass, and a few-gram foot
would otherwise sink centimetres into the floor under the robot's weight;
without the priority flag MuJoCo would average our stiff solref/solimp (and
friction) with the floor's soft defaults instead of using them.
"""
from __future__ import annotations

import colorsys

import numpy as np

from .geometry import fibonacci_sphere

SLEEVE_STUB = 0.006   # sleeve port protrusion beyond the sphere surface (m);
                      # near-flush so a retracted bar sinks into the ball
TIP_GAP = 0.004       # foot centre clearance past the sleeve mouth at zero extension (m)
FOOT_RADIUS = 0.013   # rounded contact foot at the rod tip (m)


def rolling_radius(sphere_radius: float, extension: float) -> float:
    """Core centre → outermost point of a bar's foot at the given extension.

    The effective "wheel" radius of the robot; the env uses it to spawn the
    sphere just above the floor.
    """
    return sphere_radius + SLEEVE_STUB + TIP_GAP + extension + FOOT_RADIUS


def build_robot_mjcf(
    n_bars: int = 60,
    sphere_radius: float = 0.15,
    max_extend: float = 0.12,
    bar_length: float | None = None,
    sleeve_radius: float = 0.012,
    inner_radius: float = 0.008,
) -> tuple[str, np.ndarray]:
    """Build a self-contained MJCF for the radial-sphere robot.

    Args:
        bar_length: sliding rod length; default keeps the rod's rear end inside
            the sleeve at full extension (so the telescope never comes apart).

    Returns:
        xml: MJCF string.
        dirs: (n_bars, 3) unit direction vectors for each bar (body frame).
    """
    dirs = fibonacci_sphere(n_bars)
    tip0 = sphere_radius + SLEEVE_STUB + TIP_GAP      # foot centre at zero extension
    sleeve_mouth = sphere_radius + SLEEVE_STUB
    if bar_length is None:
        bar_length = max_extend + TIP_GAP + 0.35 * sphere_radius
    assert bar_length >= max_extend + TIP_GAP, \
        "rod would fully exit the sleeve at max extension"

    bars: list[str] = []
    actuators: list[str] = []
    for k, (ux, uy, uz) in enumerate(dirs):
        u = np.array([ux, uy, uz])
        # Per-bar hue so individual bars are trackable in videos: with uniform
        # colors the extension pattern (long at back, short at front) is fixed
        # relative to the chase camera and telescoping reads as a rigid ball.
        rr, gg, bb = colorsys.hsv_to_rgb(k / n_bars, 0.90, 1.00)
        fr, fg, fb = colorsys.hsv_to_rgb(k / n_bars, 0.90, 0.65)
        sleeve_from = (0.55 * sphere_radius) * u
        sleeve_to = sleeve_mouth * u
        rod_to = tip0 * u
        rod_from = (tip0 - bar_length) * u
        foot = tip0 * u
        bars.append(
            f"""
            <geom name="sleeve_{k}" type="capsule"
                  fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                          {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"
                  size="{sleeve_radius}" rgba="1.0 0.82 0.15 1" mass="0.005"
                  contype="0" conaffinity="0"/>
            <body name="inner_{k}" pos="0 0 0">
                <joint name="slide_{k}" type="slide"
                       axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                       range="0 {max_extend}" armature="0.02"/>
                <geom name="inner_geom_{k}" type="capsule"
                      fromto="{rod_from[0]:.5f} {rod_from[1]:.5f} {rod_from[2]:.5f}
                              {rod_to[0]:.5f}   {rod_to[1]:.5f}   {rod_to[2]:.5f}"
                      size="{inner_radius}" rgba="{rr:.3f} {gg:.3f} {bb:.3f} 1" mass="0.008"
                      contype="0" conaffinity="0"/>
                <geom name="foot_{k}" type="sphere"
                      pos="{foot[0]:.5f} {foot[1]:.5f} {foot[2]:.5f}"
                      size="{FOOT_RADIUS}" rgba="{fr:.3f} {fg:.3f} {fb:.3f} 1" mass="0.004"
                      friction="4.0 0.05 0.002" condim="4" priority="1"
                      solref="0.005 1" solimp="0.95 0.99 0.001"/>
            </body>
            """
        )
        # RoboVerse's mujoco handler looks up actuators by JOINT name, so the
        # actuator name must equal the joint name (slide_k).
        actuators.append(
            f'<position name="slide_{k}" joint="slide_{k}" '
            f'ctrlrange="0 {max_extend}" forcerange="-80 80"/>'
        )

    xml = f"""<mujoco model="radial_sphere">
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
    <worldbody>
        <body name="core" pos="0 0 0">
            <geom name="core_geom" type="sphere" size="{sphere_radius}"
                  rgba="1.0 0.82 0.15 1" mass="0.5"/>
            {''.join(bars)}
        </body>
    </worldbody>
    <actuator>
        {''.join(actuators)}
    </actuator>
</mujoco>
"""
    return xml, dirs
