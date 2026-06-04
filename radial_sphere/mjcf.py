"""MJCF XML generation for the radial-sphere robot.

    xml, dirs = build_robot_mjcf(n_bars=60)

RoboVerse provides the floor, lighting and skybox via the scenario, so we
only emit the <worldbody>'s single core body + actuators.  The handler adds a
freejoint when fix_base_link=False, so we deliberately do NOT include one here.
"""
from __future__ import annotations

import numpy as np

from .geometry import fibonacci_sphere


def build_robot_mjcf(
    n_bars: int = 60,
    sphere_radius: float = 0.15,
    max_extend: float = 0.12,
    bar_length: float = 0.10,
    sleeve_radius: float = 0.010,
    inner_radius: float = 0.008,
) -> tuple[str, np.ndarray]:
    """Build a self-contained MJCF for the radial-sphere robot.

    Returns:
        xml: MJCF string.
        dirs: (n_bars, 3) unit direction vectors for each bar (body frame).
    """
    dirs = fibonacci_sphere(n_bars)
    bars: list[str] = []
    actuators: list[str] = []
    for k, (ux, uy, uz) in enumerate(dirs):
        u = np.array([ux, uy, uz])
        sleeve_from = (sphere_radius * 0.20) * u
        sleeve_to = (sphere_radius * 0.95) * u
        inner_to = sphere_radius * u
        inner_from = (sphere_radius - bar_length) * u
        bars.append(
            f"""
            <geom name="sleeve_{k}" type="capsule"
                  fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                          {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"
                  size="{sleeve_radius}" rgba="0.35 0.40 0.50 1" mass="0.005"
                  contype="0" conaffinity="0"/>
            <body name="inner_{k}" pos="0 0 0">
                <joint name="slide_{k}" type="slide"
                       axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                       range="0 {max_extend}" armature="0.02"/>
                <geom name="inner_geom_{k}" type="capsule"
                      fromto="{inner_from[0]:.5f} {inner_from[1]:.5f} {inner_from[2]:.5f}
                              {inner_to[0]:.5f}   {inner_to[1]:.5f}   {inner_to[2]:.5f}"
                      size="{inner_radius}" rgba="0.1 0.4 1.0 1" mass="0.01"
                      friction="4.0 0.05 0.002" condim="4"/>
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
                  rgba="1.0 0.85 0.1 1" mass="0.5"/>
            {''.join(bars)}
        </body>
    </worldbody>
    <actuator>
        {''.join(actuators)}
    </actuator>
</mujoco>
"""
    return xml, dirs
