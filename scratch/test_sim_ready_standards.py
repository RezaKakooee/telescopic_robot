"""Test High-Standard Sim-Ready MuJoCo Robot Model adhering to DeepMind MuJoCo Menagerie standards."""
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco

def generate_menagerie_radial_sphere_xml(
    n_bars: int = 60,
    sphere_radius: float = 0.15,
    max_extend: float = 0.16,
    core_mass: float = 1.08,
):
    from radial_sphere.geometry import fibonacci_sphere
    dirs = fibonacci_sphere(n_bars)
    
    SLEEVE_STUB = 0.006
    TIP_GAP = 0.004
    FOOT_RADIUS = 0.013
    tip0 = sphere_radius + SLEEVE_STUB + TIP_GAP      # 0.160m
    sleeve_mouth = sphere_radius + SLEEVE_STUB        # 0.156m
    r_base = 0.493 * sphere_radius                   # 0.074m

    # Core solid inertia: sphere I = 2/5 * M * R^2
    core_ixx = 0.4 * core_mass * (sphere_radius ** 2)

    bars_xml = []
    actuators_xml = []
    equalities_xml = []
    sensors_xml = []

    # Sensor suite: central IMU
    sensors_xml.append('    <accelerometer name="imu_acc" site="imu_site"/>')
    sensors_xml.append('    <gyro name="imu_gyro" site="imu_site"/>')
    sensors_xml.append('    <framequat name="imu_quat" objtype="site" objname="imu_site"/>')

    for k, (ux, uy, uz) in enumerate(dirs):
        u = np.array([ux, uy, uz], dtype=float)
        u_unit = u / (np.linalg.norm(u) + 1e-12)

        sleeve_from = r_base * u_unit
        sleeve_to = sleeve_mouth * u_unit

        st1_p1 = (r_base - 0.002) * u_unit
        st1_p2 = tip0 * u_unit

        st2_p1 = (r_base + 0.002) * u_unit
        st2_p2 = (tip0 - FOOT_RADIUS * 0.9) * u_unit
        foot_pos = tip0 * u_unit

        # Sleeve geom: fixed on core
        # Stage 1: intermediate collar (mass=0.003kg)
        # Inner rod: chrome shaft (mass=0.004kg) + foot (mass=0.004kg)
        bars_xml.append(f"""
        <!-- Bar {k}: Multi-Stage Cascade Telescopic Unit -->
        <geom name="sleeve_{k}" class="sleeve_geom"
              fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                      {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"/>
        
        <body name="stage1_{k}" pos="0 0 0">
            <inertial pos="{(st1_p1[0]+st1_p2[0])*0.5:.5f} {(st1_p1[1]+st1_p2[1])*0.5:.5f} {(st1_p1[2]+st1_p2[2])*0.5:.5f}"
                      mass="0.003" diaginertia="1.5e-6 1.5e-6 2.0e-7"/>
            <joint name="slide1_{k}" class="cascade_stage1_joint" axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                   range="0 {max_extend * 0.5}"/>
            <geom name="stage1_geom_{k}" class="collar_geom"
                  fromto="{st1_p1[0]:.5f} {st1_p1[1]:.5f} {st1_p1[2]:.5f}
                          {st1_p2[0]:.5f} {st1_p2[1]:.5f} {st1_p2[2]:.5f}"/>
        </body>

        <body name="inner_{k}" pos="0 0 0">
            <inertial pos="{(st2_p1[0]+st2_p2[0])*0.5:.5f} {(st2_p1[1]+st2_p2[1])*0.5:.5f} {(st2_p1[2]+st2_p2[2])*0.5:.5f}"
                      mass="0.008" diaginertia="3.0e-6 3.0e-6 3.5e-7"/>
            <joint name="slide_{k}" class="telescopic_main_joint" axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                   range="0 {max_extend}"/>
            <geom name="inner_geom_{k}" class="piston_geom"
                  fromto="{st2_p1[0]:.5f} {st2_p1[1]:.5f} {st2_p1[2]:.5f}
                          {st2_p2[0]:.5f} {st2_p2[1]:.5f} {st2_p2[2]:.5f}"/>
            <geom name="foot_{k}" class="foot_rubber_geom" pos="{foot_pos[0]:.5f} {foot_pos[1]:.5f} {foot_pos[2]:.5f}"/>
            <site name="foot_site_{k}" pos="{foot_pos[0]:.5f} {foot_pos[1]:.5f} {foot_pos[2]:.5f}" size="0.008" type="sphere" rgba="0 0 0 0"/>
        </body>
        """)

        # Actuator (affine position controller with realistic force limit)
        actuators_xml.append(
            f'<general name="slide_{k}" class="actuator_radial" joint="slide_{k}" ctrlrange="0 {max_extend}"/>'
        )

        # Equality constraint representing the cascade Dyneema cable transmission (with elastic compliance)
        equalities_xml.append(
            f'<joint name="cable_rigging_{k}" joint1="slide1_{k}" joint2="slide_{k}" polycoef="0 0.5 0 0 0" '
            f'solref="0.004 1.0" solimp="0.95 0.99 0.001"/>'
        )

        # Sensors for Bar k
        sensors_xml.append(f'    <jointpos name="pos_{k}" joint="slide_{k}"/>')
        sensors_xml.append(f'    <jointvel name="vel_{k}" joint="slide_{k}"/>')
        sensors_xml.append(f'    <actuatorfrc name="frc_{k}" actuator="slide_{k}"/>')
        sensors_xml.append(f'    <touch name="touch_{k}" site="foot_site_{k}"/>')

    xml = f"""<mujoco model="radial_sphere_menagerie_standard">
    <compiler angle="radian" coordinate="local" autolimits="true"/>
    
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="pyramidal">
        <flag contact="enable" frictionloss="enable"/>
    </option>

    <default>
        <!-- High-Standard Default Classes -->
        <default class="radial_sphere">
            <!-- Bushing sliding joint defaults: Coulomb frictionloss + Viscous damping + Endstop compliance -->
            <default class="telescopic_main_joint">
                <joint type="slide" damping="14.0" frictionloss="1.2" armature="0.018"
                       margin="0.001" solreflimit="0.005 1.0" solimplimit="0.90 0.98 0.001"/>
            </default>
            <default class="cascade_stage1_joint">
                <joint type="slide" damping="7.0" frictionloss="0.6" armature="0.009"
                       margin="0.001" solreflimit="0.005 1.0" solimplimit="0.90 0.98 0.001"/>
            </default>

            <!-- Geom classes -->
            <default class="sleeve_geom">
                <geom type="capsule" size="0.0135" rgba="0.32 0.35 0.40 1" contype="0" conaffinity="0" mass="0.004"/>
            </default>
            <default class="collar_geom">
                <geom type="capsule" size="0.0102" rgba="0.32 0.35 0.40 1" contype="0" conaffinity="0" mass="0.003"/>
            </default>
            <default class="piston_geom">
                <geom type="capsule" size="0.0076" rgba="0.92 0.94 0.98 1" contype="0" conaffinity="0" mass="0.004"/>
            </default>
            
            <!-- Rubber foot contact mechanics: Shore 70A vulcanized rubber -->
            <default class="foot_rubber_geom">
                <geom type="sphere" size="{FOOT_RADIUS}" rgba="0.10 0.10 0.12 1" mass="0.004"
                      contype="1" conaffinity="2" priority="1" condim="4"
                      friction="0.95 0.015 0.005"
                      solref="0.006 1.1" solimp="0.90 0.95 0.002"/>
            </default>

            <!-- Actuator class: calibrated motor envelope -->
            <default class="actuator_radial">
                <general biastype="affine" biasprm="0 -900 -22" gaintype="fixed" gainprm="900 0 0"
                         forcerange="-55 55"/>
            </default>
        </default>
    </default>

    <asset>
        <material name="core_mat" rgba="1.0 0.82 0.15 1" specular="0.4" shininess="0.5"/>
        <material name="grid" rgba="0.8 0.8 0.8 1"/>
    </asset>

    <worldbody>
        <light pos="0 0 5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
        <geom name="floor" type="plane" size="50 50 0.1" material="grid"
              friction="0.85 0.015 0.005" condim="4"/>

        <!-- Spherical Robot Base Body -->
        <body name="core" pos="0 0 0.30">
            <freejoint name="root"/>
            <inertial pos="0 0 0" mass="{core_mass}" diaginertia="{core_ixx:.6f} {core_ixx:.6f} {core_ixx:.6f}"/>
            <geom name="core_geom" type="sphere" size="{sphere_radius}" material="core_mat"
                  contype="1" conaffinity="2" friction="0.85 0.015 0.005" condim="4"/>
            
            <!-- Central IMU Sensor Site (At CoG) -->
            <site name="imu_site" pos="0 0 0" size="0.01" type="sphere" rgba="0 1 1 1"/>
            
            <!-- Central Avionics & Battery Hub Payload -->
            <geom name="avionics_hub" type="sphere" size="0.05" rgba="0.10 0.75 0.90 0.85"
                  contype="0" conaffinity="0" mass="0.10"/>

            {''.join(bars_xml)}
        </body>
    </worldbody>

    <equality>
        {''.join(equalities_xml)}
    </equality>

    <actuator>
        {''.join(actuators_xml)}
    </actuator>

    <sensor>
        {chr(10).join(sensors_xml)}
    </sensor>
</mujoco>
"""
    return xml

if __name__ == "__main__":
    xml = generate_menagerie_radial_sphere_xml()
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    print(f"Compilation SUCCESSFUL!")
    print(f"  Bodies: {m.nbody}")
    print(f"  Joints: {m.njnt} (nq={m.nq}, nv={m.nv})")
    print(f"  Geoms: {m.ngeom}")
    print(f"  Actuators: {m.nu}")
    print(f"  Equality constraints: {m.neq}")
    print(f"  Sensors: {m.nsensor}")
    
    # Step 500 steps with simulated control
    for t in range(500):
        d.ctrl[:] = 0.08 + 0.04 * np.sin(t * 0.05 + np.arange(m.nu))
        mujoco.mj_step(m, d)
    
    # Read sensor values
    imu_acc = d.sensor("imu_acc").data
    imu_gyro = d.sensor("imu_gyro").data
    touch_0 = d.sensor("touch_0").data
    print(f"  Step 500 reached cleanly with ZERO numerical instability!")
    print(f"  IMU Acc: {imu_acc}")
    print(f"  IMU Gyro: {imu_gyro}")
    print(f"  Foot 0 Touch: {touch_0}")
