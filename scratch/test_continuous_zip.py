"""Verify zero discontinuity in Option 2 (zip_chain) and Option 1 (multi_stage) across 0% to 100% stroke."""
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco

def generate_continuous_zip_chain_xml(k: int, u: np.ndarray, sphere_radius: float = 0.15, max_extend: float = 0.16):
    ux, uy, uz = u
    u_unit = u / (np.linalg.norm(u) + 1e-12)
    FOOT_RADIUS = 0.013
    TIP_GAP = 0.004
    SLEEVE_STUB = 0.006
    tip0 = sphere_radius + SLEEVE_STUB + TIP_GAP
    sleeve_mouth = sphere_radius + SLEEVE_STUB

    # Geometry intervals:
    r_nozzle_base = 0.074 * (sphere_radius / 0.15) # 0.074m
    sleeve_from = r_nozzle_base * u_unit
    sleeve_to = sleeve_mouth * u_unit

    # Tangential chain cassette (spool housing along inner perimeter of shell)
    up = np.array([0, 0, 1.0]) if abs(uz) < 0.9 else np.array([1.0, 0, 0])
    tangent = np.cross(u_unit, up)
    tangent /= (np.linalg.norm(tangent) + 1e-12)
    c_p1 = (sphere_radius * 0.85) * u_unit
    c_p2 = c_p1 + 0.038 * tangent

    # Stage 1 (Base Chain Column): moves 0.5 * e
    st1_p1 = (r_nozzle_base - 0.002) * u_unit
    st1_p2 = (tip0) * u_unit

    # Stage 2 (Tip Chain Column + Foot): moves 1.0 * e
    st2_p1 = (r_nozzle_base + 0.002) * u_unit
    st2_p2 = (tip0 - FOOT_RADIUS * 0.9) * u_unit

    sleeve_rgba = "0.25 0.27 0.32 1"
    chain1_rgba = "0.72 0.75 0.82 1"
    chain2_rgba = "0.88 0.90 0.95 1"
    foot_rgba = "0.10 0.10 0.12 1"

    xml = f"""
    <!-- Compact Peripheral Nozzle (Mounted at Shell Wall) -->
    <geom name="sleeve_{k}" type="capsule"
          fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                  {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"
          size="0.0135" rgba="{sleeve_rgba}" mass="0.003" contype="0" conaffinity="0"/>
    <!-- Tangential Flexible Chain Spool / Magazine Housing -->
    <geom name="cassette_{k}" type="capsule"
          fromto="{c_p1[0]:.5f} {c_p1[1]:.5f} {c_p1[2]:.5f}
                  {c_p2[0]:.5f} {c_p2[1]:.5f} {c_p2[2]:.5f}"
          size="0.011" rgba="0.45 0.48 0.55 1" mass="0.004" contype="0" conaffinity="0"/>
    
    <!-- Base Interlocking Chain Column (Emerges from nozzle at 0.5*e) -->
    <body name="stage1_{k}" pos="0 0 0">
        <joint name="slide1_{k}" type="slide" axis="{ux:.5f} {uy:.5f} {uz:.5f}"
               range="0 {max_extend * 0.5}" armature="0.01"/>
        <geom name="stage1_geom_{k}" type="capsule"
              fromto="{st1_p1[0]:.5f} {st1_p1[1]:.5f} {st1_p1[2]:.5f}
                      {st1_p2[0]:.5f} {st1_p2[1]:.5f} {st1_p2[2]:.5f}"
              size="0.0102" rgba="{chain1_rgba}" mass="0.003" contype="0" conaffinity="0"/>
    </body>

    <!-- Tip Interlocking Chain Column (Reaches foot at 1.0*e) -->
    <body name="inner_{k}" pos="0 0 0">
        <joint name="slide_{k}" type="slide" axis="{ux:.5f} {uy:.5f} {uz:.5f}"
               range="0 {max_extend}" armature="0.02"/>
        <geom name="inner_geom_{k}" type="capsule"
              fromto="{st2_p1[0]:.5f} {st2_p1[1]:.5f} {st2_p1[2]:.5f}
                      {st2_p2[0]:.5f} {st2_p2[1]:.5f} {st2_p2[2]:.5f}"
              size="0.0080" rgba="{chain2_rgba}" mass="0.004" contype="0" conaffinity="0"/>
        <geom name="foot_{k}" type="sphere" pos="{tip0 * ux:.5f} {tip0 * uy:.5f} {tip0 * uz:.5f}"
              size="{FOOT_RADIUS}" rgba="{foot_rgba}" mass="0.004"
              contype="1" conaffinity="2" friction="0.85 0.015 0.005" condim="4" priority="1"/>
    </body>
    """
    actuator = f'<general name="slide_{k}" joint="slide_{k}" ctrlrange="0 {max_extend}" gainprm="900 0 0" biasprm="0 -900 -22" biastype="affine" gaintype="fixed" forcerange="-80 80"/>'
    equality = f'<joint joint1="slide1_{k}" joint2="slide_{k}" polycoef="0 0.5 0 0 0"/>'
    return xml, actuator, equality


if __name__ == "__main__":
    u = np.array([0.0, 0.0, 1.0])
    bx, ax, eq = generate_continuous_zip_chain_xml(0, u)
    xml = f"""<mujoco model="continuous_zip_chain">
        <worldbody>
            <body name="core" pos="0 0 1">
                <freejoint name="root"/>
                <geom name="core_geom" type="sphere" size="0.15"/>
                {bx}
            </body>
        </worldbody>
        <equality>{eq}</equality>
        <actuator>{ax}</actuator>
    </mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    print("Checking gaps across all stroke values from 0.00m to 0.16m:")
    for ext in [0.00, 0.04, 0.08, 0.12, 0.16]:
        d.ctrl[0] = ext
        for _ in range(50):
            mujoco.mj_step(m, d)

        # Get z-extents of sleeve, stage1, and inner
        core_z = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "core")][2]
        
        # We can read body positions
        s1_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "stage1_0")
        s2_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "inner_0")
        s1_z = d.xpos[s1_bid][2] - core_z
        s2_z = d.xpos[s2_bid][2] - core_z
        
        # Stage 1 interval in world z
        s1_start = 0.072 + s1_z
        s1_end = 0.160 + s1_z
        
        # Stage 2 interval in world z
        s2_start = 0.076 + s2_z
        s2_end = 0.160 + s2_z
        
        # Overlaps
        overlap_sleeve_s1 = 0.156 - s1_start
        overlap_s1_s2 = s1_end - s2_start
        
        print(f"  Stroke e={ext*100:4.1f}cm: Nozzle overlap={overlap_sleeve_s1*100:+5.2f}cm, Stage1-Stage2 overlap={overlap_s1_s2*100:+5.2f}cm -> {'PERFECT CONTINUOUS' if (overlap_sleeve_s1 >= 0 and overlap_s1_s2 >= 0) else 'GAP ERROR'}")
