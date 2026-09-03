"""Geometry helper with 100% continuous overlap across all 3 rod mechanisms."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco

def generate_bar_xml(mode: str, k: int, u: np.ndarray, sphere_radius: float = 0.15, max_extend: float = 0.16):
    ux, uy, uz = u
    u_unit = u / (np.linalg.norm(u) + 1e-12)
    FOOT_RADIUS = 0.013
    TIP_GAP = 0.004
    SLEEVE_STUB = 0.006
    tip0 = sphere_radius + SLEEVE_STUB + TIP_GAP
    sleeve_mouth = sphere_radius + SLEEVE_STUB
    sleeve_radius = 0.012
    inner_radius = 0.008

    if mode == "single_stage":
        bar_len = max_extend + TIP_GAP + 0.35 * sphere_radius
        sleeve_from = (0.55 * sphere_radius) * u_unit
        sleeve_to = sleeve_mouth * u_unit
        rod_from = (tip0 - bar_len) * u_unit
        rod_to = (tip0 - FOOT_RADIUS * 0.9) * u_unit
        foot = tip0 * u_unit

        xml = f"""
        <geom name="sleeve_{k}" type="capsule"
              fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                      {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"
              size="{sleeve_radius}" rgba="0.38 0.42 0.48 1" mass="0.005" contype="0" conaffinity="0"/>
        <body name="inner_{k}" pos="0 0 0">
            <joint name="slide_{k}" type="slide" axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                   range="0 {max_extend}" armature="0.02"/>
            <geom name="inner_geom_{k}" type="capsule"
                  fromto="{rod_from[0]:.5f} {rod_from[1]:.5f} {rod_from[2]:.5f}
                          {rod_to[0]:.5f}   {rod_to[1]:.5f}   {rod_to[2]:.5f}"
                  size="{inner_radius}" rgba="0.88 0.90 0.94 1" mass="0.008" contype="0" conaffinity="0"/>
            <geom name="foot_{k}" type="sphere" pos="{foot[0]:.5f} {foot[1]:.5f} {foot[2]:.5f}"
                  size="{FOOT_RADIUS}" rgba="0.10 0.10 0.12 1" mass="0.004"
                  contype="1" conaffinity="2" friction="0.85 0.015 0.005" condim="4" priority="1"/>
        </body>
        """
        actuator = f'<general name="slide_{k}" joint="slide_{k}" ctrlrange="0 {max_extend}" gainprm="900 0 0" biasprm="0 -900 -22" biastype="affine" gaintype="fixed" forcerange="-80 80"/>'
        return xml, actuator, ""

    elif mode in ["multi_stage", "concentric_telescopic"]:
        r_base = 0.493 * sphere_radius
        sleeve_from = r_base * u_unit
        sleeve_to = sleeve_mouth * u_unit

        st1_p1 = (r_base - 0.002) * u_unit
        st1_p2 = tip0 * u_unit
        st2_p1 = (r_base + 0.002) * u_unit
        st2_p2 = (tip0 - FOOT_RADIUS * 0.9) * u_unit

        xml = f"""
        <geom name="sleeve_{k}" type="capsule"
              fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                      {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"
              size="{sleeve_radius * 1.10:.5f}" rgba="0.32 0.35 0.40 1" mass="0.004" contype="0" conaffinity="0"/>
        <body name="stage1_{k}" pos="0 0 0">
            <joint name="slide1_{k}" type="slide" axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                   range="0 {max_extend * 0.5}" armature="0.01"/>
            <geom name="stage1_geom_{k}" type="capsule"
                  fromto="{st1_p1[0]:.5f} {st1_p1[1]:.5f} {st1_p1[2]:.5f}
                          {st1_p2[0]:.5f} {st1_p2[1]:.5f} {st1_p2[2]:.5f}"
                  size="{sleeve_radius * 0.85:.5f}" rgba="0.32 0.35 0.40 1" mass="0.003" contype="0" conaffinity="0"/>
        </body>
        <body name="inner_{k}" pos="0 0 0">
            <joint name="slide_{k}" type="slide" axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                   range="0 {max_extend}" armature="0.02"/>
            <geom name="inner_geom_{k}" type="capsule"
                  fromto="{st2_p1[0]:.5f} {st2_p1[1]:.5f} {st2_p1[2]:.5f}
                          {st2_p2[0]:.5f} {st2_p2[1]:.5f} {st2_p2[2]:.5f}"
                  size="{inner_radius * 0.95:.5f}" rgba="0.92 0.94 0.98 1" mass="0.004" contype="0" conaffinity="0"/>
            <geom name="foot_{k}" type="sphere" pos="{tip0 * ux:.5f} {tip0 * uy:.5f} {tip0 * uz:.5f}"
                  size="{FOOT_RADIUS}" rgba="0.10 0.10 0.12 1" mass="0.004"
                  contype="1" conaffinity="2" friction="0.85 0.015 0.005" condim="4" priority="1"/>
        </body>
        """
        actuator = f'<general name="slide_{k}" joint="slide_{k}" ctrlrange="0 {max_extend}" gainprm="900 0 0" biasprm="0 -900 -22" biastype="affine" gaintype="fixed" forcerange="-80 80"/>'
        equality = f'<joint joint1="slide1_{k}" joint2="slide_{k}" polycoef="0 0.5 0 0 0"/>'
        return xml, actuator, equality

    elif mode in ["zip_chain", "push_chain"]:
        r_base = 0.493 * sphere_radius
        sleeve_from = r_base * u_unit
        sleeve_to = sleeve_mouth * u_unit
        up = np.array([0, 0, 1.0]) if abs(uz) < 0.9 else np.array([1.0, 0, 0])
        tangent = np.cross(u_unit, up)
        tangent /= (np.linalg.norm(tangent) + 1e-12)
        c_p1 = (sphere_radius * 0.85) * u_unit
        c_p2 = c_p1 + 0.038 * tangent

        st1_p1 = (r_base - 0.002) * u_unit
        st1_p2 = tip0 * u_unit
        st2_p1 = (r_base + 0.002) * u_unit
        st2_p2 = (tip0 - FOOT_RADIUS * 0.9) * u_unit

        xml = f"""
        <geom name="sleeve_{k}" type="capsule"
              fromto="{sleeve_from[0]:.5f} {sleeve_from[1]:.5f} {sleeve_from[2]:.5f}
                      {sleeve_to[0]:.5f}   {sleeve_to[1]:.5f}   {sleeve_to[2]:.5f}"
              size="{sleeve_radius * 1.15:.5f}" rgba="0.25 0.27 0.32 1" mass="0.003" contype="0" conaffinity="0"/>
        <geom name="cassette_{k}" type="capsule"
              fromto="{c_p1[0]:.5f} {c_p1[1]:.5f} {c_p1[2]:.5f}
                      {c_p2[0]:.5f} {c_p2[1]:.5f} {c_p2[2]:.5f}"
              size="{sleeve_radius * 0.92:.5f}" rgba="0.45 0.48 0.55 1" mass="0.004" contype="0" conaffinity="0"/>
        <body name="stage1_{k}" pos="0 0 0">
            <joint name="slide1_{k}" type="slide" axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                   range="0 {max_extend * 0.5}" armature="0.01"/>
            <geom name="stage1_geom_{k}" type="capsule"
                  fromto="{st1_p1[0]:.5f} {st1_p1[1]:.5f} {st1_p1[2]:.5f}
                          {st1_p2[0]:.5f} {st1_p2[1]:.5f} {st1_p2[2]:.5f}"
                  size="{sleeve_radius * 0.85:.5f}" rgba="0.70 0.73 0.80 1" mass="0.003" contype="0" conaffinity="0"/>
        </body>
        <body name="inner_{k}" pos="0 0 0">
            <joint name="slide_{k}" type="slide" axis="{ux:.5f} {uy:.5f} {uz:.5f}"
                   range="0 {max_extend}" armature="0.02"/>
            <geom name="inner_geom_{k}" type="capsule"
                  fromto="{st2_p1[0]:.5f} {st2_p1[1]:.5f} {st2_p1[2]:.5f}
                          {st2_p2[0]:.5f} {st2_p2[1]:.5f} {st2_p2[2]:.5f}"
                  size="{inner_radius * 1.05:.5f}" rgba="0.90 0.92 0.96 1" mass="0.007" contype="0" conaffinity="0"/>
            <geom name="foot_{k}" type="sphere" pos="{tip0 * ux:.5f} {tip0 * uy:.5f} {tip0 * uz:.5f}"
                  size="{FOOT_RADIUS}" rgba="0.10 0.10 0.12 1" mass="0.004"
                  contype="1" conaffinity="2" friction="0.85 0.015 0.005" condim="4" priority="1"/>
        </body>
        """
        actuator = f'<general name="slide_{k}" joint="slide_{k}" ctrlrange="0 {max_extend}" gainprm="900 0 0" biasprm="0 -900 -22" biastype="affine" gaintype="fixed" forcerange="-80 80"/>'
        equality = f'<joint joint1="slide1_{k}" joint2="slide_{k}" polycoef="0 0.5 0 0 0"/>'
        return xml, actuator, equality
