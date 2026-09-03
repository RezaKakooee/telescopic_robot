"""Analyze geom overlap and gap at max extension (e=0.16m) for multi_stage and zip_chain."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import mujoco
from scratch.test_mechanism_options import generate_bar_xml

u = np.array([0.0, 0.0, 1.0])
for mode in ["multi_stage", "zip_chain"]:
    bx, ax, eq = generate_bar_xml(mode, 0, u)
    eq_block = f"<equality>{eq}</equality>" if eq else ""
    xml = f"""<mujoco model="{mode}">
        <worldbody>
            <body name="core" pos="0 0 1">
                <freejoint name="root"/>
                <geom name="core_geom" type="sphere" size="0.15"/>
                {bx}
            </body>
        </worldbody>
        {eq_block}
        <actuator>{ax}</actuator>
    </mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.ctrl[0] = 0.16 # max extension
    for _ in range(50):
        mujoco.mj_step(m, d)
    
    print(f"\n--- {mode.upper()} AT MAX EXTENSION e=0.16m ---")
    for i in range(m.ngeom):
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        if gname in ["core_geom", "floor"]: continue
        # geom pos in z relative to core
        core_z = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "core")][2]
        gz = d.geom_xpos[i][2] - core_z
        print(f"  Geom '{gname}': center z = {gz:.4f} m (distance from core)")
