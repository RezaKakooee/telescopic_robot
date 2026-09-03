import argparse
import os
import mujoco
import numpy as np

def test_cylinder_geometry():
    """
    Generate a simple script to verify the geometry of the motordrome.
    """
    xml = """
    <mujoco>
        <compiler angle="degree" coordinate="local"/>
        <worldbody>
            <body name="test_apron_body" pos="2 0 0.4" euler="0 0 90">
                <geom name="test_apron_geom" type="box" size="0.4 0.1 0.01" euler="0 -45 0" rgba="1 0 0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_apron_geom")
    pos = data.geom_xpos[geom_id]
    mat = data.geom_xmat[geom_id].reshape(3, 3)
    print(f"Geom test_apron_geom at {pos}")
    print(f"Normal vector (Z-axis of box): {mat[:, 2]}")
    print(f"Long axis (X-axis of box): {mat[:, 0]}")

if __name__ == "__main__":
    test_cylinder_geometry()
