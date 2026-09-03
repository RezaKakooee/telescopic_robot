import mujoco
from scratch.test_mechanism_options import generate_bar_xml
import numpy as np

u = np.array([0, 0, 1.0])
bar_xml, act_xml, eq_xml = generate_bar_xml("multi_stage", 0, u)

full_xml = f"""<mujoco model="test">
    <worldbody>
        <body name="core" pos="0 0 0.5">
            <freejoint name="root"/>
            <geom name="core_geom" type="sphere" size="0.15" mass="0.5"/>
            {bar_xml}
        </body>
    </worldbody>
    <equality>
        {eq_xml}
    </equality>
    <actuator>
        {act_xml}
    </actuator>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(full_xml)
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
print("Multi-stage compiled successfully! nu =", m.nu, "nq =", m.nq)

# Test zip_chain
bar_xml2, act_xml2 = generate_bar_xml("zip_chain", 0, u)
full_xml2 = f"""<mujoco model="test2">
    <worldbody>
        <body name="core" pos="0 0 0.5">
            <freejoint name="root"/>
            <geom name="core_geom" type="sphere" size="0.15" mass="0.5"/>
            {bar_xml2}
        </body>
    </worldbody>
    <actuator>
        {act_xml2}
    </actuator>
</mujoco>"""

m2 = mujoco.MjModel.from_xml_string(full_xml2)
d2 = mujoco.MjData(m2)
mujoco.mj_step(m2, d2)
print("Zip-chain compiled successfully! nu =", m2.nu, "nq =", m2.nq)
