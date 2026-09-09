"""Render high-resolution visual comparison of all 3 rod mechanism architectures:
1. 'single_stage': Baseline long rigid rod
2. 'multi_stage': Concentric telescopic nesting
3. 'zip_chain': Interlocking push-chain / tangential spool drive
"""
import os
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import imageio
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from radial_sphere.geometry import fibonacci_sphere
from scratch.test_mechanism_options import generate_bar_xml


def add_caption_band(frame: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    """Place descriptive text above a render instead of covering the robot."""
    image = Image.fromarray(frame).convert("RGB")
    band_height = 116
    canvas = Image.new("RGB", (image.width, image.height + band_height), "#101828")
    canvas.paste(image, (0, band_height))

    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    title_font = ImageFont.truetype(font_path, 20)
    line_font = ImageFont.truetype(font_path, 13)
    draw.text((14, 10), title, font=title_font, fill="#FFFFFF")
    y = 42
    for line in lines:
        draw.text((14, y), line, font=line_font, fill="#D1E9FF")
        y += 21
    return np.asarray(canvas)
def render_mechanism_scene(mode: str, ext_val: float, width: int = 640, height: int = 480):
    n_bars = 60
    sphere_radius = 0.15
    max_extend = 0.16
    dirs = fibonacci_sphere(n_bars)

    bars_xml = []
    acts_xml = []
    eqs_xml = []

    for k, u in enumerate(dirs):
        bx, ax, eq = generate_bar_xml(mode, k, u, sphere_radius, max_extend)
        bars_xml.append(bx)
        acts_xml.append(ax)
        if eq:
            eqs_xml.append(eq)

    eq_block = f"<equality>{''.join(eqs_xml)}</equality>" if eqs_xml else ""

    xml = f"""<mujoco model="{mode}_showcase">
    <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>
    <visual>
        <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
        <global azimuth="140" elevation="-30"/>
    </visual>
    <worldbody>
        <light pos="0 0 5" dir="0 0 -1" directional="true"/>
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.88 0.92 1"/>
        <body name="core" pos="0 0 0.35">
            <freejoint name="root"/>
            <geom name="core_geom" type="sphere" size="{sphere_radius}"
                  rgba="1.0 0.82 0.15 0.22" mass="0.5"/>
            <!-- Central Electronics & IMU Hub Payload (Visual Marker at Center) -->
            <geom name="avionics_hub" type="sphere" size="0.045"
                  rgba="0.10 0.75 0.90 0.90" mass="0.15" contype="0" conaffinity="0"/>
            {''.join(bars_xml)}
        </body>
    </worldbody>
    {eq_block}
    <actuator>
        {''.join(acts_xml)}
    </actuator>
</mujoco>"""

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Set extension
    data.ctrl[:] = ext_val
    for _ in range(50):
        mujoco.mj_step(model, data)

    # Render tracking camera
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    core_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "core")
    cam.trackbodyid = core_bid
    cam.distance = 0.95
    cam.elevation = -18.0
    cam.azimuth = 35.0
    renderer.update_scene(data, camera=cam)
    raw = renderer.render()
    return raw


def main():
    out_dir = Path("storage_local/mechanism_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [
        ("single_stage", "Baseline: Single Rigid Rod", "Rods overlap through center"),
        ("multi_stage", "Option 1: Multi-Stage Concentric", "Nested inside 6.9cm sleeve (Hub 100% clear)"),
        ("zip_chain", "Option 2: Interlocking Zip-Chain", "Tangential shell cassette (Zero hub intrusion)"),
    ]

    for ext_name, ext_val, ext_desc in [("retracted", 0.02, "Retracted Stance (Stroke 2cm)"), ("extended", 0.14, "Extended Stance (Stroke 14cm)")]:
        frames = []
        for mode, title, note in modes:
            raw = render_mechanism_scene(mode, ext_val)
            lines = [
                f"State: {ext_desc}",
                "Hub electronics: blue sphere at centre",
                f"Clearance: {note}",
            ]
            annotated = add_caption_band(raw, title, lines)
            frames.append(annotated)

        # Stitch horizontally
        combined = np.concatenate(frames, axis=1)
        save_path = out_dir / f"compare_3_mechanisms_{ext_name}.png"
        imageio.imwrite(str(save_path), combined)
        print(f"Saved: {save_path}")
        if ext_name == "extended":
            blog_path = Path("docs/blog/assets/rod-mechanism-comparison.png")
            imageio.imwrite(str(blog_path), combined)
            print(f"Saved blog asset: {blog_path}")

    # Also build a 2x3 comprehensive grid
    img_ret = imageio.imread(str(out_dir / "compare_3_mechanisms_retracted.png"))
    img_ext = imageio.imread(str(out_dir / "compare_3_mechanisms_extended.png"))
    grid_all = np.concatenate([img_ret, img_ext], axis=0)
    final_grid_path = out_dir / "mechanism_3_way_comparison_grid.png"
    imageio.imwrite(str(final_grid_path), grid_all)
    print(f"Saved 3-way comparison grid: {final_grid_path}")


if __name__ == "__main__":
    main()
