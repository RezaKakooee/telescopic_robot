import os, time, pathlib

p = pathlib.Path("/home/azureuser/telescopic_robot/storage_local")
exclude = {"sci_out", "_assets", "_smoke_level3", "_smoke_level3_render", "_smoke_level3_step"}
rename_map = {}

for d in sorted(p.iterdir()):
    if d.is_dir() and not d.name.startswith("") and d.name not in exclude and not d.name.startswith("2026"):
        mtime = d.stat().st_mtime
        d_str = time.strftime("%Y%m%d", time.localtime(mtime))
        new_name = f"{d_str}__{d.name}"
        rename_map[d.name] = new_name
        target = p / new_name
        d.rename(target)
        print(f"Renamed: {d.name} -> {new_name}")

print(f"\nTotal renamed: {len(rename_map)}")
print("Updating references in codebase...")
root = pathlib.Path("/home/azureuser/telescopic_robot")
for target_dir in ["docs", "scratch", "configs", "radial_sphere"]:
    dir_path = root / target_dir
    if not dir_path.exists():
        continue
    for fpath in dir_path.rglob("*"):
        if fpath.is_file() and fpath.suffix in [".py", ".md", ".yaml", ".json", ".sh", ".txt"]:
            try:
                content = fpath.read_text(encoding="utf-8")
                modified = False
                for old_n, new_n in rename_map.items():
                    old_str = f"storage_local/{old_n}"
                    new_str = f"storage_local/{new_n}"
                    if old_str in content:
                        content = content.replace(old_str, new_str)
                        modified = True
                if modified:
                    fpath.write_text(content, encoding="utf-8")
                    print(f"Updated references in: {fpath.relative_to(root)}")
            except Exception as e:
                print(f"Error reading/writing {fpath}: {e}")

print("All done successfully!")
