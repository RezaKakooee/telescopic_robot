import os, pathlib

p = pathlib.Path("/home/azureuser/telescopic_robot/storage_local")
rename_map = {}

# 1. Rename directories in storage_local
for d in sorted(p.iterdir()):
    if d.is_dir() and d.name.startswith(""):
        new_name = d.name[8:] # strip ''
        rename_map[d.name] = new_name
        target = p / new_name
        d.rename(target)
        print(f"Renamed dir: {d.name} -> {new_name}")

# 2. Rename files in storage_local/sci_out
sci_out = p / "sci_out"
if sci_out.exists():
    for f in sorted(sci_out.iterdir()):
        if f.is_file() and f.name.startswith(""):
            new_name = f.name[8:]
            rename_map[f.name] = new_name
            target = sci_out / new_name
            f.rename(target)
            print(f"Renamed file: {f.name} -> {new_name}")

print(f"\nTotal items renamed: {len(rename_map)}")
print("Updating all references in repository...")

root = pathlib.Path("/home/azureuser/telescopic_robot")
for target_dir in ["docs", "scratch", "configs", "radial_sphere", "scripts", "ops", "notes"]:
    dir_path = root / target_dir
    if not dir_path.exists():
        continue
    for fpath in dir_path.rglob("*"):
        if fpath.is_file() and fpath.suffix in [".py", ".md", ".yaml", ".json", ".sh", ".txt"]:
            try:
                content = fpath.read_text(encoding="utf-8")
                modified = False
                for old_n, new_n in rename_map.items():
                    if old_n in content:
                        content = content.replace(old_n, new_n)
                        modified = True
                if modified:
                    fpath.write_text(content, encoding="utf-8")
                    print(f"Updated references in: {fpath.relative_to(root)}")
            except Exception as e:
                print(f"Error reading/writing {fpath}: {e}")

# Also check root-level markdown files (handoff.md, README.md, etc.)
for fpath in root.glob("*.md"):
    try:
        content = fpath.read_text(encoding="utf-8")
        modified = False
        for old_n, new_n in rename_map.items():
            if old_n in content:
                content = content.replace(old_n, new_n)
                modified = True
        if modified:
            fpath.write_text(content, encoding="utf-8")
            print(f"Updated references in: {fpath.relative_to(root)}")
    except Exception as e:
        print(f"Error reading/writing {fpath}: {e}")

print("All  prefixes successfully stripped and references updated!")
