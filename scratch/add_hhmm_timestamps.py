import os, time, pathlib

p = pathlib.Path("/home/azureuser/telescopic_robot/storage_local")
rename_map = {}

# Target directories to format with YYYYMMDD_HHMM__<name>
for d in sorted(p.iterdir()):
    if not d.is_dir():
        continue
    # Check if folder starts with 8 digits followed by double underscore (e.g. 20260823__name)
    parts = d.name.split("__", 1)
    if len(parts) == 2 and len(parts[0]) == 8 and parts[0].isdigit():
        base_name = parts[1]
        mtime = d.stat().st_mtime
        
        # Check files inside directory for most relevant mtime
        file_mtimes = []
        for root, subdirs, files in os.walk(d):
            for f in files:
                try:
                    file_mtimes.append(os.path.getmtime(os.path.join(root, f)))
                except:
                    pass
        if file_mtimes:
            # Use average or earliest or latest file mtime
            mtime = min(file_mtimes)
        
        ts_str = time.strftime("%Y%m%d_%H%M", time.localtime(mtime))
        new_name = f"{ts_str}__{base_name}"
        rename_map[d.name] = new_name
        target = p / new_name
        d.rename(target)
        print(f"Renamed: {d.name} -> {new_name}")

print(f"\nTotal directories renamed: {len(rename_map)}")
print("Updating all references across codebase...")

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
                    print(f"Updated: {fpath.relative_to(root)}")
            except Exception as e:
                print(f"Error updating {fpath}: {e}")

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
            print(f"Updated: {fpath.relative_to(root)}")
    except Exception as e:
        print(f"Error updating {fpath}: {e}")

print("Completed adding YYYYMMDD_HHMM__ timestamps and updating references!")
