"""Run-dir creation + code snapshot under ``storage_local`` (reproducibility).

All run outputs live under ``storage_local/`` (mirrors the ant project)::

    storage_local/radial_sphere__<YYYYMMDD_HHMM>__<jobid>__<tag>/
        renders/        # episode videos
        code/           # snapshot: radial_sphere/ + config.yaml + entry script

    from radial_sphere import make_run_dir, save_code
    run_dir = make_run_dir("heuristic")
    save_code(run_dir, __file__)
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent        # .../radial_sphere
_ROOT = _PKG_DIR.parent                            # project root
STORAGE_DIR = _ROOT / "storage_local"


def make_run_dir(tag: str) -> Path:
    """Create ``storage_local/radial_sphere__<ts>__<jobid>__<tag>/renders/`` and return the run dir."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = STORAGE_DIR / f"radial_sphere__{ts}__{job_id}__{tag}"
    (run_dir / "renders").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_code(run_dir, script_path: str | None = None) -> Path:
    """Snapshot the package source + config.yaml + entry script into ``<run_dir>/code/``."""
    dest = Path(run_dir) / "code"
    dest.mkdir(parents=True, exist_ok=True)

    # package source
    shutil.copytree(
        _PKG_DIR, dest / "radial_sphere",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    # config.yaml
    cfg = _ROOT / "config.yaml"
    if cfg.exists():
        shutil.copy2(cfg, dest / "config.yaml")
    # entry script
    if script_path:
        sp = Path(script_path)
        if sp.exists():
            shutil.copy2(sp, dest / sp.name)
    return dest
