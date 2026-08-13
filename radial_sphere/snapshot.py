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

import shutil
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent        # .../radial_sphere
_ROOT = _PKG_DIR.parent                            # project root
STORAGE_DIR = _ROOT / "storage_local"


def make_run_dir(run_id: str) -> Path:
    """Create ``storage_local/<run_id>/renders/`` and return the run dir.

    ``run_id`` comes from :func:`radial_sphere.run_id.build_run_id`, so the
    run dir, the ops .out log, and the wandb run share one name.
    """
    run_dir = STORAGE_DIR / run_id
    (run_dir / "renders").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_code(run_dir, script_path: str | None = None, cfg=None) -> Path:
    """Snapshot the package source + config + entry script into ``<run_dir>/code/``.

    Pass the loaded ``cfg`` to save the RESOLVED config (with CLI overrides
    applied); otherwise the project default yaml is copied.
    """
    dest = Path(run_dir) / "code"
    dest.mkdir(parents=True, exist_ok=True)

    # package source
    shutil.copytree(
        _PKG_DIR, dest / "radial_sphere",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    # config: resolved if given, else the project default
    if cfg is not None:
        from omegaconf import OmegaConf
        OmegaConf.save(cfg, dest / "config.yaml")
    else:
        default = _ROOT / "configs" / "rl" / "config.yaml"
        if default.exists():
            shutil.copy2(default, dest / "config.yaml")
    # entry script
    if script_path:
        sp = Path(script_path)
        if sp.exists():
            shutil.copy2(sp, dest / sp.name)
    return dest
