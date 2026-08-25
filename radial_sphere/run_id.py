"""Run identifier helpers (mirrors ant_swarm/run_id.py).

Each run gets ONE id shared by the ops .out log file, the run directory
under storage_local/, and the wandb run name. The ops wrapper
(ops/sb_train.sh) mints the id up front and exports it as
RADIAL_SPHERE_RUN_ID; Python generates one only when the environment
variable is absent (e.g. local runs).

Format (when generated here):
    <YYYYMMDD_HHMM>__<slurm job id | local>__<script>[__<tag>][__<config tag>]
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path


def normalize_name(name: str) -> str:
    """Return a filesystem-friendly name segment."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())
    return cleaned.strip("_") or "run"


def build_run_id(script_name: str = "run", tag: str | None = None) -> str:
    """One id for run dir + wandb + logs. RADIAL_SPHERE_RUN_ID overrides."""
    override = os.environ.get("RADIAL_SPHERE_RUN_ID")
    if override:
        return normalize_name(override)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    parts = [ts, job_id, normalize_name(script_name)]
    if tag:
        parts.append(normalize_name(tag))
    cfg_tag = Path(os.environ.get("RADIAL_SPHERE_CONFIG", "")).stem
    if cfg_tag and cfg_tag != "config":     # skip the default config's name
        parts.append(normalize_name(cfg_tag))
    return "__".join(parts)
