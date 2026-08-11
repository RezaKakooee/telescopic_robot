"""Load the project's ``config.yaml`` (single source of truth) into a namespace.

    from radial_sphere.config import load_config
    cfg = load_config()
    cfg.robot.n_bars        # 60
    cfg.reward.goal_eps     # 0.45

The YAML lives at the project root (one level above this package).
"""
from __future__ import annotations

import types
from pathlib import Path

import yaml

# package dir → project root → config.yaml
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _to_ns(obj):
    if isinstance(obj, dict):
        return types.SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(x) for x in obj]
    return obj


def load_config_dict(path: str | Path | None = None) -> dict:
    """Return the raw nested dict."""
    path = Path(path) if path else CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path: str | Path | None = None) -> types.SimpleNamespace:
    """Return the config as a nested attribute namespace."""
    return _to_ns(load_config_dict(path))
