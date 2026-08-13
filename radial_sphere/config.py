"""Config loading — OmegaConf-backed, with a Hydra-compose CLI entry
(mirrors ant_swarm/config.py).

Two ways to get a config:

1. Library consumers (env registration, replay of run snapshots) call
   :func:`load_config`, which reads a single yaml — the
   ``RADIAL_SPHERE_CONFIG`` env var if set, else ``configs/rl/config.yaml``.

2. Entry scripts call :func:`load_config_cli`, which adds Hydra's *compose
   API* on top (run dirs, logging, and the working directory stay under our
   control; the yamls stay plain):

       python scripts/rl/train_rl.py -cn variant rl.total_steps=5e5
       RADIAL_SPHERE_CONFIG=configs/rl/variant.yaml python scripts/rl/train_rl.py

Both return an ``omegaconf.DictConfig`` (attribute access: ``cfg.robot.n_bars``).
"""
from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# package dir → project root → configs/rl/config.yaml
_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _ROOT / "configs" / "rl" / "config.yaml"


def _default_path() -> Path:
    """Project default yaml, unless RADIAL_SPHERE_CONFIG points at a variant
    (used to run parallel jobs, each with its own config)."""
    return Path(os.environ.get("RADIAL_SPHERE_CONFIG") or CONFIG_PATH)


def load_config(path: str | Path | None = None) -> DictConfig:
    """Load one yaml as a DictConfig (no CLI, no composition)."""
    return OmegaConf.load(Path(path) if path else _default_path())


def load_config_dict(path: str | Path | None = None) -> dict:
    """Return the config as a plain nested dict."""
    return OmegaConf.to_container(load_config(path), resolve=True)


def load_config_cli(path: str | Path | None = None,
                    name: str | None = None,
                    overrides: list[str] | None = None,
                    config_dir: str | Path | None = None) -> DictConfig:
    """Hydra-compose a config for an entry script.

    Priority: explicit ``path`` > ``name`` (under configs/rl/) >
    RADIAL_SPHERE_CONFIG > configs/rl/config.yaml.  ``overrides`` is a
    ``key=value`` dotlist from the command line.
    """
    from hydra import compose, initialize_config_dir

    config_dir = Path(config_dir) if config_dir else (_ROOT / "configs" / "rl")
    if path is not None:
        p = Path(path)
        config_dir, name = p.parent, p.stem
    elif name is None and os.environ.get("RADIAL_SPHERE_CONFIG"):
        p = Path(os.environ["RADIAL_SPHERE_CONFIG"])
        config_dir, name = p.parent, p.stem
    if name is None:
        name = "config"

    bad = [o for o in (overrides or []) if "=" not in o]
    if bad:
        raise SystemExit(f"bad overrides {bad}; use key=value")
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()),
                               version_base=None):
        cfg = compose(config_name=name, overrides=list(overrides or []))
    # let downstream consumers (run-id tag, snapshot fallback) see the choice
    chosen = Path(config_dir) / f"{name}.yaml"
    if chosen.exists():
        os.environ.setdefault("RADIAL_SPHERE_CONFIG", str(chosen.resolve()))
    return cfg
