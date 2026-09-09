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
import sys
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
    """Load a config YAML, composing Hydra ``defaults`` when present."""
    chosen = Path(path) if path else _default_path()
    raw = OmegaConf.load(chosen)
    if "defaults" not in raw:
        return raw

    # Config presets such as maze_level3.yaml inherit from config.yaml. This
    # path is also used during package import/registration, before entry-point
    # scripts call load_config_cli(), so it must honour that composition too.
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(chosen.resolve().parent),
                               version_base=None):
        return compose(config_name=chosen.stem)


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


def script_config(script: str,
                  argv: list[str] | None = None,
                  *,
                  passthrough: bool = False,
                  config_dir: str | Path | None = None) -> DictConfig:
    """Compose an entry script's own knobs, plus ``key=value`` CLI overrides.

    Entry scripts used to parse flags with `argparse`, which meant every knob
    was declared twice: once in the parser and once in whatever yaml the run
    also loaded. A script's knobs now live in ``configs/scripts/<script>.yaml``
    and the command line takes Hydra's dotlist overrides::

        python docs/blog/render_rough_terrain.py seconds=30 speed=0.9
        python demos/gap/runner.py steps=800 video=false

    Because the yaml can carry a ``defaults:`` list, a script config can pull
    in a scenario preset from ``configs/rl/`` and override parts of it in the
    same file.

    A key the script does not declare is an error, so a typo or a stale flag
    name is reported instead of being silently ignored. Scripts that also feed
    scenario overrides to :func:`load_config_cli` pass ``passthrough=True``;
    for those, unknown keys are collected into ``cfg.scenario_overrides`` and
    one command line can carry both::

        python scripts/rl/train_rl.py seed=7 rl.n_steps=512

    ``--help`` prints the composed config, which is the full list of knobs.
    Nothing here touches the working directory or the run directory: those
    stay under the caller's control, which is why this composes by hand
    instead of using ``@hydra.main``.
    """
    from hydra import compose, initialize_config_dir

    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(config_dir) if config_dir else (_ROOT / "configs" / "scripts")
    path = root / f"{script}.yaml"
    if not path.exists():
        raise SystemExit(f"no config for {script!r}; expected {path}")

    if any(a in ("-h", "--help") for a in argv):
        cfg = _compose_script(compose, initialize_config_dir, root, script, [])
        print(f"{script}: knobs from {path}\n")
        print(OmegaConf.to_yaml(cfg))
        print("Override any of them on the command line as key=value, "
              "for example:\n"
              f"    python <this script> {_example_override(cfg)}")
        raise SystemExit(0)

    bad = [a for a in argv if "=" not in a]
    if bad:
        raise SystemExit(
            f"unexpected argument(s) {bad}. This script takes key=value "
            f"overrides, not flags. Run with --help to list the knobs.")

    # Split the command line: keys this script declares are its own knobs,
    # anything else is a scenario override bound for `load_config_cli`. That
    # lets one command line carry both, e.g.
    #     python scripts/rl/train_rl.py seed=7 rl.n_steps=512
    known = set(_compose_script(compose, initialize_config_dir, root, script, []).keys())
    mine = [a for a in argv if a.split("=", 1)[0].lstrip("+~").split(".")[0] in known]
    theirs = [a for a in argv if a not in mine]
    if theirs and not passthrough:
        raise SystemExit(
            f"{script}: unknown knob(s) {[t.split('=')[0] for t in theirs]}. "
            f"This script accepts: {', '.join(sorted(known))}. "
            f"Run with --help to see the current values.")
    cfg = _compose_script(compose, initialize_config_dir, root, script, mine)
    OmegaConf.set_struct(cfg, False)
    cfg.scenario_overrides = theirs
    return cfg


def _compose_script(compose, initialize_config_dir, root, script, overrides):
    with initialize_config_dir(config_dir=str(Path(root).resolve()),
                               version_base=None):
        return compose(config_name=script, overrides=list(overrides))


def _example_override(cfg: DictConfig) -> str:
    """Pick one leaf key so --help can show a runnable example."""
    for key, value in cfg.items():
        if not isinstance(value, DictConfig):
            return f"{key}={value!r}" if isinstance(value, str) else f"{key}={value}"
    return "key=value"


def demo_config(name: str, argv: list[str] | None = None) -> DictConfig:
    """Compose one demo's yaml from ``demos/<name>/demo.yaml``.

    A demo folder owns everything about showing one skill: the scenario it
    runs in, the skill and its arguments, how long to run, what to record,
    what counts as success, and any knobs its own runner reads. Script knobs
    used to live in a parallel tree under ``configs/scripts/``, which meant
    one demo was described in four places.

    ``key=value`` overrides work the same as :func:`script_config`, including
    the error on an unknown key, but they address the nested spec, so a knob
    is written ``knobs.fps=30``.
    """
    from hydra import compose, initialize_config_dir

    argv = list(sys.argv[1:] if argv is None else argv)
    root = _ROOT / "demos" / name
    path = root / "demo.yaml"
    if not path.exists():
        available = sorted(q.parent.name for q in (_ROOT / "demos").glob("*/demo.yaml"))
        raise SystemExit(f"no demo {name!r}; available: {', '.join(available)}")

    if any(a in ("-h", "--help") for a in argv):
        cfg = _compose_script(compose, initialize_config_dir, root, "demo", [])
        print(f"{name}: {path}\n")
        print(OmegaConf.to_yaml(cfg))
        raise SystemExit(0)

    overrides = [a for a in argv if "=" in a]
    return _compose_script(compose, initialize_config_dir, root, "demo", overrides)
