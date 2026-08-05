"""Load configs/workload.yaml identically for every runner.

Kept separate from the runners so arms A, B and C cannot drift into slightly
different interpretations of the same file — which would make their numbers
incomparable while still looking like they shared a config.

The *resolved* config (post-CLI-override) is what gets hashed into the
manifest, so ``apply_overrides`` returns a new dict rather than mutating: the
hash must describe what the run used, not what the file said.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Iterable, Optional

__all__ = ["load_config", "apply_overrides", "config_sha256", "DEFAULT_CONFIG_PATH"]

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs", "workload.yaml"
)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Parse the workload YAML. Requires pyyaml (in benchmarks/requirements.txt)."""
    p = os.path.abspath(path or DEFAULT_CONFIG_PATH)
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "pyyaml is required to load the workload config. "
            "pip install -r benchmarks/requirements.txt"
        ) from exc

    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{p}: expected a mapping at the top level")
    cfg["_config_path"] = p
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``cfg`` with dotted-path overrides applied.

    ``{"generation.max_new_tokens": 64}`` sets ``cfg["generation"]["max_new_tokens"]``.
    A None value is ignored so argparse defaults do not clobber the file.
    """
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        if value is None:
            continue
        parts = dotted.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"override path {dotted!r} traverses a non-mapping")
        node[parts[-1]] = value
    return out


def config_sha256(cfg: Dict[str, Any]) -> str:
    """Hash the resolved config, excluding bookkeeping keys.

    ``_config_path`` is stripped: the same config used from two checkout
    locations is the same config, and letting an absolute path into the hash
    would make every run look unique.
    """
    import hashlib

    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
