#!/usr/bin/env python3
"""Repo-root discovery and software.yaml loading for react_benchmark."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SOFTWARE_YAML = REPO_ROOT / "software.yaml"


def repo_path(*parts: str | Path) -> Path:
    """Return an absolute path under the repository root."""
    return REPO_ROOT.joinpath(*parts)


def expand_user_path(value: str) -> str:
    return os.path.expanduser(str(value))


@lru_cache(maxsize=1)
def load_software() -> dict[str, Any]:
    with SOFTWARE_YAML.open() as f:
        data = yaml.safe_load(f) or {}
    return _expand_software(data)


def _expand_software(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_software(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_software(v) for v in obj]
    if isinstance(obj, str):
        return expand_user_path(obj)
    return obj


def software_get(*keys: str, default: Any = None) -> Any:
    cur: Any = load_software()
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_project_path(path_str: str | Path) -> Path:
    """
    Resolve a path that may be:
      - absolute (returned as-is, with ~ expanded)
      - relative to REPO_ROOT (e.g. params/skf/run59)
      - soft:<key> alias into software.yaml dftb_skf
    """
    s = expand_user_path(str(path_str).strip())
    if s.startswith("soft:"):
        key = s[len("soft:") :]
        skf = software_get("dftb_skf", default={}) or {}
        if key not in skf:
            # also allow bare directory-name keys
            raise KeyError(f"Unknown soft SKF key in software.yaml dftb_skf: {key!r}")
        return Path(skf[key])
    p = Path(s)
    if p.is_absolute():
        return p
    return repo_path(p)


def resolve_dftb_skroot(value: str | Path) -> Path:
    """Resolve DFTBPLUS_skroot from run_configs (soft: alias, relative, or absolute)."""
    s = str(value).strip()
    # Map common absolute soft paths / bare names to soft: aliases when possible
    skf_map = software_get("dftb_skf", default={}) or {}
    for key, abs_path in skf_map.items():
        if s == abs_path or s.rstrip("/") == str(abs_path).rstrip("/"):
            return Path(abs_path)
        if s == key:
            return Path(abs_path)
    return resolve_project_path(s)


if __name__ == "__main__":
    sw = load_software()
    print(f"REPO_ROOT={REPO_ROOT}")
    print(f"software keys: {sorted(sw.keys())}")
    for k in ("orca", "molpro_root", "mrcc", "ambertools_sh", "apptainer"):
        print(f"  {k}={sw.get(k)}")
    print(f"  dftb_skf={list((sw.get('dftb_skf') or {}).keys())}")
    print(f"  gxtb={sw.get('gxtb')}")
