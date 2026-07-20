#!/usr/bin/env python3
"""Read one optional value from a cluster.yaml profile.

CLI usage (Net_React-compatible):
  python tools/read_cluster_setting.py --config cluster.yaml --key local_scratch --default /scratch
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_cluster_yaml(config: Path) -> dict[str, Any]:
    with Path(config).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config} must contain a mapping at the top level")
    return data


def resolve_cluster_name(data: dict[str, Any], cluster: str | None = None) -> str:
    name = cluster or data.get("current_cluster")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("cluster.yaml current_cluster must be a non-empty string")
    return name.strip()


def read_cluster_setting(
    config: Path,
    key: str,
    default: Any = None,
    *,
    cluster: str | None = None,
    choices: list[str] | None = None,
) -> str:
    """Return clusters[current].key as a stripped string, or default."""
    data = load_cluster_yaml(config)
    name = resolve_cluster_name(data, cluster)
    clusters = data.get("clusters", {})
    if not isinstance(clusters, dict) or name not in clusters:
        raise KeyError(f"cluster {name!r} is not configured in {config}")
    profile = clusters[name]
    if isinstance(profile, dict) and key in profile and profile[key] is not None:
        value = profile[key]
    else:
        value = default
    if value is None:
        raise KeyError(f"{name}.{key} is unset and no default was provided")
    value_s = str(value).strip()
    if choices is not None and value_s not in choices:
        raise ValueError(f"{name}.{key} must be one of {choices}; got {value_s!r}")
    return value_s


def optional_cluster_setting(
    config: Path,
    key: str,
    *,
    cluster: str | None = None,
) -> str | None:
    """Return clusters[current].key if present and non-empty after strip; else None."""
    data = load_cluster_yaml(config)
    name = resolve_cluster_name(data, cluster)
    clusters = data.get("clusters", {})
    if not isinstance(clusters, dict) or name not in clusters:
        raise KeyError(f"cluster {name!r} is not configured in {config}")
    profile = clusters[name]
    if not isinstance(profile, dict) or key not in profile or profile[key] is None:
        return None
    value_s = str(profile[key]).strip()
    return value_s or None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def first_enabled_partition(
    config: Path,
    *,
    cluster: str | None = None,
    default_time: str = "7-00:00:00",
) -> tuple[str, str]:
    """
    Return (partition_name, walltime) for the first partitions entry with use: true.

    Partition order follows YAML mapping order. If ``time`` is omitted on that
    partition, ``default_time`` is used.
    """
    data = load_cluster_yaml(config)
    name = resolve_cluster_name(data, cluster)
    clusters = data.get("clusters", {})
    if not isinstance(clusters, dict) or name not in clusters:
        raise KeyError(f"cluster {name!r} is not configured in {config}")
    profile = clusters[name]
    if not isinstance(profile, dict):
        raise ValueError(f"clusters.{name} must be a mapping")
    partitions = profile.get("partitions")
    if not isinstance(partitions, dict) or not partitions:
        raise KeyError(f"clusters.{name}.partitions is missing or empty")

    for pname, raw in partitions.items():
        if not isinstance(raw, dict):
            raise ValueError(f"clusters.{name}.partitions.{pname} must be a mapping")
        if not _as_bool(raw.get("use", True)):
            continue
        walltime = str(raw.get("time") or default_time).strip()
        if not walltime:
            walltime = default_time
        return str(pname).strip(), walltime

    raise KeyError(
        f"clusters.{name}.partitions has no entry with use: true"
    )


def memory_per_cpu_setting(
    config: Path,
    *,
    cluster: str | None = None,
    default: str = "4G",
) -> str:
    """
    Return Slurm --mem-per-cpu value.

    Preference: first enabled partition.memory_per_cpu, then cluster.memory_per_cpu,
    then ``default``.
    """
    data = load_cluster_yaml(config)
    name = resolve_cluster_name(data, cluster)
    clusters = data.get("clusters", {})
    if not isinstance(clusters, dict) or name not in clusters:
        raise KeyError(f"cluster {name!r} is not configured in {config}")
    profile = clusters[name]
    if not isinstance(profile, dict):
        raise ValueError(f"clusters.{name} must be a mapping")

    partitions = profile.get("partitions")
    if isinstance(partitions, dict):
        for _pname, raw in partitions.items():
            if not isinstance(raw, dict) or not _as_bool(raw.get("use", True)):
                continue
            part_mem = raw.get("memory_per_cpu")
            if part_mem is not None and str(part_mem).strip():
                return str(part_mem).strip()
            break

    cluster_mem = profile.get("memory_per_cpu")
    if cluster_mem is not None and str(cluster_mem).strip():
        return str(cluster_mem).strip()
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cluster")
    parser.add_argument("--key")
    parser.add_argument("--default")
    parser.add_argument("--choices", nargs="*")
    parser.add_argument(
        "--first-enabled-partition",
        action="store_true",
        help="Print 'name\\ttime' for the first partitions entry with use: true",
    )
    args = parser.parse_args()
    try:
        if args.first_enabled_partition:
            pname, walltime = first_enabled_partition(
                args.config, cluster=args.cluster
            )
            print(f"{pname}\t{walltime}")
            return
        if not args.key or args.default is None:
            parser.error("--key and --default are required unless --first-enabled-partition")
        print(
            read_cluster_setting(
                args.config,
                args.key,
                args.default,
                cluster=args.cluster,
                choices=args.choices,
            )
        )
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
