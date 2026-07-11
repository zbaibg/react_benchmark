#!/usr/bin/env python3
"""
Trim an XYZ structure by molecule (residue) indices using the project's xyz_to_mda().

Intended for g-xtb many-body workflows:
  - generation step copies full xyz to source.xyz
  - job step runs this script to produce input.xyz for monomer/dimer/trimer jobs
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PY_SCRIPTS = _PROJECT_ROOT / "python_scripts"
sys.path.insert(0, str(_PY_SCRIPTS))

from zif_meoh_assign_name import xyz_to_mda  # noqa: E402


def _parse_keep_list(text: str) -> list[int]:
    items: list[int] = []
    for chunk in text.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(int(chunk))
    return items


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_xyz", required=True, help="Input XYZ (full system)")
    p.add_argument("--out", dest="out_xyz", required=True, help="Output XYZ (trimmed)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--keep", default=None, help="Comma-separated 0-based molecule indices to keep")
    g.add_argument("--keep-file", default=None, help="File containing keep list (comma/newline separated)")
    args = p.parse_args()

    in_path = Path(args.in_xyz)
    out_path = Path(args.out_xyz)
    if not in_path.exists():
        raise FileNotFoundError(f"Input xyz not found: {in_path}")

    if args.keep_file is not None:
        keep_text = Path(args.keep_file).read_text()
        keep = _parse_keep_list(keep_text)
    else:
        keep = _parse_keep_list(args.keep or "")

    if not keep:
        raise ValueError("Keep list is empty.")

    u = xyz_to_mda(str(in_path))
    resids = [i + 1 for i in keep]
    sel = u.select_atoms(" or ".join(f"resid {r}" for r in resids))
    if len(sel) == 0:
        raise ValueError(f"Selection empty for keep={keep} (resids={resids}) from {in_path}")

    sel.write(str(out_path))


if __name__ == "__main__":
    main()

