#!/usr/bin/env python3
"""
Parse Amber+DL-Find minimization min.out files under subdirectories and print a summary table:
目录名、优化步数、最终 Etot (kcal/mol)、Energy calculation finished 行中的能量。
表格按步数升序排列。

用法（脚本放在某个 run 目录内，例如 run6）:
  python extract_min_out_table.py              # 默认扫描「脚本所在目录」下各子目录的 min.out
  python extract_min_out_table.py --tsv
  python extract_min_out_table.py --markdown
  python extract_min_out_table.py /path/to/other_run   # 显式指定别的根目录
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 无参数时始终扫脚本所在目录，可从任意 cwd 运行（例如 python path/to/run6/extract_min_out_table.py）
_SCRIPT_DIR = Path(__file__).resolve().parent

STEPS_RE = re.compile(r"Number of steps\s*\.+\s*(\d+)\s*$", re.MULTILINE)
ETOT_RE = re.compile(r"^\s*Etot\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)", re.MULTILINE)
FINISHED_RE = re.compile(
    r"Energy calculation finished,\s*energy:\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)"
)


def parse_min_out(path: Path) -> dict[str, str | int | float | None]:
    text = path.read_text(errors="replace")
    steps_m = list(STEPS_RE.finditer(text))
    steps = int(steps_m[-1].group(1)) if steps_m else None

    etot_m = list(ETOT_RE.finditer(text))
    etot = float(etot_m[-1].group(1)) if etot_m else None

    fin_m = list(FINISHED_RE.finditer(text))
    energy_finished = float(fin_m[-1].group(1)) if fin_m else None

    return {
        "dir": path.parent.name,
        "steps": steps,
        "etot_kcal": etot,
        "energy_finished": energy_finished,
    }


def gather_rows(root: Path) -> list[dict[str, str | int | float | None]]:
    rows = []
    for min_out in sorted(root.glob("*/min.out")):
        rows.append(parse_min_out(min_out))
    return rows


def sort_rows_by_steps(rows: list[dict[str, str | int | float | None]]) -> list[dict]:
    """步数升序；无步数数据的行排在最后；相同步数按目录名排序。"""

    def key(r: dict) -> tuple[float, str]:
        s = r["steps"]
        n = float(s) if s is not None else float("inf")
        return (n, str(r["dir"]))

    return sorted(rows, key=key)


def _fmt_row(r: dict) -> tuple[str, str, str, str]:
    d = str(r["dir"])
    s = "" if r["steps"] is None else str(r["steps"])
    e = "" if r["etot_kcal"] is None else f"{r['etot_kcal']:.4f}"
    f = "" if r["energy_finished"] is None else f"{r['energy_finished']:.10g}"
    return d, s, e, f


def print_plain(rows: list[dict]) -> None:
    w = [28, 6, 18, 22]
    hdr = ["目录设置", "步数", "Etot (kcal/mol)", "energy finished"]
    line = " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr))
    print(line)
    print("---".join("-" * w[i] for i in range(4)))
    for r in rows:
        d, s, e, f = _fmt_row(r)
        print(" | ".join([d.ljust(w[0]), s.rjust(w[1]), e.rjust(w[2]), f.rjust(w[3])]))


def print_markdown(rows: list[dict]) -> None:
    print("| 目录 | 步数 | Etot (kcal/mol) | Energy calculation finished |")
    print("|------|-----:|----------------:|----------------------------:|")
    for r in rows:
        d, s, e, f = _fmt_row(r)
        print(f"| {d} | {s} | {e or '—'} | {f or '—'} |")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Amber DL-Find min.out files.")
    ap.add_argument(
        "root",
        nargs="?",
        default=_SCRIPT_DIR,
        type=Path,
        help="含多个子目录、各含 min.out 的根路径（默认：脚本所在目录）",
    )
    ap.add_argument("--tsv", action="store_true", help="输出 TSV")
    ap.add_argument("--markdown", action="store_true", help="输出 Markdown 表格")
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1
    rows = sort_rows_by_steps(gather_rows(root))
    if not rows:
        print(f"No */min.out found under {root}", file=sys.stderr)
        return 1
    if args.tsv:
        # TSV 用固定列名
        print("directory\tsteps\tEtot_kcal_mol\tenergy_finished")
        for r in rows:
            d, s, e, f = _fmt_row(r)
            print(f"{d}\t{s}\t{e}\t{f}")
    elif args.markdown:
        print_markdown(rows)
    else:
        print_plain(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
