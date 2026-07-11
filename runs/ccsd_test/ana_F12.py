#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

SOURCE_SPECS = {
    "mrcc": {
        "csv_path": ROOT / "mrcc" / "dE.csv",
        "runs": {3, 4, 5, 6, 7, 8},
    },
    "mrcc_new": {
        "csv_path": ROOT / "mrcc_new" / "dE.csv",
        "runs": {3, 4, 5, 6, 7, 8},
    },
    "molpro": {
        "csv_path": ROOT / "molpro" / "dE.csv",
        "runs": {12, 13, 19, 20, 21, 22},
    },
    "molpro_pno": {
        "csv_path": ROOT / "dE.csv",
        "runs": {60, 69, 70, 71, 72, 73},
        "method_pattern": r"Molpro\s+PNO-LCCSD\(T\)",
    },
    # DF-HF + pno-lccsd(t) / pno-lccsd(t)-f12, avdz/avtz/avqz-f12 (runs 84–91 subset)
    "molpro_df_hf_pno": {
        "csv_path": ROOT / "dE.csv",
        "runs": {84, 85, 88, 89, 90, 91},
        "method_pattern": r"Molpro\s+PNO-LCCSD.*DF-HF",
        "figure_title": "DF-HF PNO-LCCSD(T)",
    },
    # LDF-HF + pno-lccsd(t) / pno-lccsd(t)-f12, avdz/avtz/avqz-f12 (runs 93–98)
    "molpro_ldf_hf_pno": {
        "csv_path": ROOT / "dE.csv",
        "runs": {93, 94, 95, 96, 97, 98},
        "method_pattern": r"Molpro\s+PNO-LCCSD.*ldf-hf",
        "figure_title": "LDF-HF PNO-LCCSD(T)",
    },
}


def run_to_int(run_name: str) -> int | None:
    m = re.search(r"run(\d+)", str(run_name).strip(), re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def is_explicit_f12_method(method: str) -> bool:
    """True for CCSD(T)-F12 / DF-CCSD(T)-F12 style methods, not for CCSD(T), avtz-f12 (basis name only)."""
    m = (method or "").strip()
    if re.search(r"ccsd\s*\(\s*t\s*\)\s*-\s*f12", m, re.IGNORECASE):
        return True
    if re.search(r"df\s*-\s*ccsd\s*\(\s*t\s*\)\s*-\s*f12", m, re.IGNORECASE):
        return True
    if re.search(r"pno\s*-\s*lccsd\s*\(\s*t\s*\)\s*-\s*f12", m, re.IGNORECASE):
        return True
    return False


def basis_info(method: str) -> tuple[int, str]:
    s = (method or "").lower()
    patterns = [
        (6, r"(?:av6z|v6z|\b6z\b)", "6Z"),
        (5, r"(?:av5z|v5z|\b5z\b)", "5Z"),
        (4, r"(?:avqz|vqz|\bqz\b)", "QZ"),
        (3, r"(?:avtz|vtz|\btz\b)", "TZ"),
        (2, r"(?:avdz|vdz|\bdz\b)", "DZ"),
    ]
    for rank, pat, label in patterns:
        if re.search(pat, s):
            return rank, label
    return 0, "UNK"


def pick_ebind(row: dict[str, str]) -> float | None:
    candidates = [
        "Ebind",
        "Ebind_dist_Zn_MeOH",
        "Ebind_T+",
        "Ebind_T*",
        "Ebind_T",
    ]
    for col in candidates:
        val = row.get(col, "")
        if val is None:
            continue
        val = str(val).strip()
        if not val:
            continue
        try:
            if val.lower() in {"nan", "inf", "-inf", "+inf"}:
                continue
            x = float(val)
            if math.isnan(x) or math.isinf(x):
                continue
            return x
        except ValueError:
            continue
    return None


def load_records(
    name: str, csv_path: Path, runs: set[int], method_pattern: str | None = None
) -> list[dict]:
    records: list[dict] = []
    method_re = re.compile(method_pattern, re.IGNORECASE) if method_pattern else None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_idx = run_to_int(row.get("Run", ""))
            if run_idx is None or run_idx not in runs:
                continue

            method = row.get("Method", "")
            if method_re and not method_re.search(method):
                continue
            bsse_raw = str(row.get("BSSE", "")).strip().lower()
            bsse_yes = bsse_raw in {"yes", "y", "true", "1"}
            is_f12 = is_explicit_f12_method(method)
            ebind = pick_ebind(row)
            basis_rank, basis_label = basis_info(method)

            if ebind is None:
                continue

            records.append(
                {
                    "source": name,
                    "run": run_idx,
                    "method": method,
                    "is_f12": is_f12,
                    "bsse_yes": bsse_yes,
                    "ebind": ebind,
                    "basis_rank": basis_rank,
                    "basis_label": basis_label,
                }
            )
    return records


def aggregate(records: list[dict]) -> dict[tuple[bool, bool], list[tuple[int, str, float]]]:
    grouped = defaultdict(list)
    for rec in records:
        key = (rec["is_f12"], rec["bsse_yes"])
        grouped[key].append((rec["basis_rank"], rec["basis_label"], rec["ebind"]))

    out = {}
    for key, vals in grouped.items():
        by_basis = defaultdict(list)
        for rank, label, ebind in vals:
            by_basis[(rank, label)].append(ebind)
        merged = []
        for (rank, label), ebind_vals in by_basis.items():
            merged.append((rank, label, sum(ebind_vals) / len(ebind_vals)))
        merged.sort(key=lambda x: x[0])
        out[key] = merged
    return out


def draw_single_source(
    src: str, records: list[dict], *, figure_title: str | None = None
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.8))
    color_map = {False: "#1f77b4", True: "#d62728"}
    line_map = {False: "-", True: "--"}
    marker_map = {False: "o", True: "s"}
    combo_order = [(False, False), (False, True), (True, False), (True, True)]
    grouped = aggregate(records)
    used_x = {}
    for key in combo_order:
        is_f12, bsse_yes = key
        pts = grouped.get(key, [])
        if not pts:
            continue
        x = [p[0] for p in pts]
        y = [p[2] for p in pts]
        for rank, label, _ in pts:
            used_x[rank] = label
        leg = f"{'F12' if is_f12 else 'noF12'}, {'BSSE corrected' if bsse_yes else 'not BSSE corrected'}"
        ax.plot(
            x,
            y,
            marker=marker_map[bsse_yes],
            linestyle=line_map[bsse_yes],
            color=color_map[is_f12],
            linewidth=2.0,
            markersize=6,
            label=leg,
        )

    if used_x:
        x_sorted = sorted(used_x)
        ax.set_xticks(x_sorted)
        ax.set_xticklabels([used_x[v] for v in x_sorted])

    ax.set_xlabel("Basis size")
    ax.set_ylabel("Ebind")
    head = figure_title if figure_title else src
    ax.set_title(f"{head}\nEbind vs basis (F12 / BSSE)", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_png = ROOT / f"Ebind_F12_BSSE_vs_basis_{src}.png"
    fig.savefig(out_png, dpi=200)
    print(f"Saved figure: {out_png}")


def main() -> None:
    all_data = {}
    for name, spec in SOURCE_SPECS.items():
        csv_path = spec["csv_path"]
        if not csv_path.exists():
            print(f"[WARN] Missing file: {csv_path}")
            all_data[name] = []
            continue
        recs = load_records(name, csv_path, spec["runs"], spec.get("method_pattern"))
        all_data[name] = recs
        print(f"[INFO] {name}: loaded {len(recs)} usable rows from {csv_path.name}")
    plot_order = [
        "mrcc",
        "mrcc_new",
        "molpro",
        "molpro_pno",
        "molpro_df_hf_pno",
        "molpro_ldf_hf_pno",
    ]
    for src in plot_order:
        spec = SOURCE_SPECS.get(src, {})
        draw_single_source(
            src,
            all_data.get(src, []),
            figure_title=spec.get("figure_title"),
        )


if __name__ == "__main__":
    main()
