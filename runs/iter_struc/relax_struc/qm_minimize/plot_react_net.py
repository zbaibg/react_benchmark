from __future__ import annotations

import argparse
import io
import re
import shutil
from collections import defaultdict, deque
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import pandas as pd


def _workdir() -> Path:
    """Process current working directory (where you run e.g. `python ../plot_react_net.py`)."""
    return Path.cwd()

# CODATA: 1 Hartree = E_h / (kcal/mol per Hartree). Used to convert ana.log lowest_E (Hartree).
HARTREE_TO_KCAL_MOL = 627.5094738897745

# Monomer rows in min_last_convergence.tsv (column `system`) used for stoichiometric ΔE_rxn.
_MONOMER_REF_SYSTEM_NAMES = frozenset({"MIm", "MImH", "MeOH", "H"})

# If set to an int, drop rows with complex charge q < SCREEN_CHARGE (q = 2 - n_MIm) from the
# network, reaction tables, and optional scatter plots. None disables screening.
SCREEN_CHARGE: Optional[int] = None

# When True, draw gray filled nodes for missing stoichiometries that are sub-stoichiometries of
# some stable structure; when False, skip those circles and the matching legend entry.
PLOT_DERIVED_STRUC = False

# When True, draw a red border on active nodes with ``converged_signal_or_converged_loose_criteria ==
# no`` in min_last_convergence.tsv. When False, same border as other active nodes.
COLOR_UNCOVERGED = True

_UNCONVERGED_EDGECOLOR = "#d0021b"

_CONVERGED_LOOSE_COL = "converged_signal_or_converged_loose_criteria"

# Stoichiometry ``formula_id`` (network node key, reaction tables) comes only from this map:
# column ``formula`` -> ``formula_id`` (see ``zn_msmiles_enumeration.tsv``).
_MSMILES_ENUM_TSV = Path(__file__).resolve().parent.parent.parent / "zn_msmiles_enumeration.tsv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot zinc-imidazole-methanol reaction network from "
        "min_last_convergence.tsv (default, ORCA QM minimize summary) or "
        "check_metallogen_short_bonds ana.log / compatible TSV. Stoichiometry formula_id is "
        f"always taken from {_MSMILES_ENUM_TSV.name} (formula column), not from the input file."
    )
    p.add_argument(
        "data_file",
        type=Path,
        nargs="?",
        default=None,
        help="Input path (default: min_last_convergence.tsv in the current working directory). "
        "If that file is missing, pass ana.log or another formula TSV (with a formula column).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("react_net.png"),
        help="Output PNG path (default: react_net.png).",
    )
    p.add_argument(
        "--no-print-reactions",
        action="store_true",
        help="Do not print the reaction dE table to stdout.",
    )
    p.add_argument(
        "--no-print-gray-derivation",
        action="store_true",
        help="Do not print how each gray (derivable missing) node is derived from a stable structure.",
    )
    p.add_argument(
        "--de-table-png",
        type=Path,
        default=Path("react_net_dE.png"),
        help="Base path for LaTeX ΔE figures: one PNG per reaction type, e.g. "
        "react_net_dE.png -> react_net_dE_Add_MeOH.png, ... (default: react_net_dE.png).",
    )
    p.add_argument(
        "--no-de-table-png",
        action="store_true",
        help="Skip pdflatex + PNG rendering of the ΔE table.",
    )
    p.add_argument(
        "--latex-dpi",
        type=int,
        default=200,
        help="Rasterization DPI when converting the LaTeX PDF to PNG (default: 200).",
    )
    p.add_argument(
        "--no-etot-scatter-png",
        action="store_true",
        help="When using min_last_convergence.tsv, skip extra PNGs: Etot scatter and "
        "Zn–ligand distance scatter.",
    )
    return p.parse_args()


def _is_dash_separator(line: str) -> bool:
    s = line.strip()
    return bool(s) and all(c == "-" for c in s)


def extract_formula_table_text(path: Path) -> str:
    """Extract the formula table block (header starting with formula_id) from ana.log or a bare TSV."""
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    start: Optional[int] = None
    for i, line in enumerate(lines):
        if line.startswith("formula_id\t"):
            start = i
            break
        if "\tformula\t" in line and line.strip().startswith("formula_id"):
            start = i
            break
    if start is None:
        if lines and lines[0].lstrip().startswith("formula_id"):
            start = 0
        else:
            raise SystemExit(
                f"Could not find a header line starting with 'formula_id' in {path}"
            )

    j = start + 1
    if j < len(lines) and _is_dash_separator(lines[j]):
        j += 1

    out_lines: list[str] = []
    while j < len(lines):
        line = lines[j]
        if not line.strip():
            break
        if line.strip() == "runtime_messages" or line.startswith("runtime_messages"):
            break
        if _is_dash_separator(line):
            j += 1
            continue
        out_lines.append(line)
        j += 1

    if not out_lines:
        raise SystemExit(f"No data rows found after formula table header in {path}")
    # Header must be included so pandas gets column names (e.g. formula, lowest_E, ...).
    return lines[start] + "\n" + "\n".join(out_lines)


def read_formula_dataframe(data_str: str) -> pd.DataFrame:
    header_line = data_str.splitlines()[0]
    sep = "\t" if header_line.count("\t") >= 4 else r"\s+"
    df = pd.read_csv(io.StringIO(data_str), sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


_FORMULA_ZN_RE = re.compile(
    r"^1Zn_(\d+)MIm_(\d+)MImH_(\d+)MeOH$",
)


def _parse_tsv_float(val: object) -> Optional[float]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    s = str(val).strip()
    if not s or s.upper() == "N/A":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _row_unconverged_by_loose_criteria(row: pd.Series) -> bool:
    """True if ``converged_signal_or_converged_loose_criteria`` is ``no`` (min_last_convergence TSV)."""
    if _CONVERGED_LOOSE_COL not in row.index:
        return False
    v = row[_CONVERGED_LOOSE_COL]
    try:
        if pd.isna(v):
            return False
    except TypeError:
        pass
    return str(v).strip().lower() == "no"


def is_min_last_convergence_tsv(path: Path) -> bool:
    """True if file looks like parse_min_last_convergence output (system + ORCA Etot)."""
    try:
        line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except OSError:
        return False
    if "\t" not in line:
        return False
    cols = [c.strip() for c in line.split("\t")]
    return "system" in cols and "Etot_last_cycle_kcal_mol" in cols


def read_min_last_convergence_dataframe(path: Path) -> pd.DataFrame:
    """Build a DataFrame from min_last_convergence.tsv (no ``formula_id``; use enumeration map).

    Keeps only rows whose ``system`` matches ``1Zn_*MIm_*MImH_*MeOH`` (drops bare
    Zn / monomer / fragment lines). Energies stay in ``Etot_last_cycle_kcal_mol``
    (kcal/mol); reaction edges use cluster ΔE from these totals only.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "system" not in df.columns or "Etot_last_cycle_kcal_mol" not in df.columns:
        raise SystemExit(
            f"{path}: expected tab-separated columns 'system' and "
            "'Etot_last_cycle_kcal_mol'"
        )
    df = df.rename(columns={"system": "formula"})
    mask = df["formula"].astype(str).str.match(_FORMULA_ZN_RE)
    df = df.loc[mask].copy()
    if df.empty:
        raise SystemExit(f"{path}: no rows matching 1Zn_*MIm_*MImH_*MeOH after filter")

    etot_k = df["Etot_last_cycle_kcal_mol"].map(_parse_tsv_float)
    df["Etot_last_cycle_kcal_mol"] = etot_k
    df = df.sort_values("formula", kind="stable").reset_index(drop=True)
    df["n_conf_passed"] = etot_k.notna().astype(int)

    return df


def load_formula_id_map(enum_path: Path) -> Dict[str, int]:
    """``formula`` string -> ``formula_id`` from ``zn_msmiles_enumeration.tsv`` (unique per formula)."""
    if not enum_path.is_file():
        raise SystemExit(f"Formula ID enumeration file not found: {enum_path}")
    edf = pd.read_csv(enum_path, sep="\t", dtype=str, keep_default_na=False)
    edf.columns = [str(c).strip() for c in edf.columns]
    if "formula" not in edf.columns or "formula_id" not in edf.columns:
        raise SystemExit(
            f"{enum_path}: expected tab-separated columns 'formula' and 'formula_id'"
        )
    edf["formula"] = edf["formula"].astype(str).str.strip()
    fid_num = pd.to_numeric(edf["formula_id"], errors="coerce")
    if fid_num.isna().any():
        raise SystemExit(f"{enum_path}: non-numeric formula_id value(s) in column formula_id")
    edf = edf.assign(_fid=fid_num.astype(int))
    nuniq = edf.groupby("formula", sort=False)["_fid"].nunique()
    bad = nuniq[nuniq > 1]
    if not bad.empty:
        ex = bad.index[:5].tolist()
        raise SystemExit(
            f"{enum_path}: same formula maps to multiple formula_id values (e.g. {ex!r})"
        )
    first = edf.drop_duplicates(subset=["formula"], keep="first")
    return dict(zip(first["formula"], first["_fid"]))


def assign_formula_ids_from_enumeration(
    df: pd.DataFrame,
    formula_to_id: Dict[str, int],
    *,
    src_desc: str,
) -> pd.DataFrame:
    """Set ``formula_id`` from ``formula`` using ``formula_to_id``; fail on unknown or duplicate formulas."""
    if "formula" not in df.columns:
        raise SystemExit(f"{src_desc}: no 'formula' column (needed for formula_id lookup)")
    out = df.copy()
    formulas = out["formula"].astype(str).str.strip()
    missing = sorted(set(formulas) - set(formula_to_id.keys()))
    if missing:
        raise SystemExit(
            f"{src_desc}: {len(missing)} formula(s) not listed in {_MSMILES_ENUM_TSV.name} "
            f"(first few: {missing[:10]!r})"
        )
    dup = formulas.duplicated()
    if dup.any():
        raise SystemExit(
            f"{src_desc}: duplicate 'formula' on {int(dup.sum())} row(s); deduplicate input first."
        )
    out["formula_id"] = formulas.map(formula_to_id).astype(int)
    return out


def load_monomer_refs_kcal_from_tsv(path: Path) -> Dict[str, float]:
    """Monomer E_tot (kcal/mol) from rows with ``system`` in MIm, MImH, MeOH, H."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "system" not in df.columns or "Etot_last_cycle_kcal_mol" not in df.columns:
        return {}
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        name = str(r["system"]).strip()
        if name not in _MONOMER_REF_SYSTEM_NAMES:
            continue
        e = _parse_tsv_float(r["Etot_last_cycle_kcal_mol"])
        if e is not None:
            out[name] = e
    return out


def _latex_escape(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("_", r"\_")
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    s = s.replace("#", r"\#")
    return s


def _charge_to_latex_superscript(charge: int) -> str:
    """Total complex charge as LaTeX superscript, e.g. ^{2+}, ^{-}."""
    if charge == 0:
        return ""
    sign = "+" if charge > 0 else "-"
    mag = abs(charge)
    if mag == 1:
        return rf"^{{{sign}}}"
    return rf"^{{{mag}{sign}}}"


def _formula_compact_to_latex_math(formula: str) -> str:
    """Format `1Zn_xMIm_yMImH_zMeOH` as bracketed Zn complex: ligands without ion charges; only total q."""
    s = str(formula).strip()
    m = _FORMULA_ZN_RE.match(s)
    if not m:
        return rf"\text{{{_latex_escape(s)}}}"

    n_mim, n_mimh, n_meoh = int(m.group(1)), int(m.group(2)), int(m.group(3))
    q = 2 - n_mim
    parts: List[str] = []
    if n_mim > 0:
        parts.append(
            r"(\mathrm{MeIm})"
            if n_mim == 1
            else rf"(\mathrm{{MeIm}})_{{{n_mim}}}"
        )
    if n_mimh > 0:
        parts.append(
            r"(\mathrm{MeImH})"
            if n_mimh == 1
            else rf"(\mathrm{{MeImH}})_{{{n_mimh}}}"
        )
    if n_meoh > 0:
        parts.append(
            r"(\mathrm{MeOH})"
            if n_meoh == 1
            else rf"(\mathrm{{MeOH}})_{{{n_meoh}}}"
        )

    if not parts:
        inner = r"\mathrm{Zn}"
    else:
        inner = r"\mathrm{Zn}" + "".join(parts)

    return rf"\left[{inner}\right]{_charge_to_latex_superscript(q)}"


def balanced_reaction_latex(edge_type: str, fu: str, fv: str) -> str:
    """Stoichiometric reaction matching network edge_type; R/P as Zn complex LaTeX."""
    r = _formula_compact_to_latex_math(fu)
    p = _formula_compact_to_latex_math(fv)
    if edge_type == "Add MeOH":
        return rf"{r} + \mathrm{{MeOH}} \rightarrow {p}"
    if edge_type == "Add MImH":
        return rf"{r} + \mathrm{{MImH}} \rightarrow {p}"
    if edge_type == "Add MIm(-)":
        return rf"{r} + \mathrm{{MIm}}^{{-}} \rightarrow {p}"
    if edge_type == "Deprotonation (-H+)":
        return rf"{r} \rightarrow {p} + \mathrm{{H}}^{{+}}"
    if edge_type == "Exchange (-MeOH, +MIm(-))":
        return rf"{r} + \mathrm{{MIm}}^{{-}} \rightarrow {p} + \mathrm{{MeOH}}"
    if edge_type == "Exchange (-MeOH, +MImH)":
        return rf"{r} + \mathrm{{MImH}} \rightarrow {p} + \mathrm{{MeOH}}"
    return rf"{r} \rightarrow {p}"


def balanced_reaction_plain(edge_type: str, fu: str, fv: str) -> str:
    """ASCII/plain form for TSV."""
    if edge_type == "Add MeOH":
        return f"{fu} + MeOH -> {fv}"
    if edge_type == "Add MImH":
        return f"{fu} + MImH -> {fv}"
    if edge_type == "Add MIm(-)":
        return f"{fu} + MIm- -> {fv}"
    if edge_type == "Deprotonation (-H+)":
        return f"{fu} -> {fv} + H+"
    if edge_type == "Exchange (-MeOH, +MIm(-))":
        return f"{fu} + MIm- -> {fv} + MeOH"
    if edge_type == "Exchange (-MeOH, +MImH)":
        return f"{fu} + MImH -> {fv} + MeOH"
    return f"{fu} -> {fv}"


def zn_triples_coordination_4_5_6() -> List[Tuple[int, int, int]]:
    """All nonnegative (MIm, MImH, MeOH) with MIm + MImH + MeOH ∈ {4, 5, 6}."""
    out: List[Tuple[int, int, int]] = []
    for n in (4, 5, 6):
        for a in range(n + 1):
            rem = n - a
            for b in range(rem + 1):
                c = rem - b
                out.append((a, b, c))
    return out


def augment_dataframe_missing_grid_cn456(
    df: pd.DataFrame,
    formula_to_id: Dict[str, int],
    *,
    enum_path: Path,
) -> pd.DataFrame:
    """Add placeholder rows for stoichiometries in the CN=4,5,6 grid absent from ``df``.

    Used with ``min_last_convergence.tsv`` so the network plot shows hollow / gray
    ``missing`` nodes for compositions not present in the table. Each placeholder
    ``formula_id`` is taken from ``formula_to_id`` (enumeration TSV).
    """
    if df.empty:
        return df
    present = {
        (int(r["MIm"]), int(r["MImH"]), int(r["MeOH"]))
        for _, r in df.iterrows()
    }
    grid_set = set(zn_triples_coordination_4_5_6())
    missing = sorted(grid_set - present)
    if not missing:
        return df
    cols = list(df.columns)

    def _placeholder(col: str) -> object:
        dt = df[col].dtype
        if pd.api.types.is_float_dtype(dt):
            return float("nan")
        if pd.api.types.is_integer_dtype(dt) and not pd.api.types.is_extension_array_dtype(
            dt
        ):
            return 0
        return pd.NA

    extra_rows: List[Dict[str, object]] = []
    for a, b, c in missing:
        fstr = f"1Zn_{a}MIm_{b}MImH_{c}MeOH"
        if fstr not in formula_to_id:
            raise SystemExit(
                f"CN=4–6 grid placeholder {fstr!r} has no formula_id in {enum_path}"
            )
        row = {col: _placeholder(col) for col in cols}
        row.update(
            {
                "formula_id": int(formula_to_id[fstr]),
                "formula": fstr,
                "MIm": a,
                "MImH": b,
                "MeOH": c,
                "CN": a + b + c,
                "Charge": 2 - a,
                "im_total": a + b,
                "n_conf_passed": 0,
            }
        )
        if "Etot_last_cycle_kcal_mol" in row:
            row["Etot_last_cycle_kcal_mol"] = float("nan")
        extra_rows.append(row)
    extra = pd.DataFrame(extra_rows, columns=cols)
    return pd.concat([df, extra], ignore_index=True)


def triples_reachable_by_ligand_removal(
    stable_triples: set[Tuple[int, int, int]],
) -> set[Tuple[int, int, int]]:
    """All (MIm, MImH, MeOH) tuples reachable from any stable triple by repeatedly
    subtracting 1 from MIm, MImH, or MeOH (non-negative). Equivalently: any tuple
    that lies component-wise below some stable stoichiometry."""
    reachable: set[Tuple[int, int, int]] = set(stable_triples)
    q: deque[Tuple[int, int, int]] = deque(stable_triples)
    while q:
        a, b, c = q.popleft()
        for na, nb, nc in ((a - 1, b, c), (a, b - 1, c), (a, b, c - 1)):
            if na >= 0 and nb >= 0 and nc >= 0 and (na, nb, nc) not in reachable:
                reachable.add((na, nb, nc))
                q.append((na, nb, nc))
    return reachable


def closest_stable_derivation(
    target: Tuple[int, int, int],
    df_active: pd.DataFrame,
) -> Optional[Tuple[int, str, Tuple[int, int, int], List[str]]]:
    """Stable parent with minimum total ligand removals (-MIm/-MImH/-MeOH); tie-break: lower formula_id.

    Returns ``(parent_fid, parent_formula, (A,B,C), steps)`` with ``steps`` like
    ``['-MIm', '-MeOH', ...]`` (order: all -MIm, then -MImH, then -MeOH).
    """
    a, b, c = target
    best_key: Optional[Tuple[int, int]] = None
    best_fid = 0
    best_formula = ""
    best_abc = (0, 0, 0)
    for _, r in df_active.iterrows():
        A, B, C = int(r["MIm"]), int(r["MImH"]), int(r["MeOH"])
        if A < a or B < b or C < c:
            continue
        removals = (A - a) + (B - b) + (C - c)
        fid = int(r["formula_id"])
        key = (removals, fid)
        if best_key is None or key < best_key:
            best_key = key
            best_fid = fid
            best_formula = str(r["formula"])
            best_abc = (A, B, C)
    if best_key is None:
        return None
    A, B, C = best_abc
    steps: List[str] = []
    steps.extend(["-MIm"] * (A - a))
    steps.extend(["-MImH"] * (B - b))
    steps.extend(["-MeOH"] * (C - c))
    return (best_fid, best_formula, (A, B, C), steps)


def print_gray_missing_derivations(
    df_missing: pd.DataFrame,
    df_active: pd.DataFrame,
    derivable_triples: set[Tuple[int, int, int]],
    *,
    out_stream,
) -> None:
    """Stdout: for each missing node shown as gray, one chosen stable parent and removal steps."""
    lines: List[
        Tuple[int, str, Tuple[int, int, int], int, str, Tuple[int, int, int], List[str]]
    ] = []
    for _, row in df_missing.iterrows():
        t = (int(row["MIm"]), int(row["MImH"]), int(row["MeOH"]))
        if t not in derivable_triples:
            continue
        ex = closest_stable_derivation(t, df_active)
        if ex is None:
            continue
        parent_fid, parent_formula, parent_abc, steps = ex
        lines.append(
            (
                int(row["formula_id"]),
                str(row["formula"]),
                t,
                parent_fid,
                parent_formula,
                parent_abc,
                steps,
            )
        )
    lines.sort(key=lambda x: x[0])
    if not lines:
        return

    print("", file=out_stream)
    print(
        "# Gray (derivable missing) structures: one stable parent per row.",
        file=out_stream,
    )
    print(
        "# Rule: among stable stoichiometries that dominate (MIm,MImH,MeOH) component-wise,",
        file=out_stream,
    )
    print(
        "# pick the one with the fewest total ligand removals; tie-break: lower formula_id.",
        file=out_stream,
    )
    print(
        "# Steps list: remove that many MIm / MImH / MeOH from the parent (order: -MIm, then -MImH, then -MeOH).",
        file=out_stream,
    )
    print(
        "# missing_fid\tmissing_formula\tmissing_tuple\tparent_fid\tparent_formula\t"
        "parent_tuple\tn_steps\tsteps_joined",
        file=out_stream,
    )
    for mid, mform, mt, pid, pform, pt, steps in lines:
        joined = ",".join(steps) if steps else "(none — same counts as parent)"
        n_steps = len(steps)
        print(
            f"{mid}\t{mform}\t{mt[0]},{mt[1]},{mt[2]}\t"
            f"{pid}\t{pform}\t{pt[0]},{pt[1]},{pt[2]}\t{n_steps}\t{joined}",
            file=out_stream,
        )
    print("", file=out_stream)
    print("# Human-readable (same rows):", file=out_stream)
    for mid, mform, mt, pid, pform, pt, steps in lines:
        if steps:
            how = " then ".join(steps)
            print(
                f"#   missing ID {mid} {mform} {mt}  <=  stable ID {pid} {pform} {pt}  |  remove: {how}",
                file=out_stream,
            )
        else:
            print(
                f"#   missing ID {mid} {mform} {mt}  |  same (MIm,MImH,MeOH) as stable ID {pid} {pform}",
                file=out_stream,
            )
    print(f"# Total gray (derivable missing) entries: {len(lines)}", file=out_stream)


def _parse_lowest_e(val: object) -> Optional[float]:
    """Parse `lowest_E` from ana.log: energy in Hartree, or 'none'."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    s = str(val).strip()
    if not s or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# eu, ev, dE in kcal/mol; dE_rxn optional (stoichiometric ΔE when refs suffice)
ReactionDERow = Tuple[
    str, int, int, str, str, Optional[float], Optional[float], Optional[float], Optional[float]
]


def collect_reaction_dE_rows(
    G: nx.DiGraph,
    fid_to_e: Dict[int, Optional[float]],
    fid_to_formula: Dict[int, str],
) -> List[Tuple[str, int, int, str, str, Optional[float], Optional[float], Optional[float]]]:
    """One row per directed edge: eu, ev, dE in Hartree (from ana.log lowest_E)."""
    rows: List[
        Tuple[str, int, int, str, str, Optional[float], Optional[float], Optional[float]]
    ] = []
    for u, v, d in G.edges(data=True):
        et = str(d["type"])
        iu, iv = int(u), int(v)
        fu = fid_to_formula.get(iu, "")
        fv = fid_to_formula.get(iv, "")
        eu, ev = fid_to_e.get(iu), fid_to_e.get(iv)
        de = (ev - eu) if (eu is not None and ev is not None) else None
        rows.append((et, iu, iv, fu, fv, eu, ev, de))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows


def reaction_rows_cluster_kcal_only(
    G: nx.DiGraph,
    fid_to_e_kcal: Dict[int, Optional[float]],
    fid_to_formula: Dict[int, str],
) -> List[ReactionDERow]:
    """Per edge: ΔE_cluster = E_tot(product) - E_tot(reactant) in kcal/mol (ORCA totals)."""
    rows: List[ReactionDERow] = []
    for u, v, d in G.edges(data=True):
        et = str(d["type"])
        iu, iv = int(u), int(v)
        fu = fid_to_formula.get(iu, "")
        fv = fid_to_formula.get(iv, "")
        eu, ev = fid_to_e_kcal.get(iu), fid_to_e_kcal.get(iv)
        de_k = (ev - eu) if (eu is not None and ev is not None) else None
        rows.append((et, iu, iv, fu, fv, eu, ev, de_k, None))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows


def compute_de_rxn_kcal(
    edge_type: str,
    eu_kcal: float,
    ev_kcal: float,
    ref_kcal: Dict[str, float],
) -> Optional[float]:
    """Stoichiometric ΔE_rxn in kcal/mol; monomer E_tot from same TSV as clusters.

    Cluster step energy change d = ev - eu (kcal/mol). Formulas match balanced_reaction_*.
    """
    d = ev_kcal - eu_kcal
    if edge_type == "Add MIm(-)":
        e = ref_kcal.get("MIm")
        return None if e is None else d - e
    if edge_type == "Add MImH":
        e = ref_kcal.get("MImH")
        return None if e is None else d - e
    if edge_type == "Deprotonation (-H+)":
        e = ref_kcal.get("H")
        return None if e is None else d + e
    if edge_type == "Add MeOH":
        e = ref_kcal.get("MeOH")
        return None if e is None else d - e
    if edge_type == "Exchange (-MeOH, +MIm(-))":
        e_mim = ref_kcal.get("MIm")
        e_meoh = ref_kcal.get("MeOH")
        if e_mim is None or e_meoh is None:
            return None
        return d - e_mim + e_meoh
    if edge_type == "Exchange (-MeOH, +MImH)":
        e_mimh = ref_kcal.get("MImH")
        e_meoh = ref_kcal.get("MeOH")
        if e_mimh is None or e_meoh is None:
            return None
        return d - e_mimh + e_meoh
    return None


def apply_monomer_rxn_kcal(
    rows: List[ReactionDERow],
    ref_kcal: Dict[str, float],
) -> List[ReactionDERow]:
    """Fill ``de_rxn`` (9th field) using ``ref_kcal`` when monomer rows exist in TSV."""
    if not ref_kcal:
        return rows
    out: List[ReactionDERow] = []
    for et, iu, iv, fu, fv, eu, ev, de_k, _ in rows:
        de_rxn: Optional[float] = None
        if eu is not None and ev is not None:
            de_rxn = compute_de_rxn_kcal(et, float(eu), float(ev), ref_kcal)
        out.append((et, iu, iv, fu, fv, eu, ev, de_k, de_rxn))
    return out


def hartree_rows_to_kcal_mol(
    rows_ha: List[
        Tuple[str, int, int, str, str, Optional[float], Optional[float], Optional[float]]
    ],
) -> List[ReactionDERow]:
    """Convert eu, ev, dE from Hartree to kcal/mol (ana.log); no monomer correction."""
    c = HARTREE_TO_KCAL_MOL
    out: List[ReactionDERow] = []
    for et, iu, iv, fu, fv, eu, ev, de in rows_ha:
        eu_k = eu * c if eu is not None else None
        ev_k = ev * c if ev is not None else None
        de_k = de * c if de is not None else None
        out.append((et, iu, iv, fu, fv, eu_k, ev_k, de_k, None))
    return out


def print_reaction_dE_table(
    rows: List[ReactionDERow],
    *,
    out_stream,
    from_convergence_tsv: bool = False,
    ref_monomer_kcal: Optional[Dict[str, float]] = None,
) -> None:
    """Print TSV: reaction rows and ΔE in kcal/mol."""
    print("", file=out_stream)
    if from_convergence_tsv:
        print(
            "# dE_cluster_kcal_mol = E_tot(to) - E_tot(from) from Etot_last_cycle_kcal_mol.",
            file=out_stream,
        )
        print(
            "# dE_rxn_kcal_mol = stoichiometric ΔE using monomer E_tot from the same TSV "
            "(rows system = MIm, MImH, MeOH, H); N/A if a needed monomer is missing.",
            file=out_stream,
        )
        print(
            "#   Add MIm(-):      (E(P)-E(R)) - E(MIm);  Add MImH: (E(P)-E(R)) - E(MImH);",
            file=out_stream,
        )
        print(
            "#   Deprotonation:   (E(P)-E(R)) + E(H);    Add MeOH: (E(P)-E(R)) - E(MeOH);",
            file=out_stream,
        )
        print(
            "#   Exch. +MIm-:     (E(P)-E(R)) - E(MIm) + E(MeOH);  "
            "Exch. +MImH: (E(P)-E(R)) - E(MImH) + E(MeOH).",
            file=out_stream,
        )
        if ref_monomer_kcal:
            print("# Monomer E_tot reference (kcal/mol) from TSV:", file=out_stream)
            for k in ("MImH", "MIm", "MeOH", "H"):
                if k not in ref_monomer_kcal:
                    continue
                print(
                    f"#   {k}: {ref_monomer_kcal[k]:.8f}",
                    file=out_stream,
                )
        hdr = (
            "# reaction_type\tfrom_fid\tto_fid\tfrom_formula\tto_formula\t"
            "balanced_reaction\tdE_cluster_kcal_mol\tdE_rxn_kcal_mol"
        )
    else:
        print(
            "# dE_cluster_kcal_mol from ana.log lowest_E (Hartree) × "
            f"{HARTREE_TO_KCAL_MOL:.6f}; no monomer correction in this mode.",
            file=out_stream,
        )
        hdr = (
            "# reaction_type\tfrom_fid\tto_fid\tfrom_formula\tto_formula\t"
            "balanced_reaction\tdE_cluster_kcal_mol"
        )
    print(hdr, file=out_stream)
    for et, iu, iv, fu, fv, eu, ev, de, de_rxn in rows:
        dc_s = f"{de:.12f}" if de is not None else "N/A"
        bal = balanced_reaction_plain(et, fu, fv)
        if from_convergence_tsv:
            rx_s = f"{de_rxn:.12f}" if de_rxn is not None else "N/A"
            print(
                f"{et}\t{iu}\t{iv}\t{fu}\t{fv}\t{bal}\t{dc_s}\t{rx_s}",
                file=out_stream,
            )
        else:
            print(
                f"{et}\t{iu}\t{iv}\t{fu}\t{fv}\t{bal}\t{dc_s}",
                file=out_stream,
            )
    print(f"# Total reactions: {len(rows)}", file=out_stream)


def _sanitize_type_for_filename(s: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in s)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return (slug[:80] if slug else "reaction")


def _latex_document_reaction_dE_for_type(
    rows: List[ReactionDERow],
    reaction_type: str,
    *,
    from_convergence_tsv: bool = False,
    ref_monomer_kcal: Optional[Dict[str, float]] = None,
) -> str:
    """LaTeX longtable: #, ID→ID, balanced reaction, ΔE (one reaction class per document)."""
    lines: List[str] = []
    lines.append(r"\documentclass[10pt]{article}")
    lines.append(r"\usepackage[margin=10mm]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\usepackage{array}")
    lines.append(r"\usepackage{amsmath}")
    lines.append(r"\usepackage{caption}")
    lines.append(r"\captionsetup{font=small}")
    lines.append(r"\setlength{\LTpre}{0pt}")
    lines.append(r"\setlength{\LTpost}{0pt}")
    lines.append(r"\begin{document}")
    lines.append(r"\small")
    lines.append(r"\begin{center}")
    lines.append(r"\captionsetup{labelformat=empty}")
    et_e = _latex_escape(reaction_type)
    if from_convergence_tsv:
        dlabel = r"$\Delta E$ (kcal\,mol$^{-1}$)"
        cap = (
            r"\captionof{table}{"
            + dlabel
            + r" (\texttt{"
            + et_e
            + r"}): prefer stoichiometric $\Delta E_{\mathrm{rxn}}$ when monomer "
            r"rows exist in \texttt{min\_last\_convergence.tsv}, else cluster "
            r"$E_{\mathrm{tot}}(\mathrm{to}) - E_{\mathrm{tot}}(\mathrm{from})$.}"
        )
    else:
        dlabel = r"$\Delta E_{\mathrm{cluster}}$"
        cap = (
            r"\captionof{table}{"
            + dlabel
            + r" (\texttt{"
            + et_e
            + r"}) in kcal\,mol$^{-1}$; from \texttt{lowest\_E} (Hartree $\times "
            + f"{HARTREE_TO_KCAL_MOL:.6f}"
            + r").}"
        )
    colhdr = dlabel + r" \\"
    lines.append(cap)
    lines.append(r"\end{center}")
    lines.append(r"\vspace{1mm}")
    lines.append(
        r"\begin{longtable}{r p{0.12\textwidth} p{0.58\textwidth} r}"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\# & ID $\rightarrow$ ID & balanced reaction & " + colhdr
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(
        r"\# & ID $\rightarrow$ ID & balanced reaction & " + colhdr
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")

    for i, (_et, iu, iv, fu, fv, eu, ev, de, de_rxn) in enumerate(rows, start=1):
        idcol = rf"$\mathrm{{ID}}\,{iu} \rightarrow \mathrm{{ID}}\,{iv}$"
        bal = balanced_reaction_latex(_et, fu, fv)
        formcol = rf"$\displaystyle {bal}$"
        if from_convergence_tsv:
            if de_rxn is not None:
                ddr = f"{de_rxn:+.6f}"
            elif de is not None:
                ddr = f"{de:+.6f}"
            else:
                ddr = "---"
        else:
            ddr = f"{de:+.6f}" if de is not None else "---"
        lines.append(f"{i:d} & {idcol} & {formcol} & {ddr} \\\\")

    lines.append(r"\end{longtable}")
    if from_convergence_tsv:
        lines.append(
            r"\textit{Note:} Monomer $E_{\mathrm{tot}}$ from the same TSV "
            r"(\texttt{MIm}, \texttt{MImH}, \texttt{MeOH}, \texttt{H}). "
            r"Balanced reactions use $\mathrm{MeOH}$, $\mathrm{MImH}$, "
            r"$\mathrm{MIm}^{-}$, $\mathrm{H}^{+}$; $q = 2 - n_{\mathrm{MeIm}}$."
        )
    else:
        lines.append(
            r"\textit{Note:} Values are cluster total-energy differences only "
            r"(no monomer correction for \texttt{ana.log} input). "
            r"Complex charge $q = 2 - n_{\mathrm{MeIm}}$."
        )
    if from_convergence_tsv and ref_monomer_kcal:
        parts = []
        for k in ("MImH", "MIm", "MeOH", "H"):
            if k not in ref_monomer_kcal:
                continue
            ek = ref_monomer_kcal[k]
            parts.append(
                rf"$E(\mathrm{{{k}}}) = {ek:.4f}$ kcal\,mol$^{{-1}}$"
            )
        lines.append(
            r"\textit{Monomer references (ORCA, same TSV):} " + r";\, ".join(parts) + "."
        )
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def _render_pdf_to_single_png(pdf_path: Path, png_path: Path, dpi: int = 200) -> None:
    try:
        import fitz  # PyMuPDF
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "PyMuPDF is required to render PDF to PNG. Install with:\n"
            "  python3 -m pip install --user pymupdf"
        ) from e
    try:
        from PIL import Image
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Pillow is required. Install with:\n  python3 -m pip install --user pillow"
        ) from e

    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    images: list = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)

    if not images:
        raise RuntimeError(f"No pages rendered from PDF: {pdf_path}")

    width = max(im.width for im in images)
    height = sum(im.height for im in images)
    stitched = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for im in images:
        stitched.paste(im, (0, y))
        y += im.height

    png_path.parent.mkdir(parents=True, exist_ok=True)
    stitched.save(png_path, format="PNG")


def write_reaction_dE_latex_pngs_by_type(
    rows: List[ReactionDERow],
    out_png: Path,
    *,
    dpi: int,
    from_convergence_tsv: bool = False,
    ref_monomer_kcal: Optional[Dict[str, float]] = None,
) -> List[Path]:
    """One LaTeX PDF/PNG per reaction type; filenames: {stem}_{type_slug}.png."""
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError(
            "pdflatex not found in PATH. Install a LaTeX distribution (e.g. TeX Live) "
            "or pass --no-de-table-png to skip this step."
        )

    by_type: Dict[str, List[ReactionDERow]] = defaultdict(list)
    for row in rows:
        by_type[row[0]].append(row)
    for et in by_type:
        by_type[et].sort(key=lambda r: (r[1], r[2]))

    base = out_png if out_png.is_absolute() else _workdir() / out_png
    stem = base.stem
    parent = base.parent
    written: List[Path] = []

    for reaction_type in sorted(by_type.keys()):
        subrows = by_type[reaction_type]
        slug = _sanitize_type_for_filename(reaction_type)
        out_abs = parent / f"{stem}_{slug}.png"
        tex_body = _latex_document_reaction_dE_for_type(
            subrows,
            reaction_type,
            from_convergence_tsv=from_convergence_tsv,
            ref_monomer_kcal=ref_monomer_kcal,
        )

        with tempfile.TemporaryDirectory(prefix="react_net_dE_") as td:
            tdir = Path(td)
            tex_path = tdir / "react_net_dE.tex"
            pdf_path = tdir / "react_net_dE.pdf"
            tex_path.write_text(tex_body, encoding="utf-8")

            for _ in range(2):
                proc = subprocess.run(
                    [
                        pdflatex,
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-output-directory",
                        str(tdir),
                        str(tex_path),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"pdflatex failed for type {reaction_type!r}.\n"
                        + (proc.stdout or "")
                    )

            if not pdf_path.exists():
                raise RuntimeError("pdflatex did not produce the expected PDF.")

            _render_pdf_to_single_png(pdf_path=pdf_path, png_path=out_abs, dpi=dpi)
        written.append(out_abs)

    return written


def _resolve_input_path(p: Path) -> Path:
    r = p.resolve()
    if r.is_file():
        return r
    raise SystemExit(f"Input file not found: {p}")


def write_etot_last_cycle_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter im_total vs MeOH, color = E_tot last cycle (kcal/mol); needs numeric column."""
    if "Etot_last_cycle_kcal_mol" not in df.columns:
        return
    sub = df[
        pd.notna(df["Etot_last_cycle_kcal_mol"])
        & df["formula"].astype(str).str.match(_FORMULA_ZN_RE)
    ].copy()
    if sub.empty:
        return
    et = sub["Etot_last_cycle_kcal_mol"].astype(float)
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        sub["im_total"],
        sub["MeOH"],
        c=et,
        s=140,
        cmap="viridis",
        edgecolors="#333333",
        linewidths=0.6,
        alpha=0.92,
    )
    plt.colorbar(sc, ax=ax, label=r"$E_{\mathrm{tot}}$ last cycle (kcal mol$^{-1}$)")
    ax.set_xlabel("Total imidazole (MIm + MImH)", fontsize=12, fontweight="bold")
    ax.set_ylabel("MeOH count", fontsize=12, fontweight="bold")
    ax.set_title(
        "ORCA minimized clusters: total energy (last geom. opt. cycle)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(range(int(sub["im_total"].min()), int(sub["im_total"].max()) + 1))
    ax.set_yticks(range(int(sub["MeOH"].min()), int(sub["MeOH"].max()) + 1))
    plt.tight_layout()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


_COL_ZN_N = "max_Zn-Nmin_MIM(Å)"
_COL_ZN_O = "max_Zn-O_MeOH(Å)"


def write_zn_ligand_distance_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter max Zn–N(MIm) vs max Zn–O(MeOH) when both distances are numeric."""
    if _COL_ZN_N not in df.columns or _COL_ZN_O not in df.columns:
        return
    sub = df[df["formula"].astype(str).str.match(_FORMULA_ZN_RE)].copy()
    rn = sub[_COL_ZN_N].map(_parse_tsv_float)
    ro = sub[_COL_ZN_O].map(_parse_tsv_float)
    sub = sub.assign(_zn_n=rn, _zn_o=ro)
    sub = sub[pd.notna(sub["_zn_n"]) & pd.notna(sub["_zn_o"])]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 7))
    if "Etot_last_cycle_kcal_mol" in sub.columns and sub["Etot_last_cycle_kcal_mol"].notna().any():
        et = sub["Etot_last_cycle_kcal_mol"].astype(float)
        sc = ax.scatter(
            sub["_zn_n"],
            sub["_zn_o"],
            c=et,
            s=120,
            cmap="coolwarm",
            edgecolors="#222222",
            linewidths=0.5,
            alpha=0.9,
        )
        plt.colorbar(sc, ax=ax, label=r"$E_{\mathrm{tot}}$ last cycle (kcal mol$^{-1}$)")
    else:
        ax.scatter(sub["_zn_n"], sub["_zn_o"], s=120, c="#4477aa", edgecolors="#222222", alpha=0.9)
    ax.set_xlabel(r"max Zn–N (MIm), Å", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"max Zn–O (MeOH), Å", fontsize=12, fontweight="bold")
    ax.set_title("Zn–ligand distances (last minimized geometry)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_file = args.data_file
    if data_file is None:
        data_file = _workdir() / "min_last_convergence.tsv"
    data_path = _resolve_input_path(data_file)

    formula_to_id = load_formula_id_map(_MSMILES_ENUM_TSV)

    if is_min_last_convergence_tsv(data_path):
        df = read_min_last_convergence_dataframe(data_path)
        energy_footer = (
            f"Energies: {data_path.name}, column Etot_last_cycle_kcal_mol "
            f"(ΔE on edges = difference of those totals, kcal/mol)"
        )
        from_convergence_tsv = True
    else:
        data_str = extract_formula_table_text(data_path)
        df = read_formula_dataframe(data_str)
        energy_footer = (
            f"ΔE (stdout / LaTeX): kcal/mol from Hartree × {HARTREE_TO_KCAL_MOL:.6f}"
        )
        from_convergence_tsv = False

    df = assign_formula_ids_from_enumeration(
        df, formula_to_id, src_desc=str(data_path)
    )

    pattern = r"^1Zn_(\d+)MIm_(\d+)MImH_(\d+)MeOH$"

    def parse_formula(f: str) -> tuple[int, int, int]:
        match = re.match(pattern, str(f))
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return 0, 0, 0

    df[["MIm", "MImH", "MeOH"]] = df["formula"].apply(
        lambda x: pd.Series(parse_formula(x))
    )
    df["CN"] = df["MIm"] + df["MImH"] + df["MeOH"]
    df["Charge"] = 2 - df["MIm"]
    df["im_total"] = df["MIm"] + df["MImH"]

    if from_convergence_tsv:
        df = augment_dataframe_missing_grid_cn456(
            df, formula_to_id, enum_path=_MSMILES_ENUM_TSV
        )

    if SCREEN_CHARGE is not None:
        n_before = len(df)
        df = df.loc[df["Charge"] >= SCREEN_CHARGE].copy()
        if df.empty:
            raise SystemExit(
                f"SCREEN_CHARGE={SCREEN_CHARGE}: no rows left with charge >= {SCREEN_CHARGE} "
                f"(filtered out all {n_before} row(s))."
            )

    # Separate nodes
    df_active = df[df["n_conf_passed"] > 0].copy()
    df_missing = df[df["n_conf_passed"] == 0].copy()

    stable_triples: set[Tuple[int, int, int]] = {
        (int(r["MIm"]), int(r["MImH"]), int(r["MeOH"]))
        for _, r in df_active.iterrows()
    }
    derivable_triples = triples_reachable_by_ligand_removal(stable_triples)

    if not args.no_print_gray_derivation:
        print_gray_missing_derivations(
            df_missing, df_active, derivable_triples, out_stream=sys.stdout
        )

    # Build directed graph (NetworkX)
    G = nx.DiGraph()

    for _, row in df.iterrows():
        y_pos = row["MeOH"] * 24 + row["Charge"] * 5
        x_pos = row["im_total"] + row["Charge"] * 0.25

        status = "active" if row["n_conf_passed"] > 0 else "missing"
        label = (
            f"ID:{row['formula_id']}\n({row['Charge']:+d})" if status == "active" else ""
        )

        if status == "missing":
            t = (int(row["MIm"]), int(row["MImH"]), int(row["MeOH"]))
            missing_class = "derivable" if t in derivable_triples else "isolated"
        else:
            missing_class = None

        unconverged = (
            from_convergence_tsv
            and COLOR_UNCOVERGED
            and status == "active"
            and _row_unconverged_by_loose_criteria(row)
        )

        G.add_node(
            row["formula_id"],
            pos=(x_pos, y_pos),
            label=label,
            charge=row["Charge"],
            status=status,
            missing_class=missing_class,
            unconverged=unconverged,
        )

    node_data = df_active.set_index("formula_id").to_dict("index")
    node_ids = list(node_data.keys())

    for i in range(len(node_ids)):
        for j in range(len(node_ids)):
            if i == j:
                continue
            id1, id2 = node_ids[i], node_ids[j]
            n1, n2 = node_data[id1], node_data[id2]

            dA, dB, dC = (
                n2["MIm"] - n1["MIm"],
                n2["MImH"] - n1["MImH"],
                n2["MeOH"] - n1["MeOH"],
            )

            edge_type = None
            if (dA, dB, dC) == (0, 0, 1):
                edge_type = "Add MeOH"
            elif (dA, dB, dC) == (0, 1, 0):
                edge_type = "Add MImH"
            elif (dA, dB, dC) == (1, 0, 0):
                edge_type = "Add MIm(-)"
            elif (dA, dB, dC) == (1, -1, 0):
                edge_type = "Deprotonation (-H+)"
            elif (dA, dB, dC) == (1, 0, -1):
                edge_type = "Exchange (-MeOH, +MIm(-))"
            elif (dA, dB, dC) == (0, 1, -1):
                edge_type = "Exchange (-MeOH, +MImH)"

            if edge_type:
                G.add_edge(id1, id2, type=edge_type)

    fid_to_formula = {int(r["formula_id"]): str(r["formula"]) for _, r in df.iterrows()}

    ref_monomer_kcal: Optional[Dict[str, float]] = None
    if from_convergence_tsv:
        ref_monomer_kcal = load_monomer_refs_kcal_from_tsv(data_path)
        fid_to_e_kcal: Dict[int, Optional[float]] = {}
        for _, r in df.iterrows():
            fid = int(r["formula_id"])
            v = r["Etot_last_cycle_kcal_mol"]
            fid_to_e_kcal[fid] = None if pd.isna(v) else float(v)
        base_rows = reaction_rows_cluster_kcal_only(G, fid_to_e_kcal, fid_to_formula)
        dE_rows = apply_monomer_rxn_kcal(base_rows, ref_monomer_kcal)
    else:
        if "lowest_E" in df.columns:
            fid_to_e = {
                int(r["formula_id"]): _parse_lowest_e(r["lowest_E"])
                for _, r in df.iterrows()
            }
        else:
            fid_to_e = {int(r["formula_id"]): None for _, r in df.iterrows()}
        rows_ha = collect_reaction_dE_rows(G, fid_to_e, fid_to_formula)
        dE_rows = hartree_rows_to_kcal_mol(rows_ha)

    if not args.no_print_reactions:
        print_reaction_dE_table(
            dE_rows,
            out_stream=sys.stdout,
            from_convergence_tsv=from_convergence_tsv,
            ref_monomer_kcal=ref_monomer_kcal,
        )

    if not args.no_de_table_png:
        de_out = args.de_table_png
        if not de_out.is_absolute():
            de_out = _workdir() / de_out
        try:
            paths = write_reaction_dE_latex_pngs_by_type(
                dE_rows,
                de_out,
                dpi=args.latex_dpi,
                from_convergence_tsv=from_convergence_tsv,
                ref_monomer_kcal=ref_monomer_kcal,
            )
            for p in paths:
                print(f"Wrote LaTeX ΔE table PNG to {p}")
        except Exception as e:
            print(f"WARNING: could not build LaTeX ΔE PNG ({e})", file=sys.stderr)

    fig = plt.figure(figsize=(14, 10))
    ax = plt.gca()

    max_meoh = max(6, int(df["MeOH"].max()))
    colors = ["#f4f6f9", "#ffffff"]
    for i in range(max_meoh + 1):
        y_center = i * 24
        ax.axhspan(y_center - 11, y_center + 11, facecolor=colors[i % 2], alpha=0.8, zorder=0)
        ax.text(
            -0.6,
            y_center,
            f"{i} MeOH Floor",
            fontsize=12,
            color="#888888",
            fontweight="bold",
            va="center",
        )

    pos = nx.get_node_attributes(G, "pos")

    edge_colors = {
        "Add MeOH": ("#4a90e2", "dashed", 0.08, 0.6),
        "Add MImH": ("#50e3c2", "dashed", 0.12, 0.7),
        "Add MIm(-)": ("#417505", "dashdot", 0.12, 0.7),
        "Deprotonation (-H+)": ("#d0021b", "solid", 0.25, 0.8),
        "Exchange (-MeOH, +MIm(-))": ("#f5a623", "dotted", 0.18, 0.8),
        "Exchange (-MeOH, +MImH)": ("#8b572a", "dotted", 0.18, 0.8),
    }

    for e_type, (color, style, rad, alpha_val) in edge_colors.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == e_type]
        if edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color=color,
                style=style,
                arrows=True,
                arrowsize=10,
                connectionstyle=f"arc3,rad={rad}",
                alpha=alpha_val,
            )

    charge_palette = {
        2: "#ffb3ba",
        1: "#ffdfba",
        0: "#ffffba",
        -1: "#baffc9",
        -2: "#bae1ff",
        -3: "#d3baff",
        -4: "#f0e6fa",
    }

    active_nodes = [n for n, attr in G.nodes(data=True) if attr["status"] == "active"]
    missing_isolated = [
        n
        for n, attr in G.nodes(data=True)
        if attr["status"] == "missing" and attr.get("missing_class") == "isolated"
    ]
    missing_derivable = [
        n
        for n, attr in G.nodes(data=True)
        if attr["status"] == "missing" and attr.get("missing_class") == "derivable"
    ]

    # Missing, not a sub-stoichiometry of any stable composition: hollow dashed circles
    if missing_isolated:
        isolated_plot = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=missing_isolated,
            node_size=400,
            node_color="none",
            edgecolors="#888888",
            linewidths=2.0,
        )
        if isolated_plot is not None:
            n_iso = len(missing_isolated)
            try:
                isolated_plot.set_linestyles(["--"] * n_iso)
            except (AttributeError, TypeError):
                try:
                    isolated_plot.set_linestyle("--")
                except (AttributeError, TypeError):
                    pass

    # Missing but (MIm,MImH,MeOH) obtainable by removing ligands from some stable structure
    if PLOT_DERIVED_STRUC and missing_derivable:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=missing_derivable,
            node_size=400,
            node_color="#c8c8c8",
            edgecolors="#6a6a6a",
            linewidths=1.2,
        )

    default_active_edge = "#333333"
    active_node_colors = [charge_palette[G.nodes[n]["charge"]] for n in active_nodes]
    if COLOR_UNCOVERGED:
        active_edgecolors = [
            _UNCONVERGED_EDGECOLOR if G.nodes[n].get("unconverged") else default_active_edge
            for n in active_nodes
        ]
        active_linewidths = [
            2.2 if G.nodes[n].get("unconverged") else 1.2 for n in active_nodes
        ]
    else:
        active_edgecolors = default_active_edge
        active_linewidths = 1.2
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=active_nodes,
        node_size=600,
        node_color=active_node_colors,
        edgecolors=active_edgecolors,
        linewidths=active_linewidths,
    )

    labels = {n: G.nodes[n]["label"] for n in active_nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=7.5,
        font_family="sans-serif",
        font_weight="bold",
    )

    plt.title(
        "Zinc-Imidazole-Methanol Complex Reaction Network\n(X = Total Imidazole Count)",
        fontsize=16,
        pad=20,
        fontweight="bold",
        color="#333333",
    )
    plt.xlabel(
        "Total Imidazole Ligands (MIm + MImH)",
        fontsize=12,
        fontweight="bold",
        color="#555555",
        labelpad=10,
    )

    max_im = max(6, int(df["im_total"].max()))
    plt.xticks(range(max_im + 1), [f"{i} Im" for i in range(max_im + 1)], fontsize=11, color="#555555")
    plt.yticks([])

    plt.xlim(-0.8, max_im + 0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    legend_elements = [
        plt.Line2D([0], [0], color=c, lw=2, linestyle=s, label=t, alpha=a)
        for t, (c, s, r, a) in edge_colors.items()
    ]
    if PLOT_DERIVED_STRUC:
        legend_elements.append(
            Patch(
                facecolor="#c8c8c8",
                edgecolor="#6a6a6a",
                linewidth=1.2,
                label="Missing (sub-stoichiometry of stable)",
            )
        )
    legend_elements.append(
        Patch(
            facecolor="none",
            edgecolor="#bbbbbb",
            linewidth=1.5,
            linestyle="--",
            label="Missing (not derivable)",
        )
    )
    if (
        from_convergence_tsv
        and COLOR_UNCOVERGED
        and any(G.nodes[n].get("unconverged") for n in active_nodes)
    ):
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#ffffba",
                markeredgecolor=_UNCONVERGED_EDGECOLOR,
                markersize=9,
                markeredgewidth=2.0,
                linestyle="None",
                label=f"Active: {_CONVERGED_LOOSE_COL} = no",
            )
        )

    plt.legend(
        handles=legend_elements,
        loc="upper right",
        title="Reaction Types / nodes",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="#dddddd",
    )

    plt.tight_layout(rect=[0, 0.055, 1, 0.98])
    fig.text(
        0.5,
        0.012,
        energy_footer,
        ha="center",
        fontsize=9,
        color="#666666",
    )
    out = args.output
    if not out.is_absolute():
        out = _workdir() / out
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)

    if from_convergence_tsv and not args.no_etot_scatter_png:
        etot_png = out.parent / "react_net_etot_last_cycle.png"
        write_etot_last_cycle_scatter(df, etot_png)
        print(f"Wrote {etot_png}")
        dist_png = out.parent / "react_net_zn_ligand_distances.png"
        write_zn_ligand_distance_scatter(df, dist_png)
        if dist_png.is_file():
            print(f"Wrote {dist_png}")


if __name__ == "__main__":
    main()
