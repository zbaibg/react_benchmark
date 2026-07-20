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
# The TSV writes them with the "_monomer" suffix; we strip it to the canonical species name.
_MONOMER_SUFFIX = "_monomer"
_MONOMER_REF_SYSTEM_NAMES = frozenset({"MeOH_monomer", "NO3_monomer", "Zn_monomer"})

# If set to an int, drop rows with complex charge q < SCREEN_CHARGE (q = 2 - n_NO3) from the
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
# column ``formula`` -> ``formula_id`` (see ``zn_no3_meoh_msmiles_enumeration.tsv``).
_MSMILES_ENUM_TSV = (
    Path(__file__).resolve().parent.parent.parent / "zn_no3_meoh_msmiles_enumeration.tsv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot zinc-nitrate-methanol reaction network from "
        "min_last_convergence.tsv (default, ORCA QM minimize summary) or "
        "a compatible TSV with a formula column. Stoichiometry formula_id is "
        f"always taken from {_MSMILES_ENUM_TSV.name} (formula column), not from the input file."
    )
    p.add_argument(
        "data_file",
        type=Path,
        nargs="?",
        default=None,
        help="Input path (default: min_last_convergence.tsv in the current working directory). "
        "If that file is missing, pass a formula TSV (with a formula column).",
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
    """Extract a formula table block (header starting with formula_id) from a compatible TSV."""
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
    return lines[start] + "\n" + "\n".join(out_lines)


def read_formula_dataframe(data_str: str) -> pd.DataFrame:
    header_line = data_str.splitlines()[0]
    sep = "\t" if header_line.count("\t") >= 4 else r"\s+"
    df = pd.read_csv(io.StringIO(data_str), sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


_FORMULA_ZN_RE = re.compile(
    r"^1Zn_(\d+)NO3_(\d+)MeOH$",
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

    Keeps only rows whose ``system`` matches ``1Zn_*NO3_*MeOH`` (drops bare Zn /
    monomer / fragment lines). Energies stay in ``Etot_last_cycle_kcal_mol``
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
        raise SystemExit(f"{path}: no rows matching 1Zn_*NO3_*MeOH after filter")

    etot_k = df["Etot_last_cycle_kcal_mol"].map(_parse_tsv_float)
    df["Etot_last_cycle_kcal_mol"] = etot_k
    df = df.sort_values("formula", kind="stable").reset_index(drop=True)
    df["n_conf_passed"] = etot_k.notna().astype(int)

    return df


def load_formula_id_map(enum_path: Path) -> Dict[str, int]:
    """``formula`` string -> ``formula_id`` from the enumeration TSV (unique per formula)."""
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
    """Monomer E_tot (kcal/mol) from rows with ``system`` in MeOH_monomer, NO3_monomer.

    Returned keys use the bare species name (``MeOH``, ``NO3``, ``Zn``) for
    convenience downstream.
    """
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
        if e is None:
            continue
        key = name[: -len(_MONOMER_SUFFIX)] if name.endswith(_MONOMER_SUFFIX) else name
        out[key] = e
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
    """Format `1Zn_xNO3_yMeOH` as bracketed Zn complex: ligands without ion charges; only total q."""
    s = str(formula).strip()
    m = _FORMULA_ZN_RE.match(s)
    if not m:
        return rf"\text{{{_latex_escape(s)}}}"

    n_no3, n_meoh = int(m.group(1)), int(m.group(2))
    q = 2 - n_no3
    parts: List[str] = []
    if n_no3 > 0:
        parts.append(
            r"(\mathrm{NO_3})"
            if n_no3 == 1
            else rf"(\mathrm{{NO_3}})_{{{n_no3}}}"
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
    if edge_type == "Add NO3(-)":
        return rf"{r} + \mathrm{{NO_3}}^{{-}} \rightarrow {p}"
    if edge_type == "Exchange (-MeOH, +NO3(-))":
        return rf"{r} + \mathrm{{NO_3}}^{{-}} \rightarrow {p} + \mathrm{{MeOH}}"
    return rf"{r} \rightarrow {p}"


def balanced_reaction_plain(edge_type: str, fu: str, fv: str) -> str:
    """ASCII/plain form for TSV."""
    if edge_type == "Add MeOH":
        return f"{fu} + MeOH -> {fv}"
    if edge_type == "Add NO3(-)":
        return f"{fu} + NO3- -> {fv}"
    if edge_type == "Exchange (-MeOH, +NO3(-))":
        return f"{fu} + NO3- -> {fv} + MeOH"
    return f"{fu} -> {fv}"


def zn_pairs_from_enumeration(formula_to_id: Dict[str, int]) -> List[Tuple[int, int]]:
    """All (n_NO3, n_MeOH) pairs present in the enumeration TSV."""
    pairs: set[Tuple[int, int]] = set()
    for f in formula_to_id.keys():
        m = _FORMULA_ZN_RE.match(f)
        if m:
            pairs.add((int(m.group(1)), int(m.group(2))))
    return sorted(pairs)


def augment_dataframe_missing_grid(
    df: pd.DataFrame,
    formula_to_id: Dict[str, int],
    *,
    enum_path: Path,
) -> pd.DataFrame:
    """Add placeholder rows for stoichiometries from the enumeration TSV absent from ``df``.

    Used with ``min_last_convergence.tsv`` so the network plot shows hollow / gray
    ``missing`` nodes for compositions not present in the table. Each placeholder
    ``formula_id`` is taken from ``formula_to_id`` (enumeration TSV).
    """
    if df.empty:
        return df
    present = {
        (int(r["NO3"]), int(r["MeOH"]))
        for _, r in df.iterrows()
    }
    grid_set = set(zn_pairs_from_enumeration(formula_to_id))
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
    for a, c in missing:
        fstr = f"1Zn_{a}NO3_{c}MeOH"
        if fstr not in formula_to_id:
            raise SystemExit(
                f"Enumeration grid placeholder {fstr!r} has no formula_id in {enum_path}"
            )
        row = {col: _placeholder(col) for col in cols}
        row.update(
            {
                "formula_id": int(formula_to_id[fstr]),
                "formula": fstr,
                "NO3": a,
                "MeOH": c,
                "CN_min": a + c,
                "CN_max": 2 * a + c,
                "Charge": 2 - a,
                "n_conf_passed": 0,
            }
        )
        if "Etot_last_cycle_kcal_mol" in row:
            row["Etot_last_cycle_kcal_mol"] = float("nan")
        extra_rows.append(row)
    extra = pd.DataFrame(extra_rows, columns=cols)
    return pd.concat([df, extra], ignore_index=True)


def pairs_reachable_by_ligand_removal(
    stable_pairs: set[Tuple[int, int]],
) -> set[Tuple[int, int]]:
    """All (n_NO3, n_MeOH) tuples reachable from any stable pair by repeatedly
    subtracting 1 from NO3 or MeOH (non-negative). Equivalently: any pair
    that lies component-wise below some stable stoichiometry."""
    reachable: set[Tuple[int, int]] = set(stable_pairs)
    q: deque[Tuple[int, int]] = deque(stable_pairs)
    while q:
        a, c = q.popleft()
        for na, nc in ((a - 1, c), (a, c - 1)):
            if na >= 0 and nc >= 0 and (na, nc) not in reachable:
                reachable.add((na, nc))
                q.append((na, nc))
    return reachable


def closest_stable_derivation(
    target: Tuple[int, int],
    df_active: pd.DataFrame,
) -> Optional[Tuple[int, str, Tuple[int, int], List[str]]]:
    """Stable parent with minimum total ligand removals (-NO3/-MeOH); tie-break: lower formula_id.

    Returns ``(parent_fid, parent_formula, (A,C), steps)`` with ``steps`` like
    ``['-NO3', '-MeOH', ...]`` (order: all -NO3, then -MeOH).
    """
    a, c = target
    best_key: Optional[Tuple[int, int]] = None
    best_fid = 0
    best_formula = ""
    best_ac = (0, 0)
    for _, r in df_active.iterrows():
        A, C = int(r["NO3"]), int(r["MeOH"])
        if A < a or C < c:
            continue
        removals = (A - a) + (C - c)
        fid = int(r["formula_id"])
        key = (removals, fid)
        if best_key is None or key < best_key:
            best_key = key
            best_fid = fid
            best_formula = str(r["formula"])
            best_ac = (A, C)
    if best_key is None:
        return None
    A, C = best_ac
    steps: List[str] = []
    steps.extend(["-NO3"] * (A - a))
    steps.extend(["-MeOH"] * (C - c))
    return (best_fid, best_formula, (A, C), steps)


def print_gray_missing_derivations(
    df_missing: pd.DataFrame,
    df_active: pd.DataFrame,
    derivable_pairs: set[Tuple[int, int]],
    *,
    out_stream,
) -> None:
    """Stdout: for each missing node shown as gray, one chosen stable parent and removal steps."""
    lines: List[
        Tuple[int, str, Tuple[int, int], int, str, Tuple[int, int], List[str]]
    ] = []
    for _, row in df_missing.iterrows():
        t = (int(row["NO3"]), int(row["MeOH"]))
        if t not in derivable_pairs:
            continue
        ex = closest_stable_derivation(t, df_active)
        if ex is None:
            continue
        parent_fid, parent_formula, parent_ac, steps = ex
        lines.append(
            (
                int(row["formula_id"]),
                str(row["formula"]),
                t,
                parent_fid,
                parent_formula,
                parent_ac,
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
        "# Rule: among stable stoichiometries that dominate (NO3,MeOH) component-wise,",
        file=out_stream,
    )
    print(
        "# pick the one with the fewest total ligand removals; tie-break: lower formula_id.",
        file=out_stream,
    )
    print(
        "# Steps list: remove that many NO3 / MeOH from the parent (order: -NO3, then -MeOH).",
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
            f"{mid}\t{mform}\t{mt[0]},{mt[1]}\t"
            f"{pid}\t{pform}\t{pt[0]},{pt[1]}\t{n_steps}\t{joined}",
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
                f"#   missing ID {mid} {mform} {mt}  |  same (NO3,MeOH) as stable ID {pid} {pform}",
                file=out_stream,
            )
    print(f"# Total gray (derivable missing) entries: {len(lines)}", file=out_stream)


def _parse_lowest_e(val: object) -> Optional[float]:
    """Parse `lowest_E` from ana.log-style TSV: energy in Hartree, or 'none'."""
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
    """One row per directed edge: eu, ev, dE in Hartree (from lowest_E)."""
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
    if edge_type == "Add NO3(-)":
        e = ref_kcal.get("NO3")
        return None if e is None else d - e
    if edge_type == "Add MeOH":
        e = ref_kcal.get("MeOH")
        return None if e is None else d - e
    if edge_type == "Exchange (-MeOH, +NO3(-))":
        e_no3 = ref_kcal.get("NO3")
        e_meoh = ref_kcal.get("MeOH")
        if e_no3 is None or e_meoh is None:
            return None
        return d - e_no3 + e_meoh
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
    """Convert eu, ev, dE from Hartree to kcal/mol; no monomer correction."""
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
            "(rows system = MeOH_monomer, NO3_monomer); N/A if a needed monomer is missing.",
            file=out_stream,
        )
        print(
            "#   Add NO3(-):          (E(P)-E(R)) - E(NO3);   Add MeOH: (E(P)-E(R)) - E(MeOH);",
            file=out_stream,
        )
        print(
            "#   Exch. +NO3-:         (E(P)-E(R)) - E(NO3) + E(MeOH).",
            file=out_stream,
        )
        if ref_monomer_kcal:
            print("# Monomer E_tot reference (kcal/mol) from TSV:", file=out_stream)
            for k in ("NO3", "MeOH", "Zn"):
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
            "# dE_cluster_kcal_mol from lowest_E (Hartree) × "
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
            r"(\texttt{MeOH\_monomer}, \texttt{NO3\_monomer}). "
            r"Balanced reactions use $\mathrm{MeOH}$ and $\mathrm{NO_3}^{-}$; "
            r"$q = 2 - n_{\mathrm{NO_3}}$."
        )
    else:
        lines.append(
            r"\textit{Note:} Values are cluster total-energy differences only "
            r"(no monomer correction for \texttt{ana.log}-style input). "
            r"Complex charge $q = 2 - n_{\mathrm{NO_3}}$."
        )
    if from_convergence_tsv and ref_monomer_kcal:
        parts = []
        for k in ("NO3", "MeOH", "Zn"):
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
    """Scatter n_NO3 vs n_MeOH, color = E_tot last cycle (kcal/mol); needs numeric column."""
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
        sub["NO3"],
        sub["MeOH"],
        c=et,
        s=140,
        cmap="viridis",
        edgecolors="#333333",
        linewidths=0.6,
        alpha=0.92,
    )
    plt.colorbar(sc, ax=ax, label=r"$E_{\mathrm{tot}}$ last cycle (kcal mol$^{-1}$)")
    ax.set_xlabel(r"NO$_3^-$ count", fontsize=12, fontweight="bold")
    ax.set_ylabel("MeOH count", fontsize=12, fontweight="bold")
    ax.set_title(
        "ORCA minimized clusters: total energy (last geom. opt. cycle)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(range(int(sub["NO3"].min()), int(sub["NO3"].max()) + 1))
    ax.set_yticks(range(int(sub["MeOH"].min()), int(sub["MeOH"].max()) + 1))
    plt.tight_layout()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


_COL_ZN_O_NO3 = "max_Zn-Omin_NO3(Å)"
_COL_ZN_O_MEOH = "max_Zn-O_MeOH(Å)"


def write_zn_ligand_distance_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter max Zn–O(NO3) vs max Zn–O(MeOH) when both distances are numeric."""
    if _COL_ZN_O_NO3 not in df.columns or _COL_ZN_O_MEOH not in df.columns:
        return
    sub = df[df["formula"].astype(str).str.match(_FORMULA_ZN_RE)].copy()
    rn = sub[_COL_ZN_O_NO3].map(_parse_tsv_float)
    ro = sub[_COL_ZN_O_MEOH].map(_parse_tsv_float)
    sub = sub.assign(_zn_no3=rn, _zn_meoh=ro)
    sub = sub[pd.notna(sub["_zn_no3"]) & pd.notna(sub["_zn_meoh"])]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 7))
    if "Etot_last_cycle_kcal_mol" in sub.columns and sub["Etot_last_cycle_kcal_mol"].notna().any():
        et = sub["Etot_last_cycle_kcal_mol"].astype(float)
        sc = ax.scatter(
            sub["_zn_no3"],
            sub["_zn_meoh"],
            c=et,
            s=120,
            cmap="coolwarm",
            edgecolors="#222222",
            linewidths=0.5,
            alpha=0.9,
        )
        plt.colorbar(sc, ax=ax, label=r"$E_{\mathrm{tot}}$ last cycle (kcal mol$^{-1}$)")
    else:
        ax.scatter(
            sub["_zn_no3"], sub["_zn_meoh"], s=120, c="#4477aa", edgecolors="#222222", alpha=0.9
        )
    ax.set_xlabel(r"max Zn–O (NO$_3$), Å", fontsize=12, fontweight="bold")
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

    pattern = r"^1Zn_(\d+)NO3_(\d+)MeOH$"

    def parse_formula(f: str) -> tuple[int, int]:
        match = re.match(pattern, str(f))
        if match:
            return int(match.group(1)), int(match.group(2))
        return 0, 0

    df[["NO3", "MeOH"]] = df["formula"].apply(
        lambda x: pd.Series(parse_formula(x))
    )
    # NO3 can be monodentate (contributes 1 to CN) or bidentate (contributes 2);
    # the cluster CN ranges between CN_min = n_NO3 + n_MeOH and CN_max = 2*n_NO3 + n_MeOH.
    df["CN_min"] = df["NO3"] + df["MeOH"]
    df["CN_max"] = 2 * df["NO3"] + df["MeOH"]
    df["Charge"] = 2 - df["NO3"]

    if from_convergence_tsv:
        df = augment_dataframe_missing_grid(
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

    df_active = df[df["n_conf_passed"] > 0].copy()
    df_missing = df[df["n_conf_passed"] == 0].copy()

    stable_pairs: set[Tuple[int, int]] = {
        (int(r["NO3"]), int(r["MeOH"]))
        for _, r in df_active.iterrows()
    }
    derivable_pairs = pairs_reachable_by_ligand_removal(stable_pairs)

    if not args.no_print_gray_derivation:
        print_gray_missing_derivations(
            df_missing, df_active, derivable_pairs, out_stream=sys.stdout
        )

    G = nx.DiGraph()

    for _, row in df.iterrows():
        y_pos = row["MeOH"] * 24 + row["Charge"] * 3
        x_pos = row["NO3"]

        status = "active" if row["n_conf_passed"] > 0 else "missing"
        label = (
            f"ID:{row['formula_id']}\n({row['Charge']:+d})" if status == "active" else ""
        )

        if status == "missing":
            t = (int(row["NO3"]), int(row["MeOH"]))
            missing_class = "derivable" if t in derivable_pairs else "isolated"
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

            dA, dC = (
                n2["NO3"] - n1["NO3"],
                n2["MeOH"] - n1["MeOH"],
            )

            edge_type = None
            if (dA, dC) == (0, 1):
                edge_type = "Add MeOH"
            elif (dA, dC) == (1, 0):
                edge_type = "Add NO3(-)"
            elif (dA, dC) == (1, -1):
                edge_type = "Exchange (-MeOH, +NO3(-))"

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

    fig = plt.figure(figsize=(12, 10))
    ax = plt.gca()

    max_meoh = max(6, int(df["MeOH"].max()))
    max_no3 = max(6, int(df["NO3"].max()))
    colors = ["#f4f6f9", "#ffffff"]
    col_colors = ["#eef2f7", "#ffffff"]
    y_bot_band = -28
    x_meoh_label = -1.3

    for j in range(max_no3 + 1):
        ax.axvspan(j - 0.5, j + 0.5, facecolor=col_colors[j % 2], alpha=0.35, zorder=0)
        ax.text(
            j,
            y_bot_band,
            f"{j} NO$_3^-$",
            fontsize=12,
            color="#666666",
            fontweight="bold",
            ha="center",
            va="center",
        )

    for i in range(max_meoh + 1):
        y_center = i * 24
        ax.axhspan(y_center - 11, y_center + 11, facecolor=colors[i % 2], alpha=0.55, zorder=0)
        ax.text(
            x_meoh_label,
            y_center,
            f"{i} MeOH Floor",
            fontsize=12,
            color="#888888",
            fontweight="bold",
            ha="right",
            va="center",
        )

    pos = nx.get_node_attributes(G, "pos")

    edge_colors = {
        "Add MeOH": ("#4a90e2", "dashed", 0.08, 0.6),
        "Add NO3(-)": ("#417505", "dashdot", 0.12, 0.7),
        "Exchange (-MeOH, +NO3(-))": ("#f5a623", "dotted", 0.18, 0.8),
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
    active_node_colors = [charge_palette.get(G.nodes[n]["charge"], "#cccccc") for n in active_nodes]
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
        "Zinc-Nitrate-Methanol Complex Reaction Network\n"
        r"(X = NO$_3^-$ count, rows = MeOH count; q = 2 $-$ n$_{\mathrm{NO_3}}$)",
        fontsize=16,
        pad=20,
        fontweight="bold",
        color="#333333",
    )
    plt.xlabel(
        r"Number of NO$_3^-$ ligands (per Zn$^{2+}$)",
        fontsize=12,
        fontweight="bold",
        color="#555555",
        labelpad=22,
    )

    plt.xticks(
        range(max_no3 + 1),
        [str(i) for i in range(max_no3 + 1)],
        fontsize=11,
        color="#555555",
    )
    plt.yticks([])

    plt.xlim(-3.4, max_no3 + 1.0)
    ax.set_ylim(bottom=y_bot_band - 8)

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
