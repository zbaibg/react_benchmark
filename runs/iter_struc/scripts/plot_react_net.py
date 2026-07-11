
# --- repo path bootstrap (auto) ---
from pathlib import Path as _Path
import sys as _sys
_REPO_CAND = _Path(__file__).resolve().parent
while _REPO_CAND != _REPO_CAND.parent and not (_REPO_CAND / "software.yaml").exists():
    _REPO_CAND = _REPO_CAND.parent
if not (_REPO_CAND / "software.yaml").exists():
    raise RuntimeError("Could not locate repo root (software.yaml)")
REPO_ROOT = _REPO_CAND
TOOLS_DIR = REPO_ROOT / "tools"
_sys.path.insert(0, str(TOOLS_DIR))
try:
    from paths import load_software as _load_software
    _SW = _load_software()
except Exception:
    _SW = {}
# --- end bootstrap ---
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

_ITER_STRUC = Path(__file__).resolve().parent.parent

# CODATA: 1 Hartree = E_h / (kcal/mol per Hartree). Used to convert ana.log lowest_E (Hartree).
HARTREE_TO_KCAL_MOL = 627.5094738897745

# Default DFTB+ SP monomer references for Total energy in detailed.out.
_DEFAULT_REF_RUN6_DIR = Path(
    str(REPO_ROOT / "runs" / "deprotonate/")
    "MIMH_PBE_STRUC/SP_init/run6"
)
# MeOH monomer SP lives under lig_exchange run6 (not deprotonate run6).
_DEFAULT_REF_MEOH_RUN6_DIR = Path(
    str(REPO_ROOT / "runs" / "lig_exchange/")
    "MIMH_PBE_STRUC/SP_init/run6"
)

_TOTAL_ENERGY_H_RE = re.compile(
    r"^\s*Total energy:\s+([-+0-9.Ee]+)\s+H",
    re.IGNORECASE | re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot zinc-imidazole-methanol reaction network from "
        "check_metallogen_short_bonds ana.log (formula_id table) or a TSV with the same columns."
    )
    p.add_argument(
        "data_file",
        type=Path,
        nargs="?",
        default=Path("./ana.log"),
        help="Path to input file (default: ./ana.log), e.g. gen_struc_dftbplus/ana.log "
        "(contains the formula_id table).",
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
        "--ref-run6-dir",
        type=Path,
        default=None,
        help="Directory containing MImH_monomer, MIm_monomer, Wat_monomer/detailed.out "
        "(parse Total energy Hartree). If omitted, uses the default run6 path when it exists.",
    )
    p.add_argument(
        "--no-ref-run6",
        action="store_true",
        help="Do not load monomer reference energies from run6 detailed.out.",
    )
    p.add_argument(
        "--ref-meoh-run6-dir",
        type=Path,
        default=None,
        help="Directory with MeOH_monomer/detailed.out (lig_exchange SP run6). "
        "If omitted, uses the default lig_exchange run6 path when it exists.",
    )
    return p.parse_args()


def _is_dash_separator(line: str) -> bool:
    s = line.strip()
    return bool(s) and all(c == "-" for c in s)


def extract_formula_table_text(path: Path) -> str:
    """Extract the formula_id ... passed_lowE_list_by_id block from ana.log or a bare TSV."""
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
    # Header must be included so pandas gets column names (formula_id, formula, ...).
    return lines[start] + "\n" + "\n".join(out_lines)


def read_formula_dataframe(data_str: str) -> pd.DataFrame:
    header_line = data_str.splitlines()[0]
    sep = "\t" if header_line.count("\t") >= 4 else r"\s+"
    df = pd.read_csv(io.StringIO(data_str), sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _latex_escape(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("_", r"\_")
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    s = s.replace("#", r"\#")
    return s


_FORMULA_ZN_RE = re.compile(
    r"^1Zn_(\d+)MIm_(\d+)MImH_(\d+)MeOH$",
)


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


def parse_dftb_total_energy_hartree(detailed_out: Path) -> float:
    """Read DFTB+ `Total energy: ... H` from detailed.out (final SCC total)."""
    text = detailed_out.read_text(encoding="utf-8", errors="replace")
    m = _TOTAL_ENERGY_H_RE.search(text)
    if not m:
        raise ValueError(f"No 'Total energy: ... H' line in {detailed_out}")
    return float(m.group(1))


def load_run6_monomer_energies_hartree(
    run6_dir: Path,
    *,
    meoh_run6_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Monomer Total energy (Hartree): MImH, MIm, Wat, H from run6_dir; MeOH optional."""
    mapping = {
        "MImH": run6_dir / "MImH_monomer" / "detailed.out",
        "MIm": run6_dir / "MIm_monomer" / "detailed.out",
        "Wat": run6_dir / "Wat_monomer" / "detailed.out",
        "H": run6_dir / "H_monomer" / "detailed.out",
    }
    out: Dict[str, float] = {}
    for k, p in mapping.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing reference detailed.out: {p}")
        out[k] = parse_dftb_total_energy_hartree(p)

    meoh_root: Optional[Path] = None
    if meoh_run6_dir is not None:
        r = meoh_run6_dir.resolve()
        if r.is_dir():
            meoh_root = r
    elif _DEFAULT_REF_MEOH_RUN6_DIR.is_dir():
        meoh_root = _DEFAULT_REF_MEOH_RUN6_DIR.resolve()
    if meoh_root is not None:
        meoh_p = meoh_root / "MeOH_monomer" / "detailed.out"
        if meoh_p.is_file():
            out["MeOH"] = parse_dftb_total_energy_hartree(meoh_p)
    return out


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
    """Stable parent with minimum total ligand removals (−MIm/−MImH/−MeOH); tie-break: lower formula_id.

    Returns ``(parent_fid, parent_formula, (A,B,C), steps)`` with ``steps`` like
    ``['-MIm', '-MeOH', ...]`` (order: all −MIm, then −MImH, then −MeOH).
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
        "# Steps list: remove that many MIm / MImH / MeOH from the parent (order: −MIm, then −MImH, then −MeOH).",
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


def compute_de_rxn_ha(
    edge_type: str,
    eu: float,
    ev: float,
    ref_ha: Dict[str, float],
) -> Optional[float]:
    """Gas-phase stoichiometric ΔE (Hartree, 0 K) using run6 monomers where possible.

    ``dE_cluster = ev - eu`` is E(product) - E(reactant) for whole-cluster totals from
    ana.log. For edges that add / remove a monomer with a reference energy, the
    balanced reaction uses:

      Add MIm(-):          R + MIm- -> P
        ΔE_rxn = (ev - eu) - E(MIm_ref)

      Add MImH:            R + MImH -> P
        ΔE_rxn = (ev - eu) - E(MImH_ref)

      Deprotonation (-H+): R -> P + H+
        Use a hydrogen monomer reference H (same DFTB setup) as an approximate
        proton reference:
        ΔE_rxn ≈ (ev - eu) + E(H_ref)

      Add MeOH: R + MeOH -> P
        ΔE_rxn = (ev - eu) - E(MeOH_ref)

      Exchange (-MeOH, +MIm(-)): R + MIm- -> P + MeOH
        ΔE_rxn = (ev - eu) - E(MIm) + E(MeOH)

      Exchange (-MeOH, +MImH): R + MImH -> P + MeOH
        ΔE_rxn = (ev - eu) - E(MImH) + E(MeOH)
    """
    d = ev - eu
    if edge_type == "Add MIm(-)":
        e = ref_ha.get("MIm")
        return None if e is None else d - e
    if edge_type == "Add MImH":
        e = ref_ha.get("MImH")
        return None if e is None else d - e
    if edge_type == "Deprotonation (-H+)":
        e = ref_ha.get("H")
        return None if e is None else d + e
    if edge_type == "Add MeOH":
        e = ref_ha.get("MeOH")
        return None if e is None else d - e
    if edge_type == "Exchange (-MeOH, +MIm(-))":
        e_mim = ref_ha.get("MIm")
        e_meoh = ref_ha.get("MeOH")
        if e_mim is None or e_meoh is None:
            return None
        return d - e_mim + e_meoh
    if edge_type == "Exchange (-MeOH, +MImH)":
        e_mimh = ref_ha.get("MImH")
        e_meoh = ref_ha.get("MeOH")
        if e_mimh is None or e_meoh is None:
            return None
        return d - e_mimh + e_meoh
    return None


def hartree_rows_to_kcal_mol_with_rxn(
    rows_ha: List[
        Tuple[str, int, int, str, str, Optional[float], Optional[float], Optional[float]]
    ],
    ref_ha: Optional[Dict[str, float]],
) -> List[ReactionDERow]:
    """Convert eu, ev, dE to kcal/mol; optional dE_rxn when run6 refs apply."""
    c = HARTREE_TO_KCAL_MOL
    out: List[ReactionDERow] = []
    for et, iu, iv, fu, fv, eu, ev, de in rows_ha:
        eu_k = eu * c if eu is not None else None
        ev_k = ev * c if ev is not None else None
        de_k = de * c if de is not None else None
        de_rxn_k: Optional[float] = None
        if (
            ref_ha is not None
            and eu is not None
            and ev is not None
            and de is not None
        ):
            de_rxn_ha = compute_de_rxn_ha(et, eu, ev, ref_ha)
            if de_rxn_ha is not None:
                de_rxn_k = de_rxn_ha * c
        out.append((et, iu, iv, fu, fv, eu_k, ev_k, de_k, de_rxn_k))
    return out


def print_reaction_dE_table(
    rows: List[ReactionDERow],
    *,
    ref_ha: Optional[Dict[str, float]],
    out_stream,
) -> None:
    """Print TSV: reaction type, ids, formulas, balanced equation, dE (kcal/mol)."""
    print("", file=out_stream)
    print(
        "# dE_rxn is an approximate stoichiometric ΔE in kcal/mol for balanced_reaction,",
        file=out_stream,
    )
    print(
        "# built from cluster lowest_E (Hartree) in ana.log plus monomer references:",
        file=out_stream,
    )
    print(
        "#   Add MIm(-):      (E(P) - E(R)) - E(MIm)",
        file=out_stream,
    )
    print(
        "#   Add MImH:        (E(P) - E(R)) - E(MImH)",
        file=out_stream,
    )
    print(
        "#   Deprotonation:   (E(P) - E(R)) + E(H)",
        file=out_stream,
    )
    print(
        "#   Add MeOH:        (E(P) - E(R)) - E(MeOH)",
        file=out_stream,
    )
    print(
        "#   Exch. +MIm-:     (E(P) - E(R)) - E(MIm) + E(MeOH)",
        file=out_stream,
    )
    print(
        "#   Exch. +MImH:     (E(P) - E(R)) - E(MImH) + E(MeOH)",
        file=out_stream,
    )
    if ref_ha:
        print(
            "# Monomer reference Total energy (Hartree) from run6 detailed.out:",
            file=out_stream,
        )
        for k in ("MImH", "MIm", "Wat", "H", "MeOH"):
            if k not in ref_ha:
                continue
            e = ref_ha[k]
            ek = e * HARTREE_TO_KCAL_MOL
            print(f"#   {k}: E_Ha = {e:.12f}  E_kcal_mol = {ek:.8f}", file=out_stream)
    print(
        "# reaction_type\tfrom_fid\tto_fid\tfrom_formula\tto_formula\t"
        "balanced_reaction\tdE_rxn_kcal_mol",
        file=out_stream,
    )
    for et, iu, iv, fu, fv, eu, ev, de, de_rxn in rows:
        if de_rxn is not None:
            dr_s = f"{de_rxn:.12f}"
        else:
            dr_s = "N/A"
        bal = balanced_reaction_plain(et, fu, fv)
        print(
            f"{et}\t{iu}\t{iv}\t{fu}\t{fv}\t{bal}\t{dr_s}",
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
    ref_ha: Optional[Dict[str, float]] = None,
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
    # Avoid f-string brace clash on `mol$^{-1}$` (use raw fragment for superscript).
    lines.append(
        r"\captionof{table}{$\Delta E_{\mathrm{rxn}}$ (\texttt{" + et_e + r"}) in kcal\,mol$^{-1}$; "
        + rf"built from cluster lowest\_E (Hartree $\times {HARTREE_TO_KCAL_MOL:.6f}$) "
        + r"and monomer references.}"
    )
    lines.append(r"\end{center}")
    lines.append(r"\vspace{1mm}")
    lines.append(
        r"\begin{longtable}{r p{0.12\textwidth} p{0.58\textwidth} r}"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\# & ID $\rightarrow$ ID & balanced reaction & "
        r"$\Delta E_{\mathrm{rxn}}$ \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(
        r"\# & ID $\rightarrow$ ID & balanced reaction & "
        r"$\Delta E_{\mathrm{rxn}}$ \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")

    for i, (_et, iu, iv, fu, fv, eu, ev, de, de_rxn) in enumerate(rows, start=1):
        idcol = rf"$\mathrm{{ID}}\,{iu} \rightarrow \mathrm{{ID}}\,{iv}$"
        bal = balanced_reaction_latex(_et, fu, fv)
        formcol = rf"$\displaystyle {bal}$"
        if de_rxn is not None:
            ddr = f"{de_rxn:+.6f}"
        else:
            ddr = "---"
        lines.append(f"{i:d} & {idcol} & {formcol} & {ddr} \\\\")

    lines.append(r"\end{longtable}")
    lines.append(
        r"\textit{Note:} $\Delta E_{\mathrm{rxn}}$ is an approximate stoichiometric "
        r"$\Delta E$ for the balanced reaction, constructed from cluster "
        r"$\texttt{lowest\_E}$ (Hartree) and run6 monomer references. "
        r"Balanced reactions use explicit $\mathrm{MeOH}$, $\mathrm{MImH}$, "
        r"$\mathrm{MIm}^{-}$, $\mathrm{H}^{+}$ as needed; complex charge "
        r"$q = 2 - n_{\mathrm{MeIm}}$."
    )
    if ref_ha:
        parts = []
        for k in ("MImH", "MIm", "Wat", "H", "MeOH"):
            if k not in ref_ha:
                continue
            e = ref_ha[k]
            ek = e * HARTREE_TO_KCAL_MOL
            parts.append(
                rf"$E(\mathrm{{{k}}}) = {e:.8f}$ H $= {ek:.4f}$ kcal\,mol$^{{-1}}$"
            )
        lines.append(
            r"\textit{References (DFTB+ SP, run6 monomer):} " + r";\, ".join(parts) + "."
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
    ref_ha: Optional[Dict[str, float]] = None,
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

    base = out_png if out_png.is_absolute() else Path.cwd() / out_png
    stem = base.stem
    parent = base.parent
    written: List[Path] = []

    for reaction_type in sorted(by_type.keys()):
        subrows = by_type[reaction_type]
        slug = _sanitize_type_for_filename(reaction_type)
        out_abs = parent / f"{stem}_{slug}.png"
        tex_body = _latex_document_reaction_dE_for_type(
            subrows, reaction_type, ref_ha=ref_ha
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


def main() -> None:
    args = parse_args()
    data_path = args.data_file
    if not data_path.is_file():
        alt = _ITER_STRUC / data_path
        if alt.is_file():
            data_path = alt
        else:
            raise SystemExit(f"Input file not found: {args.data_file}")

    data_str = extract_formula_table_text(data_path)
    df = read_formula_dataframe(data_str)

    pattern = r"1Zn_(\d+)MIm_(\d+)MImH_(\d+)MeOH"

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

        G.add_node(
            row["formula_id"],
            pos=(x_pos, y_pos),
            label=label,
            charge=row["Charge"],
            status=status,
            missing_class=missing_class,
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
    if "lowest_E" in df.columns:
        fid_to_e = {
            int(r["formula_id"]): _parse_lowest_e(r["lowest_E"]) for _, r in df.iterrows()
        }
    else:
        fid_to_e = {int(r["formula_id"]): None for _, r in df.iterrows()}

    ref_ha: Optional[Dict[str, float]] = None
    if not args.no_ref_run6:
        rdir = args.ref_run6_dir
        if rdir is None and _DEFAULT_REF_RUN6_DIR.is_dir():
            rdir = _DEFAULT_REF_RUN6_DIR
        if rdir is not None:
            try:
                meoh_dir = args.ref_meoh_run6_dir
                if meoh_dir is not None:
                    meoh_dir = meoh_dir.resolve()
                ref_ha = load_run6_monomer_energies_hartree(
                    rdir.resolve(),
                    meoh_run6_dir=meoh_dir,
                )
            except OSError as e:
                print(
                    f"WARNING: could not load run6 monomer references from {rdir} ({e})",
                    file=sys.stderr,
                )

    rows_ha = collect_reaction_dE_rows(G, fid_to_e, fid_to_formula)
    dE_rows = hartree_rows_to_kcal_mol_with_rxn(rows_ha, ref_ha)

    if not args.no_print_reactions:
        print_reaction_dE_table(dE_rows, ref_ha=ref_ha, out_stream=sys.stdout)

    if not args.no_de_table_png:
        de_out = args.de_table_png
        if not de_out.is_absolute():
            de_out = Path.cwd() / de_out
        try:
            paths = write_reaction_dE_latex_pngs_by_type(
                dE_rows, de_out, dpi=args.latex_dpi, ref_ha=ref_ha
            )
            for p in paths:
                print(f"Wrote LaTeX ΔE table PNG to {p}")
        except Exception as e:
            print(f"WARNING: could not build LaTeX ΔE PNG ({e})", file=sys.stderr)

    fig = plt.figure(figsize=(14, 10))
    ax = plt.gca()

    max_meoh = df["MeOH"].max()
    colors = ["#f4f6f9", "#ffffff"]
    for i in range(int(max_meoh) + 1):
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
            edgecolors="#bbbbbb",
            linewidths=1.5,
        )
        isolated_plot.set_linestyle("dashed")

    # Missing but (MIm,MImH,MeOH) obtainable by removing ligands from some stable structure
    if missing_derivable:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=missing_derivable,
            node_size=400,
            node_color="#c8c8c8",
            edgecolors="#6a6a6a",
            linewidths=1.2,
        )

    active_node_colors = [charge_palette[G.nodes[n]["charge"]] for n in active_nodes]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=active_nodes,
        node_size=600,
        node_color=active_node_colors,
        edgecolors="#333333",
        linewidths=1.2,
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

    max_im = int(df["im_total"].max())
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
        f"ΔE (stdout / LaTeX tables): kcal/mol from Hartree × {HARTREE_TO_KCAL_MOL:.6f}",
        ha="center",
        fontsize=9,
        color="#666666",
    )
    out = args.output
    if not out.is_absolute():
        out = Path.cwd() / out
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
