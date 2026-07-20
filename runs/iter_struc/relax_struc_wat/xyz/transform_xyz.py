#!/usr/bin/env python3
"""Build water-side XYZ files by demethylating relax_struc2 qm_minimize min.xyz."""

from __future__ import annotations

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

import argparse
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ASSIGN_NAME_SCRIPT = TOOLS_DIR / "zif_meoh_assign_name.py"
_C_H_BOND_A = 1.09
_O_H_BOND_A = 0.96
_HOH_ANGLE_RAD = np.radians(104.5)

_RELAX2_RUN41 = REPO_ROOT / "runs" / "iter_struc" / "relax_struc2" / "qm_minimize" / "run41"
_EXTRA_RUN41 = REPO_ROOT / "runs" / "iter_struc" / "extra_test" / "qm_minimize" / "run41"
_COMPLEX_RE = re.compile(r"^1Zn_(\d+)MIm_(\d+)MImH_(\d+)MeOH$")


@dataclass(frozen=True)
class SourceComplex:
    path: Path
    n_mim: int
    n_mih: int
    n_meoh: int


def _load_assign_name_module():
    spec = importlib.util.spec_from_file_location("zif_meoh_assign_name", ASSIGN_NAME_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {ASSIGN_NAME_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replacement_h_position(c3_xyz, c4_xyz, n1_xyz=None, n2_xyz=None):
    c3_xyz = np.asarray(c3_xyz, dtype=float)
    if n1_xyz is not None and n2_xyz is not None:
        vec = c3_xyz - (np.asarray(n1_xyz, dtype=float) + np.asarray(n2_xyz, dtype=float)) / 2.0
    else:
        vec = c3_xyz - np.asarray(c4_xyz, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-8:
        return c3_xyz + np.array([_C_H_BOND_A, 0.0, 0.0])
    return c3_xyz + _C_H_BOND_A * vec / norm


def _replacement_second_water_h(o_xyz, h1_xyz, c_xyz):
    o_xyz = np.asarray(o_xyz, dtype=float)
    h1_xyz = np.asarray(h1_xyz, dtype=float)
    c_xyz = np.asarray(c_xyz, dtype=float)

    oh = h1_xyz - o_xyz
    oh_norm = float(np.linalg.norm(oh))
    if oh_norm <= 1.0e-8:
        oh = np.array([0.0, 0.0, 1.0])
    else:
        oh = oh / oh_norm

    oc = c_xyz - o_xyz
    oc_norm = float(np.linalg.norm(oc))
    if oc_norm > 1.0e-8:
        oc = oc / oc_norm
    else:
        oc = np.array([1.0, 0.0, 0.0])

    axis = np.cross(oh, oc)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-8:
        axis = np.cross(oh, np.array([1.0, 0.0, 0.0]))
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-8:
            axis = np.cross(oh, np.array([0.0, 1.0, 0.0]))
            axis_norm = float(np.linalg.norm(axis))
    axis = axis / axis_norm

    cos_a = np.cos(_HOH_ANGLE_RAD)
    sin_a = np.sin(_HOH_ANGLE_RAD)
    oh_rot = oh * cos_a + np.cross(axis, oh) * sin_a + axis * np.dot(axis, oh) * (1.0 - cos_a)
    oh_rot_norm = float(np.linalg.norm(oh_rot))
    if oh_rot_norm <= 1.0e-8:
        oh_rot = np.array([0.0, 1.0, 0.0])
    else:
        oh_rot = oh_rot / oh_rot_norm
    return o_xyz + _O_H_BOND_A * oh_rot


def _atoms_by_name(residue):
    return {atom.name: atom for atom in residue.atoms}


def _demethylate_imidazole_residue(residue):
    by_name = _atoms_by_name(residue)
    if "C3" not in by_name or "C4" not in by_name:
        raise ValueError(f"{residue.resname} residue is missing C3/C4 atom names")

    new_resname = "IM-" if residue.resname == "MIM" else "IMH"
    h_xyz = _replacement_h_position(
        by_name["C3"].position,
        by_name["C4"].position,
        by_name["N1"].position if "N1" in by_name else None,
        by_name["N2"].position if "N2" in by_name else None,
    )
    methyl_names = {"C4", "H1", "H2", "H3"}
    out = []

    for atom in residue.atoms:
        if atom.name in methyl_names:
            continue
        element = atom.element
        name = atom.name
        if residue.resname == "MIH" and atom.name == "H6":
            name = "HN1"
        out.append((element, atom.position.copy(), name, new_resname))
        if atom.name == "C3":
            out.append(("H", h_xyz.copy(), "H6", new_resname))
    return out


def _demethylate_methanol_residue(residue):
    by_name = _atoms_by_name(residue)
    required = ("C1", "O1", "HO1")
    for name in required:
        if name not in by_name:
            raise ValueError(f"MOH residue is missing {name}")

    h2_xyz = _replacement_second_water_h(
        by_name["O1"].position,
        by_name["HO1"].position,
        by_name["C1"].position,
    )
    return [
        ("O", by_name["O1"].position.copy(), "O", "WAT"),
        ("H", by_name["HO1"].position.copy(), "H1", "WAT"),
        ("H", h2_xyz.copy(), "H2", "WAT"),
    ]


def _keep_residue_atoms(residue):
    return [(atom.element, atom.position.copy(), atom.name, residue.resname) for atom in residue.atoms]


def _transform_universe(universe, demethyl_ligand: bool, demethyl_solvent: bool):
    transformed = []
    for residue in universe.residues:
        resname = residue.resname
        if demethyl_ligand and resname in {"MIM", "MIH"}:
            transformed.extend(_demethylate_imidazole_residue(residue))
        elif demethyl_solvent and resname == "MOH":
            transformed.extend(_demethylate_methanol_residue(residue))
        else:
            transformed.extend(_keep_residue_atoms(residue))
    return transformed


def _output_name(source: SourceComplex) -> str:
    # Parallel to source 1Zn_{n}MIm_{m}MImH_{k}MeOH with demethylated residue names.
    return f"1Zn_{source.n_mim}Im-_{source.n_mih}ImH_{source.n_meoh}Wat"


def _write_xyz(path: Path, atoms: Iterable[tuple], comment: str) -> None:
    atom_list = list(atoms)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(atom_list)}\n")
        handle.write(f"{comment}\n")
        for element, position, _name, _resname in atom_list:
            handle.write(
                f"{element:<2s} {position[0]:16.8f} {position[1]:16.8f} {position[2]:16.8f}\n"
            )


def discover_sources(run_dir: Path = _RELAX2_RUN41) -> list[SourceComplex]:
    sources: list[SourceComplex] = []
    for path in sorted(run_dir.glob("1Zn_*/min.xyz")):
        match = _COMPLEX_RE.match(path.parent.name)
        if match is None:
            raise ValueError(f"Unrecognized complex directory name: {path.parent.name}")
        sources.append(
            SourceComplex(
                path=path,
                n_mim=int(match.group(1)),
                n_mih=int(match.group(2)),
                n_meoh=int(match.group(3)),
            )
        )
    if not sources:
        raise FileNotFoundError(f"No 1Zn_*/min.xyz found under {run_dir}")
    return sources


def _transform_source(source: SourceComplex, out_dir: Path, log_dir: Path) -> Path:
    assign = _load_assign_name_module()
    if not source.path.is_file():
        raise FileNotFoundError(source.path)

    universe = assign.xyz_to_mda(
        str(source.path),
        expand_nh_oh_radius=True,
        delete_wrong_bonds=True,
    )
    name = _output_name(source)
    out_path = out_dir / f"{name}.xyz"
    atoms = _transform_universe(universe, demethyl_ligand=True, demethyl_solvent=True)
    comment = (
        f"{name} from {source.path}; "
        "demethyl_ligand=True; demethyl_solvent=True"
    )
    _write_xyz(out_path, atoms, comment)
    log_path = log_dir / f"{name}.log"
    log_path.write_text(
        "\n".join(
            [
                f"source={source.path}",
                f"output={out_path}",
                "demethyl_ligand=True",
                "demethyl_solvent=True",
                f"n_atoms={len(atoms)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return out_path


def _copy_monomer(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.write_bytes(src.read_bytes())


def _demethylate_monomer_xyz(src: Path, out_path: Path, demethyl_ligand: bool, demethyl_solvent: bool) -> None:
    assign = _load_assign_name_module()
    universe = assign.xyz_to_mda(
        str(src),
        expand_nh_oh_radius=True,
        delete_wrong_bonds=True,
    )
    atoms = _transform_universe(universe, demethyl_ligand, demethyl_solvent)
    _write_xyz(
        out_path,
        atoms,
        f"{out_path.stem} from {src}; demethyl_ligand={demethyl_ligand}; demethyl_solvent={demethyl_solvent}",
    )


def copy_monomers(out_dir: Path) -> list[Path]:
    written: list[Path] = []

    # Prefer already-minimized demethylated monomers when available.
    preferred = {
        "Im-_monomer.xyz": (
            _RELAX2_RUN41 / "manual" / "Im-_monomer" / "min.xyz",
            _EXTRA_RUN41 / "Im-_monomer" / "min.xyz",
        ),
        "ImH_monomer.xyz": (
            _RELAX2_RUN41 / "manual" / "ImH_monomer" / "min.xyz",
            _EXTRA_RUN41 / "ImH_monomer" / "min.xyz",
        ),
        "Wat_monomer.xyz": (_EXTRA_RUN41 / "Wat_monomer" / "min.xyz",),
        "Zn_monomer.xyz": (
            _RELAX2_RUN41 / "Zn_monomer" / "min.xyz",
            _EXTRA_RUN41 / "Zn_monomer" / "min.xyz",
        ),
    }
    for name, candidates in preferred.items():
        dst = out_dir / name
        for src in candidates:
            if src.is_file():
                _copy_monomer(src, dst)
                written.append(dst)
                break
        else:
            # Fallback: demethylate the methylated monomer from relax_struc2.
            if name == "Im-_monomer.xyz":
                _demethylate_monomer_xyz(
                    _RELAX2_RUN41 / "MIm_monomer" / "min.xyz",
                    dst,
                    demethyl_ligand=True,
                    demethyl_solvent=False,
                )
            elif name == "ImH_monomer.xyz":
                _demethylate_monomer_xyz(
                    _RELAX2_RUN41 / "MImH_monomer" / "min.xyz",
                    dst,
                    demethyl_ligand=True,
                    demethyl_solvent=False,
                )
            elif name == "Wat_monomer.xyz":
                _demethylate_monomer_xyz(
                    _RELAX2_RUN41 / "MeOH_monomer" / "min.xyz",
                    dst,
                    demethyl_ligand=False,
                    demethyl_solvent=True,
                )
            else:
                raise FileNotFoundError(f"No source found for {name}")
            written.append(dst)

    h_monomer = out_dir / "H_monomer.xyz"
    h_src = _RELAX2_RUN41 / "H_monomer" / "min.xyz"
    if h_src.is_file():
        _copy_monomer(h_src, h_monomer)
    else:
        h_monomer.write_text(
            "1\nH+\nH     0.000000      0.000000      0.000000\n\n",
            encoding="utf-8",
        )
    written.append(h_monomer)
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=_RELAX2_RUN41,
        help="relax_struc2 qm_minimize run directory containing 1Zn_*/min.xyz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("xyz_files"),
        help="Directory for generated XYZ files.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("transform_logs"),
        help="Directory for transformation logs.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    written = copy_monomers(args.out_dir)
    for source in discover_sources(args.source_dir):
        written.append(_transform_source(source, args.out_dir, args.log_dir))

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
