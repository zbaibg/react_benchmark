#!/usr/bin/env python3
"""Copy selected solv_lig_test qm_minimize min.xyz and write solute/solvent strips."""

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
import shutil
from pathlib import Path
from typing import Iterable

ASSIGN_NAME_SCRIPT = TOOLS_DIR / "zif_meoh_assign_name.py"

# Source: solv_lig_test/qm_minimize/<run>/*/min.xyz
_SOLV_LIG_QM = (
    REPO_ROOT / "runs" / "iter_struc" / "solv_lig_test" / "qm_minimize"
)

# Keep only ImH / Im- / Wat / MeOH hex and pent complexes (+ monomers).
KEEP_COMPLEXES = (
    "1Zn_0ImH_6Wat",  # Zn6Wat
    "1Zn_0MIm_0MImH_6MeOH",  # Zn6MeOH
    "1Zn_0Im-_1ImH_5MeOH",  # ZnImH5MeOH
    "1Zn_1Im-_0ImH_5MeOH",  # ZnIm-5MeOH
    "1Zn_1ImH_5Wat",  # ZnImH5Wat
    "1Zn_1Im-_5Wat",  # ZnIm-5Wat
)

KEEP_MONOMERS = (
    "ImH_monomer",
    "Im-_monomer",
    "Wat_monomer",
    "MeOH_monomer",
    "Zn_monomer",
)

SOLVENT_RESNAMES = frozenset({"WAT", "MOH"})
SOLUTE_RESNAMES = frozenset({"ZN", "IMH", "IM-", "MIM", "MIH", "H"})


def _load_assign_name_module():
    spec = importlib.util.spec_from_file_location("zif_meoh_assign_name", ASSIGN_NAME_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {ASSIGN_NAME_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_min_xyz(structure: str) -> Path:
    """Locate structure/min.xyz under solv_lig_test/qm_minimize/run*."""
    matches = sorted(_SOLV_LIG_QM.glob(f"*/{structure}/min.xyz"))
    if matches:
        # Prefer highest run number if several exist.
        return matches[-1]
    # Fallback: solv_lig_test/xyz/xyz_files/<structure>.xyz (e.g. Zn_monomer).
    fallback = (
        REPO_ROOT
        / "runs"
        / "iter_struc"
        / "solv_lig_test"
        / "xyz"
        / "xyz_files"
        / f"{structure}.xyz"
    )
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"No min.xyz for {structure} under {_SOLV_LIG_QM}/run*/{structure}/ "
        f"and no fallback at {fallback}"
    )


def _write_xyz(path: Path, atoms: Iterable[tuple], comment: str) -> None:
    atom_list = list(atoms)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(atom_list)}\n")
        handle.write(f"{comment}\n")
        for element, position in atom_list:
            handle.write(
                f"{element:<2s} {position[0]:16.8f} {position[1]:16.8f} {position[2]:16.8f}\n"
            )


def _residue_atoms(residue) -> list[tuple]:
    out = []
    for atom in residue.atoms:
        out.append((atom.element, atom.position.copy()))
    return out


def _strip_complex(
    src_xyz: Path,
    out_dir: Path,
    base_name: str,
    log_dir: Path,
) -> list[Path]:
    """Write full copy plus _solute (no solvent) and _solvent (no solute)."""
    assign = _load_assign_name_module()
    universe = assign.xyz_to_mda(
        str(src_xyz),
        expand_nh_oh_radius=True,
        delete_wrong_bonds=True,
    )

    full_atoms: list[tuple] = []
    solute_atoms: list[tuple] = []
    solvent_atoms: list[tuple] = []
    unknown: list[str] = []

    for residue in universe.residues:
        resname = str(residue.resname).strip()
        atoms = _residue_atoms(residue)
        full_atoms.extend(atoms)
        if resname in SOLVENT_RESNAMES:
            solvent_atoms.extend(atoms)
        elif resname in SOLUTE_RESNAMES:
            solute_atoms.extend(atoms)
        else:
            unknown.append(resname)

    if unknown:
        raise ValueError(
            f"{src_xyz}: unrecognized residue names for strip: {sorted(set(unknown))}"
        )
    if not solute_atoms:
        raise ValueError(f"{src_xyz}: empty solute after strip")
    if not solvent_atoms:
        raise ValueError(f"{src_xyz}: empty solvent after strip")

    written: list[Path] = []
    specs = (
        (base_name, full_atoms, "full complex"),
        (f"{base_name}_solute", solute_atoms, "solvent stripped"),
        (f"{base_name}_solvent", solvent_atoms, "solute stripped"),
    )
    for name, atoms, kind in specs:
        out_path = out_dir / f"{name}.xyz"
        comment = f"{name} from {src_xyz}; {kind}"
        _write_xyz(out_path, atoms, comment)
        log_path = log_dir / f"{name}.log"
        log_path.write_text(
            "\n".join(
                [
                    f"source={src_xyz}",
                    f"output={out_path}",
                    f"kind={kind}",
                    f"n_atoms={len(atoms)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(out_path)
    return written


def copy_monomers(out_dir: Path, log_dir: Path) -> list[Path]:
    written: list[Path] = []
    for name in KEEP_MONOMERS:
        src = _find_min_xyz(name)
        dst = out_dir / f"{name}.xyz"
        shutil.copy2(src, dst)
        (log_dir / f"{name}.log").write_text(
            f"source={src}\noutput={dst}\nkind=monomer copy\n",
            encoding="utf-8",
        )
        written.append(dst)
    return written


def copy_complexes(out_dir: Path, log_dir: Path) -> list[Path]:
    written: list[Path] = []
    for name in KEEP_COMPLEXES:
        src = _find_min_xyz(name)
        written.extend(_strip_complex(src, out_dir, name, log_dir))
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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

    # Clear previous xyz so removed complexes do not linger.
    for old in args.out_dir.glob("*.xyz"):
        old.unlink()

    written = copy_monomers(args.out_dir, args.log_dir)
    written.extend(copy_complexes(args.out_dir, args.log_dir))

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
