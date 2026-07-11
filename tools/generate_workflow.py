#!/usr/bin/env python3
"""
Unified script to generate run directories for minimize, SP_init, and SP_opt workflows.
Usage:
    python generate_workflow.py --mode minimize
    python generate_workflow.py --mode sp_init
    python generate_workflow.py --mode sp_opt
    python generate_workflow.py --mode qm_minimize
"""
from math import e
import os
import re
import sys
import stat
import shutil
import argparse
import warnings
from pathlib import Path

import yaml
import MDAnalysis as mda

# Suppress MDAnalysis "No coordinate reader found for ... prmtop" (topology-only files are expected)
warnings.filterwarnings("ignore", message="No coordinate reader found for")

_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))

from paths import (  # noqa: E402
    REPO_ROOT,
    load_software,
    resolve_dftb_skroot,
    resolve_project_path,
)
from zif_meoh_assign_name import xyz_to_mda  # noqa: E402
from make_mrcc_genbas import (  # noqa: E402
    generate_genbas as make_mrcc_generate_genbas,
    parse_elements_arg as parse_mrcc_elements_arg,
)

_SOFTWARE = load_software()
_ORCA_PATH = str(_SOFTWARE["orca"])
_MOLPRO_ROOT = str(_SOFTWARE["molpro_root"])
_MRCC_PATH = str(_SOFTWARE["mrcc"])
_CONDAINIT = str(_SOFTWARE["condainit"])
_SCRATCH_ROOT = str(_SOFTWARE["scratch_root"])
_GXTB_HOME_DEFAULT = str((_SOFTWARE.get("gxtb") or {}).get("home") or "")
_GXTB_BINARY_DEFAULT = str((_SOFTWARE.get("gxtb") or {}).get("binary") or "")


def _detect_run_dir() -> Path:
    """Directory that holds run_configs.yaml (the active runs/<name>/ folder)."""
    cwd = Path.cwd()
    if (cwd / "run_configs.yaml").is_file():
        return cwd
    argv_dir = Path(sys.argv[0]).absolute().parent
    if (argv_dir / "run_configs.yaml").is_file():
        return argv_dir
    raise FileNotFoundError(
        "run_configs.yaml not found in cwd or script directory; "
        "run generate_workflow.py from a runs/<name>/ directory"
    )


_RUN_DIR = _detect_run_dir()
# Back-compat alias used throughout this file
_SCRIPT_DIR = _RUN_DIR
_PROJECT_ROOT = str(REPO_ROOT)


def _normalize_run_config_paths(config: dict) -> dict:
    """Resolve repo-relative / soft: paths in a single run config dict (in place)."""
    if not isinstance(config, dict):
        return config
    if "DFTBPLUS_skroot" in config:
        config["DFTBPLUS_skroot"] = str(resolve_dftb_skroot(config["DFTBPLUS_skroot"]))
    for key in ("GXTBHOME", "gxtb_home"):
        if key in config and config[key]:
            val = str(config[key]).strip()
            if val in ("soft:gxtb_home", "software:gxtb_home"):
                config[key] = _GXTB_HOME_DEFAULT
            else:
                config[key] = str(resolve_project_path(val))
    if "gxtb_path" in config and config["gxtb_path"]:
        val = str(config["gxtb_path"]).strip()
        if val in ("soft:gxtb_binary", "software:gxtb_binary"):
            config["gxtb_path"] = _GXTB_BINARY_DEFAULT
        else:
            config["gxtb_path"] = str(resolve_project_path(val))
    for key in ("mrcc_basis", "mrcc_optri", "mrcc_mp2fit", "mrcc_jkfit", "mrcc_ecp"):
        if key in config and config[key]:
            config[key] = str(resolve_project_path(config[key]))
    # ORCA custom basis file paths embedded in orca_template are left as-is;
    # run_configs should use %REPO_ROOT% or structures/... which we expand below.
    if "orca_template" in config and isinstance(config["orca_template"], str):
        tpl = config["orca_template"]
        tpl = tpl.replace("%REPO_ROOT%", str(REPO_ROOT))
        # Rewrite legacy absolute ORCA_basis paths if any remain
        legacy = "/home/zbai29/data/qmmm_test/ORCA_basis/"
        if legacy in tpl:
            tpl = tpl.replace(legacy, str(REPO_ROOT / "params" / "ORCA_basis") + "/")
        config["orca_template"] = tpl
    return config


# --- Configuration -----------------------------------------------------------

partial_struct_dict = {
    '24wat': {
        'xyz_name': 'IMH_24WAT',
        'mol_num_in_xyz': 25,
        'molid_to_include': list(range(1, 25)),
    },
}


with open(_RUN_DIR / "run_configs.yaml") as _f:
    run_configs = yaml.safe_load(_f)

# Normalize path-like fields for every run* config
for _k, _v in list(run_configs.items()):
    if isinstance(_v, dict) and str(_k).startswith("run"):
        _normalize_run_config_paths(_v)

# Extract reference QM run for minimize from YAML (required)
if "ref_qmrun_to_minimize" not in run_configs:
    raise KeyError(
        "Missing required top-level key 'ref_qmrun_to_minimize' in run_configs.yaml"
    )
ref_qmrun_to_minimize = run_configs.pop("ref_qmrun_to_minimize")
# Import key global flags from run_configs.yaml (lines 2-5)
CALC_MONOMER = run_configs.pop("CALC_MONOMER", False)
CALC_DIMER = run_configs.pop("CALC_DIMER", False)
CALC_TRIMER = run_configs.pop("CALC_TRIMER", False)
CALC_BSSE = run_configs.pop("CALC_BSSE", True)
EXPAND_NH_OH_RADIUS = run_configs.pop("expand_nh_oh_radius", False)
DELETE_WRONG_BONDS = run_configs.pop("delete_wrong_bonds", False)

_RUN_ID_KEY = re.compile(r"^run\d+$")


def is_run_id_key(key) -> bool:
    """True for YAML keys that name a QM run (run35, …), not distribute_tasks or other metadata."""
    return bool(_RUN_ID_KEY.match(str(key)))


def is_gxtb_run(config):
    """True if this run uses the g-xtb framework (no Amber; gxtb energy + xtb driver for optimization)."""
    qm = str(config.get("qm_level", "")).strip().lower()
    return qm in ("g-xtb", "gxtb")


def is_dftbplus_custom_run(config):
    """True if this run uses an external/custom DFTB+ input file."""
    qm = str(config.get("qm_level", "")).strip().lower()
    return qm == "dftbplus_custom"


def _validate_orca_template_nprocs(config, run_id: str) -> None:
    """
    If orca_template contains an nprocs setting, ensure it matches core_number.
    """
    tpl = config.get("orca_template")
    if not tpl:
        return

    core = config.get("core_number")
    if core is None:
        return

    m = re.search(r"^\s*nprocs\s+(\d+)\s*$", str(tpl), flags=re.MULTILINE | re.IGNORECASE)
    if not m:
        return

    nprocs_val = int(m.group(1))
    if nprocs_val != int(core):
        raise ValueError(
            f"run {run_id}: nprocs in orca_template ({nprocs_val}) "
            f"does not match core_number ({core})."
        )


_RESNAME_TO_CHARGE: dict[str, int] = {
    'ZN':  2,
    'MIM': -1,
    'IMH': 0,
    'IM-': -1,
    'MOH': 0,
    'NO3': -1,
    'WAT': 0,
    'MIH': 0,
    'H':   1,
    'H3O': 1,
    "O": -2,
    "OH": -1,
    "MO+": 1,
    "MI+": 1,
}

_fragment_charges_cache: dict = {}


# --- Topology utilities ------------------------------------------------------

def resolve_prmtop(directory: Path) -> Path:
    """Return directory/box.prmtop if it exists."""
    candidate = directory / "box.prmtop"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No box.prmtop found in {directory}")


def load_universe(source_path, source_type):
    """Load an MDAnalysis Universe from source_path based on source_type."""
    try:
        # Treat all *xyz-based source types the same
        if source_type in ("xyz", "qm_minimize_xyz", "gxtb_minimize_xyz"):
            # For xyz sources, optional N–H / O–H covalent bond radius expansion
            # (expand_nh_oh_radius) and optional guess_bonds cleanup (delete_wrong_bonds);
            # both from run_configs.yaml; defaults false.
            return xyz_to_mda(
                str(source_path),
                expand_nh_oh_radius=EXPAND_NH_OH_RADIUS,
                delete_wrong_bonds=DELETE_WRONG_BONDS,
            )
        if source_type == "qm_minimize_amber":
            # Per-molecule charges and residue grouping only need topology; do not
            # load min.rst here (MDAnalysis .rst handling is fragile; actual jobs
            # still use min.rst from disk via prepare.sh).
            prmtop = resolve_prmtop(Path(source_path))
            return mda.Universe(str(prmtop))
        if source_type == "minimize":
            prmtop = resolve_prmtop(Path(source_path))
            return mda.Universe(str(prmtop))
    except Exception as e:
        # Fail hard here so that missing/unsupported topologies do not silently
        # propagate and lead to wrong charges or atom counts.
        raise RuntimeError(
            f"Could not load universe from {source_path} "
            f"(source_type={source_type}): {e}"
        ) from e


def get_num_molecules(path) -> int:
    """Get number of molecules (residues) from a prmtop or xyz file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    path_str = str(path)
    if path_str.endswith(".xyz"):
        # Match load_universe: same xyz_to_mda flags as run_configs.yaml.
        u = xyz_to_mda(
            path_str,
            expand_nh_oh_radius=EXPAND_NH_OH_RADIUS,
            delete_wrong_bonds=DELETE_WRONG_BONDS,
        )
        n = max(r.resid for r in u.residues)
    elif path_str.endswith(".prmtop"):
        u = mda.Universe(path_str)
        n = len(u.residues)
    else:
        raise ValueError(f"{path} is not a valid prmtop or xyz file.")
    print(f"Found {n} molecules in {path}")
    return n


# --- Charge calculation -------------------------------------------------------

def get_per_molecule_charges(source_type, input_path):
    """Return a list of per-molecule charges (0-based index)."""
    if input_path in _fragment_charges_cache:
        return _fragment_charges_cache[input_path]

    u = load_universe(input_path, source_type)
    mol_charges = []
    if u is not None:
        for res in u.residues:
            charge = _RESNAME_TO_CHARGE.get(res.resname.upper())
            if charge is None:
                raise ValueError(
                    f"No charge defined for residue '{res.resname}' "
                    f"(resid {res.resid}). Please update _RESNAME_TO_CHARGE."
                )
            mol_charges.append(charge)

    _fragment_charges_cache[input_path] = mol_charges
    return mol_charges


def get_charge_for_structure(struc_name, source_type=None, input_path=None):
    """
    Return the total charge for a structure by summing per-molecule fragment charges.
    Parses _monomer_X, _dimer_X_Y, _trimer_X_Y_Z suffixes to select fragments.
    """
    def _mol_charges():
        if source_type is not None and input_path is not None:
            return get_per_molecule_charges(source_type, input_path)
        return None

    for pattern in [
        r'_monomer_(\d+)(_ghost)?$',
        r'_dimer_(\d+)_(\d+)$',
        r'_trimer_(\d+)_(\d+)_(\d+)$',
    ]:
        m = re.search(pattern, struc_name)
        if m:
            indices = [int(g) for g in m.groups() if g is not None and g.isdigit()]
            charges = _mol_charges()
            if charges is not None:
                return sum(charges[i] for i in indices if i < len(charges))

    charges = _mol_charges()
    if charges is not None:
        return sum(charges)

    print(f"Warning: No charge computed for '{struc_name}'. Defaulting to 0.")
    return 0


def get_charge_for_keep_molecules(keep_mol_indices, source_type, input_path):
    """Sum fragment charges for a specified list of 0-based molecule indices."""
    charges = get_per_molecule_charges(source_type, input_path)
    return sum(charges[i] for i in keep_mol_indices if i < len(charges))


def get_num_atoms(source_type, input_path):
    """Return the total number of atoms, or None if undetermined."""
    try:
        if source_type in ("xyz", "qm_minimize_xyz", "gxtb_minimize_xyz"):
            with open(input_path, 'r') as fh:
                return int(fh.readline().strip())
        if source_type in ("minimize", "qm_minimize_amber"):
            u = load_universe(input_path, source_type)
            if u:
                return len(u.atoms)
    except Exception as e:
        print(f"Warning: could not determine atom count for {input_path}: {e}")
    return None


# --- gxtb many-body: defer trimming to job-time --------------------------------

def _guess_xyz_symbol(atom) -> str:
    """Best-effort XYZ element symbol for an MDAnalysis atom."""
    symbol = getattr(atom, "element", "") or ""
    symbol = str(symbol).strip()
    if symbol:
        return symbol
    # Fallback: strip digits/punctuation from atom name, keep first 1-2 letters.
    name = re.sub(r"[^A-Za-z]", "", str(getattr(atom, "name", "")).strip())
    if not name:
        return "X"
    if len(name) >= 2 and name[1].islower():
        return name[:2]
    return name[:1]


def write_fragment_xyz_for_structure(
    struc_name: str,
    src_xyz: Path,
    dst_xyz: Path,
    keep_mols: list[int] | None,
) -> None:
    """Write structure-specific XYZ for ORCA-only mode using explicit keep_mols mapping."""
    if keep_mols is None:
        shutil.copy2(src_xyz, dst_xyz)
        return

    u = xyz_to_mda(
        str(src_xyz),
        expand_nh_oh_radius=EXPAND_NH_OH_RADIUS,
        delete_wrong_bonds=DELETE_WRONG_BONDS,
    )
    resid_terms = [f"resid {m + 1}" for m in keep_mols]
    atom_group = u.select_atoms(" or ".join(resid_terms))
    if len(atom_group) == 0:
        raise ValueError(
            f"No atoms selected for {struc_name} from {src_xyz} with keep_mols={keep_mols}"
        )

    with open(dst_xyz, "w") as f:
        f.write(f"{len(atom_group)}\n")
        f.write(f"{struc_name}\n")
        for atom in atom_group:
            x, y, z = atom.position
            elem = _guess_xyz_symbol(atom)
            f.write(f"{elem:2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")


# --- BSSE support -------------------------------------------------------------

def get_atom_indices_for_bsse(source_path, molecule_index):
    """Get 0-based atom indices for the monomer vs the rest of the system."""
    path_str = str(source_path)
    u = None
    if os.path.isdir(path_str):
        d = Path(path_str)
        prmtop = resolve_prmtop(d)
        u = mda.Universe(str(prmtop))
    elif path_str.endswith(".xyz"):
        u = xyz_to_mda(
            path_str,
            expand_nh_oh_radius=EXPAND_NH_OH_RADIUS,
            delete_wrong_bonds=DELETE_WRONG_BONDS,
        )
    elif path_str.endswith(".prmtop"):
        u = mda.Universe(path_str)

    if u is None:
        raise ValueError(f"Could not load topology from {source_path} for BSSE.")

    target_resid = molecule_index + 1
    monomer_atoms = u.select_atoms(f"resid {target_resid}")
    if len(monomer_atoms) == 0:
        raise ValueError(f"No residue for molecule_index {molecule_index} in {source_path}")

    monomer_indices = sorted(monomer_atoms.indices)
    rest_indices = sorted(set(u.atoms.indices) - set(monomer_indices))
    return monomer_indices, rest_indices


def get_fragment_atom_count(struc_name, source_type, input_path):
    """For monomer fragments, return the atom count of the specific residue.
    For everything else, return the total atom count of the structure."""
    m = re.search(r'_monomer_(\d+)(_ghost)?$', struc_name)
    if m:
        mol_idx = int(m.group(1))
        u = load_universe(input_path, source_type)
        if u is not None:
            target_resid = mol_idx + 1
            monomer_atoms = u.select_atoms(f"resid {target_resid}")
            return len(monomer_atoms)
    return get_num_atoms(source_type, input_path)


def configure_bsse_for_orca(struc_name, source_path, tpl_path):
    """Append BSSE configuration to ORCA template for ghost structures."""
    match = re.search(r"_monomer_(\d+)_ghost$", struc_name)
    if not match:
        return
    mol_idx = int(match.group(1))
    try:
        monomer_indices, rest_indices = get_atom_indices_for_bsse(source_path, mol_idx)
        monomer_str = " ".join(map(str, monomer_indices))
        rest_str = " ".join(map(str, rest_indices))

        bsse_block = f"""
### START THE PART TO MOVE TO THE FILE END
%frag
  Definition
    1 {{ {rest_str} }} end
    2 {{ {monomer_str} }} end
  end
end
%geom
  GhostFrags {{ 1 }} end
end
### END THE PART TO MOVE TO THE FILE END
"""
        with open(tpl_path, 'a') as f:
            f.write(bsse_block)
    except Exception as e:
        print(f"Error configuring BSSE for {struc_name}: {e}")


# --- Many-body decomposition -------------------------------------------------

def _add_partial_struct_entries(source_dict, keep_mols_map, structure_name, source_path):
    for key, val in partial_struct_dict.items():
        if val['xyz_name'] == structure_name:
            source_dict[key] = source_path
            selected = list(val.get('molid_to_include', []))
            keep_mols_map[key] = selected


def _add_manybody_entries(
    source_dict,
    keep_mols_map,
    name,
    path,
    num_molecules,
    has_orca: bool = False,
    has_molpro: bool = False,
    has_mrcc: bool = False,
):
    """Add full / monomer / dimer / trimer entries."""
    _add_partial_struct_entries(source_dict, keep_mols_map, name, path)
    source_dict[f"{name}_full"] = path
    keep_mols_map[f"{name}_full"] = None

    # Generate monomer jobs when explicitly requested, or when BSSE is enabled and
    # the QM backend supports/needs monomer energies (historically ORCA-only; extend to Molpro-only).
    if CALC_MONOMER or (CALC_BSSE and (has_orca or has_molpro or has_mrcc)):
        for i in range(num_molecules):
            source_dict[f"{name}_monomer_{i}"] = path
            keep_mols_map[f"{name}_monomer_{i}"] = [i]
    # Ghost jobs are ORCA-specific in this workflow.
    if CALC_BSSE and has_orca:
        for i in range(num_molecules):
            source_dict[f"{name}_monomer_{i}_ghost"] = path
            # Ghost jobs must keep full coordinates; ORCA handles ghost fragmenting.
            keep_mols_map[f"{name}_monomer_{i}_ghost"] = None
    # Molpro BSSE: also generate ghost jobs; dummy atoms will be written in run.input.
    if CALC_BSSE and has_molpro:
        for i in range(num_molecules):
            source_dict[f"{name}_monomer_{i}_ghost"] = path
            # Keep full coordinates; Molpro ghosting is done via dummy atoms in run.input.
            keep_mols_map[f"{name}_monomer_{i}_ghost"] = None
    # MRCC BSSE: ghosted atoms are specified via serial numbers in MINP.
    if CALC_BSSE and has_mrcc:
        for i in range(num_molecules):
            source_dict[f"{name}_monomer_{i}_ghost"] = path
            # Keep full coordinates; MRCC ghosting is done in MINP (ghost=serialno).
            keep_mols_map[f"{name}_monomer_{i}_ghost"] = None
    if CALC_DIMER:
        for i in range(num_molecules):
            for j in range(i + 1, num_molecules):
                source_dict[f"{name}_dimer_{i}_{j}"] = path
                keep_mols_map[f"{name}_dimer_{i}_{j}"] = [i, j]
    if CALC_TRIMER:
        for i in range(num_molecules):
            for j in range(i + 1, num_molecules):
                for k in range(j + 1, num_molecules):
                    source_dict[f"{name}_trimer_{i}_{j}_{k}"] = path
                    keep_mols_map[f"{name}_trimer_{i}_{j}_{k}"] = [i, j, k]


def _enrich_with_manybody(
    source_dict,
    keep_mols_map,
    stem,
    source_path,
    topo_path,
    has_orca: bool = False,
    has_molpro: bool = False,
    has_mrcc: bool = False,
):
    """Add a structure: plain if monomer-like, else with many-body decomposition."""
    if "monomer" in stem.lower():
        source_dict[stem] = source_path
        keep_mols_map[stem] = None
    else:
        num_mol = get_num_molecules(topo_path)
        _add_manybody_entries(
            source_dict, keep_mols_map, stem, source_path, num_mol,
            has_orca=has_orca, has_molpro=has_molpro, has_mrcc=has_mrcc
        )


# --- Source collection --------------------------------------------------------

def _collect_sources(items, source_type, enrich_with_manybody, has_orca=False, has_molpro=False, has_mrcc=False):
    """Common collection logic.
    items: iterable of (stem, source_path, topo_path).
    Returns (dict, source_type, keep_mols_map) or None."""
    sources = {}
    keep_mols_map = {}
    for stem, source_path, topo_path in items:
        if enrich_with_manybody:
            _enrich_with_manybody(
                sources,
                keep_mols_map,
                stem,
                source_path,
                topo_path,
                has_orca=has_orca,
                has_molpro=has_molpro,
                has_mrcc=has_mrcc,
            )
        else:
            sources[stem] = source_path
            keep_mols_map[stem] = None
    return (sources, source_type, keep_mols_map) if sources else None


def _collect_from_xyz_dir(xyz_dir, enrich_with_manybody, has_orca=False, has_molpro=False, has_mrcc=False):
    """Collect from xyz_dir/*.xyz. Returns (dict, 'xyz') or None."""
    if not xyz_dir.exists():
        print(f"Warning: {xyz_dir} does not exist.")
        return None
    items = [(f.stem, str(f), f) for f in sorted(xyz_dir.glob("*.xyz"))]
    return _collect_sources(items, "xyz", enrich_with_manybody, has_orca, has_molpro, has_mrcc)


def _collect_from_qm_minimize(qm_dir, enrich_with_manybody, has_orca=False, has_molpro=False, has_mrcc=False):
    """Collect from qm_minimize/*/min.xyz. Returns (dict, 'qm_minimize_xyz') or None."""
    if not qm_dir.exists():
        return None
    min_files = sorted(qm_dir.glob("*/min.xyz"))
    if not min_files:
        return None
    print(f"Using minimized structures from {qm_dir}")
    items = [(f.parent.name, str(f), f) for f in min_files]
    return _collect_sources(items, "qm_minimize_xyz", enrich_with_manybody, has_orca, has_molpro, has_mrcc)


def _collect_from_qm_minimize_amber(qm_dir, enrich_with_manybody, has_orca=False, has_molpro=False, has_mrcc=False):
    """Collect from qm_minimize/*/min.rst + prmtop. Returns (dict, 'qm_minimize_amber') or None."""
    if not qm_dir.exists():
        return None
    items = []
    for d in sorted(qm_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "min.rst").is_file():
            continue
        try:
            topo_path = resolve_prmtop(d)
        except FileNotFoundError:
            continue
        items.append((d.name, str(d), topo_path))
    if not items:
        return None
    print(f"Using qm_minimize Amber (min.rst + prmtop) from {qm_dir}")
    return _collect_sources(items, "qm_minimize_amber", enrich_with_manybody, has_orca, has_molpro, has_mrcc)


def _collect_from_minimize_run(run_dir, enrich_with_manybody, has_orca=False, has_molpro=False, has_mrcc=False):
    """Collect from a minimize run directory (enrich with manybody if specified).
    Returns (dict, 'minimize') or None."""
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return None
    print(f"Using minimized run {run_dir.name} as source for topology/coordinates.")
    items = [(d.name, str(d), d / "box.prmtop")
             for d in sorted(run_dir.iterdir())
             if d.is_dir() and (d / "prepare.sh").exists()]
    return _collect_sources(items, "minimize", enrich_with_manybody, has_orca, has_molpro, has_mrcc)


def _collect_from_minimize_run_xyz(run_dir, enrich_with_manybody, has_orca=False):
    """Collect xyz paths from a (gxtb) minimize run: structure -> run_dir/structure/min.xyz.
    Used for gxtb sp_init/sp_opt when input is xyz from a previous gxtb minimize.
    Returns (dict, 'gxtb_minimize_xyz') or None."""
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return None
    items = [(d.name, str(d / "min.xyz"), d / "min.xyz")
             for d in sorted(run_dir.iterdir())
             if d.is_dir() and (d / "min.xyz").exists()]
    if not items:
        return None
    print(f"Using minimized xyz from {run_dir.name} as source for gxtb.")
    return _collect_sources(items, "gxtb_minimize_xyz", enrich_with_manybody, has_orca)


# --- Charge selection ---------------------------------------------------------

def get_charge_molecule_indices(struc_name: str, keep_mols: list[int] | None) -> list[int] | None:
    """Return molecule indices used for charge summation for a structure."""
    # Ghost structures keep full coordinates but use monomer charge.
    m = re.search(r'_monomer_(\d+)_ghost$', struc_name)
    if m:
        return [int(m.group(1))]
    return keep_mols


# --- Main source resolution --------------------------------------------------

def get_run_sources(mode, run_id, base_dir):
    """
    Roots (each holds one subdirectory per structure name):

      qm_dir  — qm_minimize/<ref_qmrun_to_minimize>/
      min_dir — minimize/<run_id>/
      xyz_dir — xyz/xyz_files/

    History: before 2026-03-22 only the reinit_topology True paths existed.
    On 2026-03-22 reinit_topology False was added and make the default value False;
    This aims to use min.rst rather than min.xyz in common use cases to avoid round error 
    of atom positions during topology re-initialization using min.xyz.


    Amber
      qm_minimize: xyz_dir (*.xyz)
      minimize + reinit_topology True: qm_dir (min.xyz) -> xyz_dir (*.xyz)
      minimize + reinit_topology False: qm_dir (box.prmtop, min.rst) -> xyz_dir (*.xyz)
      sp_init + reinit_topology True: min_dir (box.prmtop, box_orig.inpcrd) -> qm_dir (min.xyz) -> xyz_dir (*.xyz)
      sp_init + reinit_topology False: qm_dir (box.prmtop, min.rst) -> xyz_dir (*.xyz)
      sp_opt: min_dir (box.prmtop, min.rst)

    ORCA (when `use_amber_interface=false` and 'orca_template' in config)
      qm_minimize: xyz_dir (*.xyz)
      minimize (skipped): qm_dir (min.xyz) -> xyz_dir (*.xyz)
      sp_init: qm_dir (min.xyz) -> xyz_dir (*.xyz)
      sp_opt (skipped): min_dir (min.xyz)

    g-xTB (reinit_topology not used)
      minimize | sp_init: qm_dir (min.xyz) -> xyz_dir (*.xyz)
      sp_opt: min_dir (min.xyz)

    """
    config = run_configs.get(run_id, {})
    has_orca = 'orca_template' in config
    has_molpro = 'molpro_template' in config
    has_mrcc = 'mrcc_template' in config
    gxtb = is_gxtb_run(config)
    reinit_topology = config.get("reinit_topology", False)
    use_amber_interface = bool(config.get("use_amber_interface", False))
    orca_use_xyz_only = has_orca and not use_amber_interface

    xyz_dir = base_dir / "xyz" / "xyz_files"
    qm_dir  = base_dir / "qm_minimize" / ref_qmrun_to_minimize
    min_dir = base_dir / "minimize" / run_id

    if mode == "qm_minimize":
        return _collect_from_xyz_dir(xyz_dir, enrich_with_manybody=False) or ({}, "xyz", {})

    if gxtb:
        # gxtb: only xyz inputs; no prepare_template
        if mode == "minimize":
            return (
                _collect_from_qm_minimize(qm_dir, enrich_with_manybody=False)
                or _collect_from_xyz_dir(xyz_dir, enrich_with_manybody=False)
                or ({}, "xyz", {})
            )
        if mode == "sp_init":
            return (
                _collect_from_qm_minimize(qm_dir, enrich_with_manybody=True, has_orca=False)
                or _collect_from_xyz_dir(xyz_dir, enrich_with_manybody=True, has_orca=False)
                or ({}, "xyz", {})
            )
        if mode == "sp_opt":
            return (
                _collect_from_minimize_run_xyz(min_dir, enrich_with_manybody=True, has_orca=False)
                or ({}, "gxtb_minimize_xyz", {})
            )
        return ({}, "xyz", {})

    if mode == "minimize":
        # ORCA-only: never depend on reinit_topology (we don't build/copy Amber topology).
        if orca_use_xyz_only or reinit_topology:
            qm_src = _collect_from_qm_minimize(qm_dir, enrich_with_manybody=False)
        else:
            qm_src = _collect_from_qm_minimize_amber(qm_dir, enrich_with_manybody=False)
        return (
            qm_src
            or _collect_from_xyz_dir(xyz_dir, enrich_with_manybody=False)
            or ({}, "xyz", {})
        )

    if mode == "sp_init":
        if orca_use_xyz_only:
            # ORCA-only: qm_dir(min.xyz) -> xyz_dir.
            return (
                _collect_from_qm_minimize(
                    qm_dir, enrich_with_manybody=True,
                    has_orca=has_orca, has_molpro=has_molpro, has_mrcc=has_mrcc
                )
                or _collect_from_xyz_dir(
                    xyz_dir, enrich_with_manybody=True,
                    has_orca=has_orca, has_molpro=has_molpro, has_mrcc=has_mrcc
                )
                or ({}, "xyz", {})
            )
        if reinit_topology:
            return (
                _collect_from_minimize_run(
                    min_dir, enrich_with_manybody=True,
                    has_orca=has_orca, has_molpro=has_molpro, has_mrcc=has_mrcc
                )
                or _collect_from_qm_minimize(
                    qm_dir, enrich_with_manybody=True,
                    has_orca=has_orca, has_molpro=has_molpro, has_mrcc=has_mrcc
                )
                or _collect_from_xyz_dir(
                    xyz_dir, enrich_with_manybody=True,
                    has_orca=has_orca, has_molpro=has_molpro, has_mrcc=has_mrcc
                )
                or ({}, "xyz", {})
            )
        return (
            _collect_from_qm_minimize_amber(
                qm_dir, enrich_with_manybody=True,
                has_orca=has_orca, has_molpro=has_molpro, has_mrcc=has_mrcc
            )
            or _collect_from_xyz_dir(
                xyz_dir, enrich_with_manybody=True,
                has_orca=has_orca, has_molpro=has_molpro, has_mrcc=has_mrcc
            )
            or ({}, "xyz", {})
        )

    if mode == "sp_opt":
        if orca_use_xyz_only:
            # ORCA-only: use minimized xyz rather than Amber prmtop/min.rst.
            return (
                _collect_from_minimize_run_xyz(min_dir, enrich_with_manybody=True, has_orca=has_orca)
                or ({}, "gxtb_minimize_xyz", {})
            )
        return (
            _collect_from_minimize_run(
                min_dir, enrich_with_manybody=True,
                has_orca=has_orca, has_molpro=has_molpro, has_mrcc=has_mrcc
            )
            or ({}, "minimize", {})
        )

    return {}, "xyz", {}


def generate_orca_only_structure(
    *,
    mode: str,
    struc_dir: Path,
    struc_name: str,
    source_path: str,
    source_type: str,
    config: dict,
    keep_mols: list[int] | None = None,
    qm_charge: int | None = None,
) -> None:
    """
    ORCA-only init:
      - copy xyz/min.xyz into MOL.xyz
      - render orc_job.inp from orc_job.tpl + append the xyzfile/charge block
      - write sbatch_prepare.sh to run `orca orc_job.inp`
    """
    src = Path(source_path)

    # Requirement: accept xyz file or directory with min.xyz.
    if src.is_file() and src.suffix.lower() == ".xyz":
        src_xyz = src
    elif src.is_dir() and (src / "min.xyz").is_file():
        src_xyz = src / "min.xyz"
    else:
        raise FileNotFoundError(
            f"ORCA-only expects an xyz file or a directory containing min.xyz; got {src} "
            f"for {struc_name}"
        )

    dst_mol_xyz = struc_dir / "MOL.xyz"
    write_fragment_xyz_for_structure(struc_name, src_xyz, dst_mol_xyz, keep_mols=keep_mols)

    # Use precomputed structure charge when available.
    if qm_charge is None:
        qm_charge = get_charge_for_structure(
            struc_name,
            source_type=source_type,
            input_path=source_path,
        )

    # Create orc_job.inp from orc_job.tpl + append final xyzfile block with charge.
    tpl_path = struc_dir / "orc_job.tpl"
    with open(tpl_path, "w") as f:
        f.write(config["orca_template"])
    configure_bsse_for_orca(struc_name, str(src_xyz), tpl_path)

    with open(tpl_path, "r") as f:
        tpl_content = f.read()

    append_lines = []
    engrad_for_sp = config.get("engrad_for_sp", False)
    # Allow YAML/JSON booleans as well as common string forms.
    if isinstance(engrad_for_sp, str):
        engrad_for_sp = engrad_for_sp.strip().lower() in ("1", "true", "yes", "y", "on")
    else:
        engrad_for_sp = bool(engrad_for_sp)
    if mode in ("qm_minimize", "minimize"):
        append_lines.append("! Opt Angs NoUseSym")
    else:
        append_lines.append("! ENGRAD Angs NoUseSym" if engrad_for_sp else "! Angs NoUseSym")
    append_lines.append(f"*xyzfile {qm_charge} 1 MOL.xyz")
    append_snippet = "\n".join(append_lines) + "\n"
    orc_job_inp_content = tpl_content.rstrip() + "\n" + append_snippet
    orc_inp_path = struc_dir / "orc_job.inp"
    with open(orc_inp_path, "w") as f:
        f.write(orc_job_inp_content)

    # Temporary tpl not required after orc_job.inp is written.
    if tpl_path.exists():
        tpl_path.unlink()

    core = config.get("core_number", config.get("core", 1))
    sbatch_prepare_path = struc_dir / "sbatch_prepare.sh"
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=prepare_sbatch.log
#SBATCH --error=prepare_sbatch.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks={core}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --exclude=compute-0-[0-40,44]
source {_CONDAINIT}
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:-}}"
export PATH="$PWD:$PATH:{_ORCA_PATH}/"
orca orc_job.inp > orc_job.dat 2>&1
rm -f orc_job.gbw orc_job.densities
cp orc_job.xyz min.xyz
"""
    with open(sbatch_prepare_path, "w") as f:
        f.write(sbatch_content)
    os.chmod(
        sbatch_prepare_path,
        os.stat(sbatch_prepare_path).st_mode | stat.S_IEXEC,
    )


# --- Molpro-only init ---------------------------------------------------------

def _read_xyz_atoms(xyz_path: Path) -> list[tuple[str, float, float, float]]:
    """
    Read an XYZ file (standard 1st line atom count, 2nd comment).
    Returns list of (element, x, y, z).
    """
    lines = xyz_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ too short: {xyz_path}")
    # best-effort: if first token is integer atom count, skip first two lines
    start = 0
    try:
        int(lines[0].strip().split()[0])
        start = 2
    except Exception:
        start = 0
    atoms: list[tuple[str, float, float, float]] = []
    for raw in lines[start:]:
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) < 4:
            continue
        elem = parts[0].strip()
        x, y, z = map(float, parts[1:4])
        atoms.append((elem, x, y, z))
    if not atoms:
        raise ValueError(f"No atoms parsed from {xyz_path}")
    return atoms


def _resolve_source_xyz_for_qm_only(source_path: str) -> Path:
    """Resolve xyz path from either an xyz file or a directory containing min.xyz."""
    src = Path(source_path)
    if src.is_file() and src.suffix.lower() == ".xyz":
        return src
    if src.is_dir() and (src / "min.xyz").is_file():
        return src / "min.xyz"
    raise FileNotFoundError(
        f"Molpro/ORCA qm-only mode expects an xyz file or a directory containing min.xyz; got {src}"
    )


def _atoms_from_source_with_keep_mols(
    source_path: str,
    keep_mols: list[int] | None,
) -> list[tuple[str, float, float, float]]:
    """
    Return atoms from source xyz, optionally restricted to specific molecule indices.
    keep_mols uses 0-based molecule indexing consistent with keep_mols_map.
    """
    src_xyz = _resolve_source_xyz_for_qm_only(source_path)
    if keep_mols is None:
        return _read_xyz_atoms(src_xyz)

    u = xyz_to_mda(
        str(src_xyz),
        expand_nh_oh_radius=EXPAND_NH_OH_RADIUS,
        delete_wrong_bonds=DELETE_WRONG_BONDS,
    )
    resid_terms = [f"resid {m + 1}" for m in keep_mols]
    atom_group = u.select_atoms(" or ".join(resid_terms))
    if len(atom_group) == 0:
        raise ValueError(
            f"No atoms selected from {src_xyz} with keep_mols={keep_mols}"
        )

    atoms: list[tuple[str, float, float, float]] = []
    for atom in atom_group:
        x, y, z = atom.position
        atoms.append((_guess_xyz_symbol(atom), float(x), float(y), float(z)))
    return atoms


def _label_atoms_with_element_counters(
    atoms: list[tuple[str, float, float, float]],
) -> list[tuple[str, str, float, float, float]]:
    """
    Molpro-style labels: append a per-element running index (Zn1, H1, H2, ...).
    Returns (element, label, x, y, z).
    """
    counters: dict[str, int] = {}
    labeled: list[tuple[str, str, float, float, float]] = []
    for elem, x, y, z in atoms:
        key = elem.strip()
        counters[key] = counters.get(key, 0) + 1
        label = f"{key}{counters[key]}"
        labeled.append((key, label, x, y, z))
    return labeled


def _infer_molpro_dummy_labels(struc_name: str, labeled_atoms: list[tuple[str, str, float, float, float]]) -> list[str]:
    """
    Infer dummy atoms from structure name.
    Convention supported:
      - Names beginning with 'dummy<Elem>_' or exactly 'dummy<Elem>' dummy all atoms with that element.
        Example: dummyZn_MeOH -> dummy all Zn* atoms -> ['zn1', ...]
    Returns list of dummy labels in lowercase (Molpro dummy expects atom labels).
    """
    s = str(struc_name).strip()
    m = re.match(r"(?i)^dummy([A-Za-z]{1,2})(?:_|$)", s)
    if not m:
        return []
    elem = m.group(1)
    out: list[str] = []
    for e, label, *_ in labeled_atoms:
        if e.lower() == elem.lower():
            out.append(label.lower())
    return out


def _render_molpro_geometry_block(labeled_atoms: list[tuple[str, str, float, float, float]]) -> str:
    lines = ["geometry={", "angstrom"]
    for _, label, x, y, z in labeled_atoms:
        lines.append(f"{label:6s} {x:16.8f} {y:16.8f} {z:16.8f}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _replace_or_insert_geometry_and_dummy(
    template: str,
    geometry_block: str,
    dummy_labels: list[str],
) -> str:
    """
    Place geometry and dummy into template.
    Supports:
      - %GEOMETRY% placeholder
      - existing geometry={...} block replacement
      - fallback insert after first 'basis=' line (or at file start)
    Dummy supports:
      - %DUMMY% placeholder
      - replace existing 'dummy,' line
      - insert right after geometry block if needed
    """
    dummy_line = ""
    if dummy_labels:
        dummy_line = "dummy," + ",".join(dummy_labels) + "\n"

    content = str(template)
    if "%GEOMETRY%" in content:
        content = content.replace("%GEOMETRY%", geometry_block.rstrip("\n"))
    else:
        # Replace an existing geometry={...} block if present (non-greedy).
        if re.search(r"(?is)\bgeometry\s*=\s*\{.*?\}\s*", content):
            content = re.sub(r"(?is)\bgeometry\s*=\s*\{.*?\}\s*", geometry_block, content, count=1)
        else:
            # Insert after basis line if possible, else at top.
            m_basis = re.search(r"(?im)^\s*basis\s*=.*?$", content)
            if m_basis:
                insert_at = m_basis.end()
                content = content[:insert_at] + "\n\n" + geometry_block + content[insert_at:]
            else:
                content = geometry_block + "\n" + content

    # Handle dummy placement
    if "%DUMMY%" in content:
        content = content.replace("%DUMMY%", dummy_line.rstrip("\n"))
    else:
        if re.search(r"(?im)^\s*dummy\s*,.*$", content):
            if dummy_line:
                content = re.sub(r"(?im)^\s*dummy\s*,.*$\n?", dummy_line, content, count=1)
            else:
                # remove dummy line
                content = re.sub(r"(?im)^\s*dummy\s*,.*$\n?", "", content, count=1)
        else:
            if dummy_line:
                # insert immediately after closing geometry brace
                content = re.sub(r"(?m)^\}\s*$", "}\n" + dummy_line.rstrip("\n"), content, count=1)

    return content


def _patch_molpro_wf_charge(template: str, charge: int) -> str:
    """
    Update wf,charge=... in molpro template if present.
    """
    return re.sub(
        r"(?im)^\s*wf\s*,\s*charge\s*=\s*[-+]?\d+",
        f"wf,charge={int(charge)}",
        str(template),
        count=1,
    )


def _render_molpro_charge_spin(charge: int, spin: int = 0) -> str:
    # Molpro expects spin as 2S (0 for singlet).
    return f"wf,charge={int(charge)},spin={int(spin)}"


def generate_molpro_only_structure(
    *,
    struc_dir: Path,
    struc_name: str,
    source_path: str,
    config: dict,
    qm_charge: int,
    keep_mols: list[int] | None = None,
) -> None:
    """
    Molpro-only init:
      - read xyz/min.xyz
      - create run.input from molpro_template + rendered geometry (+ optional dummy)
      - create run.sbatch strictly based on molpro/run5/MeOH/run.sbatch with fixed nodelist compute-1-5
    """
    # Non-ghost monomer/dimer/trimer jobs should use fragment geometry.
    # Ghost jobs keep full geometry and apply dummy atoms in run.input.
    m_ghost = re.search(r"_monomer_(\d+)_ghost$", str(struc_name))
    effective_keep_mols = None if m_ghost else keep_mols
    atoms = _atoms_from_source_with_keep_mols(source_path, effective_keep_mols)
    labeled = _label_atoms_with_element_counters(atoms)
    geometry_block = _render_molpro_geometry_block(labeled)
    dummy_labels: list[str] = []
    # BSSE ghost jobs: dummy everything except the target monomer.
    if m_ghost:
        mol_idx = int(m_ghost.group(1))
        try:
            _, rest_indices = get_atom_indices_for_bsse(source_path, mol_idx)
        except Exception as e:
            raise RuntimeError(f"Could not infer BSSE dummy atoms for {struc_name} from {source_path}: {e}") from e
        # Map 0-based atom indices -> Molpro labels, lowercase.
        idx_to_label = [lab.lower() for (_, lab, *_xyz) in labeled]
        dummy_labels = [idx_to_label[i] for i in rest_indices if 0 <= i < len(idx_to_label)]
    else:
        # Fallback convention: allow manual dummy by prefixing structure name with dummy<Element>_
        dummy_labels = _infer_molpro_dummy_labels(struc_name, labeled)

    tpl = str(config.get("molpro_template", ""))
    if not tpl.strip():
        raise ValueError("molpro_template is empty")
    symmetry_line = "" if len(atoms) == 1 else "symmetry,nosym"
    if "%SYMMETRY%" in tpl:
        tpl = tpl.replace("%SYMMETRY%", symmetry_line)
    if "%CHARGE_SPIN%" in tpl:
        tpl = tpl.replace("%CHARGE_SPIN%", _render_molpro_charge_spin(qm_charge, spin=0))
    else:
        tpl = _patch_molpro_wf_charge(tpl, qm_charge)
    content = _replace_or_insert_geometry_and_dummy(tpl, geometry_block, dummy_labels)
    (struc_dir / "run.input").write_text(content.rstrip() + "\n")

    # sbatch strictly following molpro/run5/MeOH/run.sbatch (with requested naming/logs)
    core = int(config.get("core_number", config.get("core", 32)))
    # Reference script uses --ntasks=31 for -n 32
    ntasks = max(1, core)
    sbatch = f"""#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=prepare_sbatch.log
#SBATCH --error=prepare_sbatch.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8448M
#SBATCH --exclude=compute-0-[0-40,44]

export inputfilename='run.input'
export LANG=en_US
export PATH="{_MOLPRO_ROOT}/bin:/sbin:/usr/sbin:/bin:/usr/bin:${{PATH}}"
export LD_LIBRARY_PATH="{_MOLPRO_ROOT}/lib:$LD_LIBRARY_PATH"
TEMPLATE="${{SLURM_JOB_ID}}_XXXXXX"
SCRATCH_DIR=$(mktemp -d "{_SCRATCH_ROOT}/${{TEMPLATE}}" 2>&1)
rm -rf /tmp/molpro*
cp $inputfilename "${{SCRATCH_DIR}}/"
(
cd "${{SCRATCH_DIR}}"
unshare --net --user --map-root-user bash -lc "
ip link set lo up
for var in \$(compgen -v | grep SLURM); do unset \$var; done
export I_MPI_HYDRA_BOOTSTRAP=fork
export HYDRA_BOOTSTRAP=fork
export I_MPI_FABRICS=shm; molpro -n {core} --ga-impl disk -m 660m --stdout $inputfilename > $SLURM_SUBMIT_DIR/molpro.log 2>&1
"
cp -rf ./* "$SLURM_SUBMIT_DIR/"
)
"""
    sbatch_path = struc_dir / "sbatch_prepare.sh"
    sbatch_path.write_text(sbatch)
    os.chmod(sbatch_path, os.stat(sbatch_path).st_mode | stat.S_IEXEC)


# --- MRCC-only init -----------------------------------------------------------

def _format_serial_ranges(indices_1based: list[int]) -> str:
    """Format sorted serial numbers as MRCC compact ranges: 1-3,5,8-10."""
    if not indices_1based:
        return ""
    nums = sorted(set(indices_1based))
    ranges: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = n
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(ranges)


def _render_mrcc_geometry_block(atoms: list[tuple[str, float, float, float]]) -> str:
    lines = ["geom=xyz", str(len(atoms)), ""]
    for elem, x, y, z in atoms:
        lines.append(f"{elem:<2s} {x:16.8f} {y:16.8f} {z:16.8f}")
    # Keep one blank line after xyz geometry block for readability/parsers.
    return "\n".join(lines) + "\n"


def _render_mrcc_charge_spin(charge: int, mult: int = 1) -> str:
    return f"charge={int(charge)}\nmult={int(mult)}"


def _render_mrcc_ghost_block(struc_name: str, source_path: str) -> str:
    m_ghost = re.search(r"_monomer_(\d+)_ghost$", str(struc_name))
    if not m_ghost:
        return ""
    mol_idx = int(m_ghost.group(1))
    try:
        _, rest_indices = get_atom_indices_for_bsse(source_path, mol_idx)
    except Exception as e:
        raise RuntimeError(f"Could not infer MRCC ghost atoms for {struc_name} from {source_path}: {e}") from e
    serial_range = _format_serial_ranges([i + 1 for i in rest_indices])
    if not serial_range:
        return ""
    return f"ghost=serialno\n{serial_range}"


def generate_mrcc_only_structure(
    *,
    struc_dir: Path,
    struc_name: str,
    source_path: str,
    config: dict,
    qm_charge: int,
    keep_mols: list[int] | None = None,
) -> None:
    """
    MRCC-only init:
      - create MINP from mrcc_template + rendered geometry/charge/(optional) ghost
      - write sbatch_prepare.sh in MRCC style
      - copy run-level GENBAS into structure directory
    """
    m_ghost = re.search(r"_monomer_(\d+)_ghost$", str(struc_name))
    effective_keep_mols = None if m_ghost else keep_mols
    atoms = _atoms_from_source_with_keep_mols(source_path, effective_keep_mols)

    tpl = str(config.get("mrcc_template", ""))
    if not tpl.strip():
        raise ValueError("mrcc_template is empty")

    content = tpl
    content = content.replace("%CHARGE_SPIN%", _render_mrcc_charge_spin(qm_charge, mult=1))
    content = content.replace("%GEOMETRY%", _render_mrcc_geometry_block(atoms))
    content = content.replace("%GHOST%", _render_mrcc_ghost_block(struc_name, source_path))
    (struc_dir / "MINP").write_text(content.rstrip() + "\n")

    core = int(config.get("core_number", config.get("core", 32)))
    sbatch = f"""#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=prepare_sbatch.log
#SBATCH --error=prepare_sbatch.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks={core}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --exclude=compute-0-[0-40,44]

export OMP_NUM_THREADS={core}
export MKL_NUM_THREADS={core}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread,close
export PATH="{_MRCC_PATH}:$PATH"
TEMPLATE="${{SLURM_JOB_ID}}_XXXXXX"
SCRATCH_DIR=$(mktemp -d "{_SCRATCH_ROOT}/${{TEMPLATE}}" 2>&1)
cp GENBAS MINP "${{SCRATCH_DIR}}/"

export I_MPI_SPAWN=on

(
cd "${{SCRATCH_DIR}}"
dmrcc > "$SLURM_SUBMIT_DIR/mrcc.log" 2>&1
#cp -rf ./* "$SLURM_SUBMIT_DIR/"
)
rm -rf "${{SCRATCH_DIR}}"
"""
    sbatch_path = struc_dir / "sbatch_prepare.sh"
    sbatch_path.write_text(sbatch)
    os.chmod(sbatch_path, os.stat(sbatch_path).st_mode | stat.S_IEXEC)


def generate_mrcc_genbas(run_dir: Path, config: dict) -> Path:
    """Generate run-level GENBAS using in-process make_mrcc_genbas helpers."""
    required = ["mrcc_basis", "mrcc_optri", "mrcc_mp2fit", "mrcc_jkfit", "mrcc_ecp"]
    missing = [k for k in required if not str(config.get(k, "")).strip()]
    if missing:
        raise ValueError(f"Missing MRCC GENBAS config keys: {', '.join(missing)}")

    genbas_path = run_dir / "GENBAS"
    # Keep same default element set as make_mrcc_genbas CLI.
    elements = parse_mrcc_elements_arg(str(config.get("mrcc_elements", "H,C,N,O,Zn")))
    if not elements:
        raise ValueError("mrcc_elements resolves to an empty element list.")

    make_mrcc_generate_genbas(
        output_path=genbas_path,
        elements=elements,
        basis_file=Path(config["mrcc_basis"]),
        optri_file=Path(config["mrcc_optri"]),
        mp2fit_file=Path(config["mrcc_mp2fit"]),
        jkfit_file=Path(config["mrcc_jkfit"]),
        ecp_file=Path(config["mrcc_ecp"]),
    )
    if not genbas_path.exists():
        raise FileNotFoundError(f"Failed to generate GENBAS at {genbas_path}")
    return genbas_path


# --- gxtb: no prepare_template, only xyz + sbatch prepare script --------------

def write_gxtb_prepare_script(mode, charge, config, run_id, trim_script_path: Path):
    """
    Generate self-contained prepare.sh (sbatch-ready) for gxtb runs.
    Assumes source.xyz exists. If KEEP_MOLS exists, input.xyz is generated by trimming source.xyz at job-time.
    .CHRG is written by the script.
    Uses: xtb for optimization (with gxtb -grad as driver), gxtb for single-point.
    Config: GXTBHOME (dir with .gxtb, .eeq, .basisq), gxtb_path (dir with gxtb binary, prepended to PATH).
    """
    nproc = str(config.get("core_number", 1))
    # optional: loose convergence for numerical gradient (README)
    opt_extra = config.get("gxtb_opt_loose", False) and " --opt loose" or " --opt"

    gxtb_home = config.get("GXTBHOME", config.get("gxtb_home", "")) or _GXTB_HOME_DEFAULT
    gxtb_path = config.get("gxtb_path", "") or _GXTB_BINARY_DEFAULT
    env_lines = []
    if gxtb_home:
        env_lines.append(f'export GXTBHOME="{gxtb_home}"')
    if gxtb_path:
        env_lines.append(f'export PATH="{gxtb_path}:$PATH"')
    env_block = "\n".join(env_lines) + "\n" if env_lines else ""

    script = f"""#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=prepare.log
#SBATCH --error=prepare.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={nproc}
#SBATCH --mem-per-cpu=8G
set -euo pipefail

# gxtb mode: no Amber; create input.xyz on the fly if needed
source {_CONDAINIT}
{env_block}
echo "{charge}" > .CHRG

if [[ -f "KEEP_MOLS" ]]; then
  python "{trim_script_path}" --in source.xyz --out input.xyz --keep-file KEEP_MOLS
else
  cp -f source.xyz input.xyz
fi

"""
    if mode == "minimize":
        script += f"""# Geometry optimization with xtb using gxtb as driver (numerical gradient)
xtb input.xyz --driver "gxtb -grad -c xtbdriver.xyz"{opt_extra}
cp xtbopt.xyz min.xyz
echo "min.xyz written."
"""
    elif mode == "sp_init":
        script += """# Single-point energy with gxtb
gxtb -c input.xyz -molden > sp_init.out 2>&1
echo "sp_init.out written."
"""
    elif mode == "sp_opt":
        script += """# Single-point energy with gxtb on (already optimized) input.xyz
gxtb -c input.xyz -molden > sp_opt.out 2>&1
echo "sp_opt.out written."
"""
    else:
        raise ValueError(f"gxtb mode unknown: {mode}")
    return script


def write_dftbplus_custom_prepare_script(mode, config, run_id, dftb_input_filename: str):
    """
    Generate sbatch_prepare.sh for runs that use an external/custom DFTB+ input file.
    Assumes that:
      - geometry for this structure was already copied (e.g. geom.xyz)
      - the DFTB+ input file was copied to dftb_input_filename (typically dftb_in.hsd)
    """
    nproc = str(config.get("core_number", 1))
    conda_env = config.get("dftbplus_conda_env", "mybase")

    out_name_map = {
        "minimize": "dftb_minimize.out",
        "sp_init": "dftb_sp_init.out",
        "sp_opt": "dftb_sp_opt.out",
        "qm_minimize": "dftb_qm_minimize.out",
    }
    out_name = out_name_map.get(mode, "dftbplus.out")

    script = f"""#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=prepare.log
#SBATCH --error=prepare.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={nproc}
#SBATCH --mem-per-cpu=8G
set -euo pipefail

source {_CONDAINIT}
conda activate {conda_env}

# Run DFTB+ with the provided input file (default name: dftb_in.hsd)
srun dftb+ > {out_name} 2>&1
echo "{out_name} written."
"""
    return script


# --- Template rendering ------------------------------------------------------

def create_notes_yaml(run_id, config):
    return (
        f"id: {run_id}\n"
        f"name: {config['name']}\n"
        f"comment: |-\n"
        f"  {config['comment']}\n"
    )


def prepare_template_base(config, script_dir):
    """
    Read prepare_template.sh and fill run-wide configuration variables.
    Structure-specific variables (%CHARGE_MOL%, etc.) are left for finalize_prepare_script.
    """
    with open(script_dir / "prepare_template.sh") as f:
        content = f.read()

    # Water model
    water_model = config.get('water_model', 'spce')
    if water_model != 'spce':
        content = content.replace('leaprc.water.spce', f'leaprc.water.{water_model}')

    # Custom strip logic for partial structures
    custom_strip_logic = ""
    for key, val in partial_struct_dict.items():
        mask_str = "|".join(f":{mid + 1}" for mid in val['molid_to_include'])
        custom_strip_logic += (
            f'\n    elif [[ "$CURRENT_JOB_NAME" == "{key}" ]]; then\n'
            f'         echo "Generating partial structure for {key} (Keep {mask_str})"\n'
            f'         cat > strip.in <<EOF\n'
            f'parm box_orig.prmtop\n'
            f'trajin min_orig.rst\n'
            f'strip !({mask_str}) parmout box.prmtop\n'
            f'trajout init.rst restart\n'
            f'run\n'
            f'EOF\n'
            f'         cpptraj -i strip.in > strip.log\n'
        )
    content = content.replace('%CUSTOM_STRIP_LOGIC%', custom_strip_logic)

    # Run-config specific replacements
    replacements = {
        '%MOL_MASK%':     str(config['mol_mask']),
        '%HCORRECTION%':  str(config['DFTBPLUS_hcorrection']),
        '%OPEN_QM_MIN%':  str(config.get('open_qm_min_wat', config['open_qm_min'])).lower(),
        '%MDFTB%':        '.true.' if config['DFTBPLUS_mdftb'] else '.false.',
        '%MDFTB_SCALE%':  '.true.' if config['DFTBPLUS_mdftb_scale'] else '.false.',
        '%SKROOT%':       str(config['DFTBPLUS_skroot']),
        '%QM_THEORY%':    str(config['qm_theory']),
        '%MM_HARDNESS%':  str(config['XTB_mm_hardness']),
        '%CORE_NUMBER%':  str(config['core_number']),
        '%TFERMI%':       str(config.get('tfermi', 0)),
        # Top-level run_configs.yaml (expand_nh_oh_radius, delete_wrong_bonds)
        '%EXPAND_NH_OH_BOND_RADIUS%': str(EXPAND_NH_OH_RADIUS).lower(),
        '%DELETE_WRONG_BONDS%':       str(DELETE_WRONG_BONDS).lower(),
    }
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)

    return content


def finalize_prepare_script(
    base_content,
    mode,
    source_type,
    input_path,
    config,
    struc_name="",
    qm_charge: int | None = None,
):
    """Fill in structure-specific variables into the base template content."""
    content = base_content

    if source_type == "minimize":
        skip_prmtop = "true"
        mol_xyz = "%MOL_XYZ%"
        rst_source = input_path
        prmtop_source = input_path
        rst_file = "box_orig.inpcrd" if mode == "sp_init" else "min.rst"
        prmtop_file = "box.prmtop"
    elif source_type == "qm_minimize_amber":
        # Same Amber skip-tleap path as minimize/, but coordinates from qm_minimize min.rst + prmtop on disk.
        skip_prmtop = "true"
        mol_xyz = "%MOL_XYZ%"
        rst_source = input_path
        prmtop_source = input_path
        rst_file = "min.rst"
        prmtop_file = resolve_prmtop(Path(input_path)).name
    else:
        skip_prmtop = "false"
        mol_xyz = input_path
        rst_source = ""
        prmtop_source = ""
        rst_file = "min.rst"
        prmtop_file = "box.prmtop"

    min_maxcyc = config['min_maxcyc']
    if mode in ("sp_init", "sp_opt") and min_maxcyc > 100:
        min_maxcyc = 1
    if mode == "qm_minimize" and min_maxcyc < 100:
        min_maxcyc = 99999

    if qm_charge is None:
        qm_charge = get_charge_for_structure(
            struc_name, source_type=source_type, input_path=input_path
        )

    fragment_atoms = get_fragment_atom_count(struc_name, source_type, input_path)
    use_dlfind = "false" if fragment_atoms == 1 else "true"

    qm_level = str(config['qm_level'])
    # if qm_level is DFTB3-D3H5 and the structure is a single Zn, set qm_level to DFTB3-D3 because DFTB3-D3H5 is not supported for single Zn
    if qm_level == 'DFTB3-D3H5' and fragment_atoms == 1:
        qm_level = 'DFTB3-D3'

    replacements = {
        '%QM_LEVEL%':                              qm_level,
        '%CHARGE_MOL%':                            str(qm_charge),
        '%USE_DLFIND_FOR_MIN%':                    use_dlfind,
        '%RST_SOURCE_DIR%':                        rst_source,
        '%PRMTOP_SOURCE_DIR%':                     prmtop_source,
        '%RST_SOURCE_FILE%':                       rst_file,
        '%PRMTOP_SOURCE_FILE%':                    prmtop_file,
        '%SKIP_PRMTOP_HARMONICRST_NDX_GENERATION%': skip_prmtop,
        '%MOL_XYZ%':                               mol_xyz,
        '%MIN_MAXCYC%':                            str(min_maxcyc),
    }
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)

    return content


# --- Main ---------------------------------------------------------------------


def _normalize_cli_list(values):
    """
    Normalize list-like CLI arguments to support comma-separated items.
    Example:
      ["run1,run2", "run3"] -> ["run1", "run2", "run3"]
    """
    if values is None:
        return None
    result = []
    for v in values:
        for item in v.split(","):
            item = item.strip()
            if item:
                result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['minimize', 'sp_init', 'sp_opt', 'qm_minimize'],
                        default='minimize', help='Operating mode')
    parser.add_argument('--runs', nargs='+', default=None, help='Specific runs to generate')
    parser.add_argument('--structure', nargs='+', default=None,
                        help='Specific structures to generate')
    args = parser.parse_args()

    mode = args.mode.lower()
    base_dir = _SCRIPT_DIR
    output_mapping = {
        'minimize': 'minimize',
        'sp_init': 'SP_init',
        'sp_opt': 'SP_opt',
        'qm_minimize': 'qm_minimize',
    }
    output_dir = base_dir / output_mapping[mode]

    print(f"Mode: {mode}")
    print(f"Output Directory: {output_dir}")

    runs_arg = _normalize_cli_list(args.runs)

    if runs_arg is None:
        runs_to_generate = (
            [ref_qmrun_to_minimize] if mode == 'qm_minimize'
            else [k for k in run_configs if is_run_id_key(k)]
        )
    else:
        runs_to_generate = runs_arg

    for run_id in runs_to_generate:
        if run_id not in run_configs:
            continue
        config = run_configs[run_id]

        # Ensure ORCA nprocs in template (if present) is consistent with core_number.
        _validate_orca_template_nprocs(config, run_id)

        if mode in ('minimize', 'sp_opt') and config.get('min_maxcyc') == 1:
            print(f"Skipping {run_id} in {mode} mode because min_maxcyc is 1.")
            continue

        use_amber_interface = bool(config.get("use_amber_interface", False))
        orca_template_present = config.get("orca_template") is not None
        molpro_template_present = config.get("molpro_template") is not None
        mrcc_template_present = config.get("mrcc_template") is not None
        source_dict, source_type, keep_mols_map = get_run_sources(mode, run_id, base_dir)
        if not source_dict:
            print(f"No source files found for {run_id} in mode {mode}. Skipping.")
            continue

        print(f"Run {run_id}: {len(set(source_dict.values()))} unique sources ({source_type}).")

        run_dir = output_dir / run_id
        run_dir.mkdir(exist_ok=True, parents=True)

        with open(run_dir / "notes.yaml", 'w') as f:
            f.write(create_notes_yaml(run_id, config))

        gxtb = is_gxtb_run(config)
        dftb_custom = is_dftbplus_custom_run(config)

        orca_only = orca_template_present and not use_amber_interface
        molpro_only = molpro_template_present
        mrcc_only = mrcc_template_present

        # For ORCA-only runs with use_amber_interface=false:
        # skip Amber/MM prepare generation entirely, only create MOL.xyz/orc_job.inp/sbatch_prepare.sh.
        if (
            gxtb
            or dftb_custom
            or orca_only
            or molpro_only
            or mrcc_only
        ):
            # gxtb / dftbplus_custom: no prepare_template; only copy xyz and write sbatch prepare.sh
            base_content = None
        else:
            base_content = prepare_template_base(config, _SCRIPT_DIR)

        count = 0
        run_genbas_path = None
        if mrcc_only:
            try:
                run_genbas_path = generate_mrcc_genbas(run_dir, config)
            except Exception as e:
                print(f"Failed to generate run-level GENBAS for {run_id}: {e}")
                continue
        structures_filter = _normalize_cli_list(args.structure)
        for struc_name, source_path in source_dict.items():
            if structures_filter is not None and struc_name not in structures_filter:
                continue

            struc_dir = run_dir / struc_name
            if struc_dir.exists():
                shutil.rmtree(struc_dir)
            struc_dir.mkdir(parents=True)

            charge_mols = get_charge_molecule_indices(struc_name, keep_mols_map.get(struc_name))
            if charge_mols is not None:
                qm_charge = get_charge_for_keep_molecules(
                    charge_mols, source_type=source_type, input_path=source_path
                )
            else:
                qm_charge = get_charge_for_structure(
                    struc_name, source_type=source_type, input_path=source_path
                )

            if gxtb:
                # Fast path: always copy full xyz to source.xyz; trim is deferred to sbatch_prepare.sh
                src = Path(source_path)
                dst_source_xyz = struc_dir / "source.xyz"
                if src.is_file():
                    shutil.copy2(src, dst_source_xyz)
                elif src.is_dir() and (src / "min.xyz").is_file():
                    shutil.copy2(src / "min.xyz", dst_source_xyz)
                else:
                    print(f"  Warning: source xyz not found for {struc_name}: {src}")
                    continue

                keep = keep_mols_map.get(struc_name)
                if keep is not None:
                    with open(struc_dir / "KEEP_MOLS", "w") as f:
                        f.write(",".join(map(str, keep)) + "\n")
                content = write_gxtb_prepare_script(
                    mode,
                    qm_charge,
                    config,
                    run_id,
                    trim_script_path=(_TOOLS_DIR / "gxtb_trim_xyz.py"),
                )
                sbatch_prepare_path = struc_dir / "sbatch_prepare.sh"
                with open(sbatch_prepare_path, 'w') as f:
                    f.write(content)
                os.chmod(sbatch_prepare_path, os.stat(sbatch_prepare_path).st_mode | stat.S_IEXEC)
            elif dftb_custom:
                # DFTB+ custom: require an xyz-like source for geometry and a user-provided input file.
                src = Path(source_path)
                if src.is_file() and src.suffix.lower() == ".xyz":
                    dst_geom = struc_dir / "geom.xyz"
                    shutil.copy2(src, dst_geom)
                else:
                    print(f"  Warning: DFTB+ custom mode currently requires xyz source; skipping {struc_name}: {src}")
                    continue

                dftb_input_rel = config.get("dftbplus_input")
                if not dftb_input_rel:
                    print(f"  Warning: 'dftbplus_input' not set for run {run_id}; skipping {struc_name}.")
                    continue

                dftb_input_path = Path(dftb_input_rel)
                if not dftb_input_path.is_absolute():
                    dftb_input_path = _SCRIPT_DIR / dftb_input_path

                if not dftb_input_path.exists():
                    print(f"  Warning: DFTB+ input file not found for run {run_id}: {dftb_input_path}")
                    continue

                # Copy to standard DFTB+ input name so we can just call `dftb+`.
                dst_dftb_input = struc_dir / "dftb_in.hsd"
                shutil.copy2(dftb_input_path, dst_dftb_input)

                content = write_dftbplus_custom_prepare_script(
                    mode,
                    config,
                    run_id,
                    dftb_input_filename=dst_dftb_input.name,
                )
                sbatch_prepare_path = struc_dir / "sbatch_prepare.sh"
                with open(sbatch_prepare_path, 'w') as f:
                    f.write(content)
                os.chmod(sbatch_prepare_path, os.stat(sbatch_prepare_path).st_mode | stat.S_IEXEC)
            elif orca_only:
                # ORCA-only init (coordinates from xyz/min.xyz, no Amber/MM prepare.sh).
                try:
                    generate_orca_only_structure(
                        mode=mode,
                        struc_dir=struc_dir,
                        struc_name=struc_name,
                        source_path=source_path,
                        source_type=source_type,
                        config=config,
                        keep_mols=keep_mols_map.get(struc_name),
                        qm_charge=qm_charge,
                    )
                except Exception as e:
                    print(f"  Warning: ORCA-only init failed for {struc_name}: {e}")
                    continue
            elif molpro_only:
                # Molpro-only init (coordinates from xyz/min.xyz, no Amber/MM prepare.sh).
                try:
                    generate_molpro_only_structure(
                        struc_dir=struc_dir,
                        struc_name=struc_name,
                        source_path=source_path,
                        config=config,
                        qm_charge=qm_charge,
                        keep_mols=keep_mols_map.get(struc_name),
                    )
                except Exception as e:
                    print(f"  Warning: Molpro-only init failed for {struc_name}: {e}")
                    continue
            elif mrcc_only:
                # MRCC-only init (coordinates from xyz/min.xyz, no Amber/MM prepare.sh).
                try:
                    generate_mrcc_only_structure(
                        struc_dir=struc_dir,
                        struc_name=struc_name,
                        source_path=source_path,
                        config=config,
                        qm_charge=qm_charge,
                        keep_mols=keep_mols_map.get(struc_name),
                    )
                    if run_genbas_path is None or not run_genbas_path.exists():
                        raise FileNotFoundError("Run-level GENBAS missing.")
                    shutil.copy2(run_genbas_path, struc_dir / "GENBAS")
                except Exception as e:
                    print(f"  Warning: MRCC-only init failed for {struc_name}: {e}")
                    continue
            else:
                content = finalize_prepare_script(
                    base_content,
                    mode,
                    source_type,
                    source_path,
                    config,
                    struc_name,
                    qm_charge=qm_charge,
                )
                prepare_path = struc_dir / "prepare.sh"
                with open(prepare_path, 'w') as f:
                    f.write(content)
                os.chmod(prepare_path, os.stat(prepare_path).st_mode | stat.S_IEXEC)

                if config.get('orca_template') is not None:
                    tpl_path = struc_dir / "orc_job.tpl"
                    with open(tpl_path, 'a') as f:
                        f.write(config['orca_template'])
                    configure_bsse_for_orca(struc_name, source_path, tpl_path)

            count += 1
            if count % 100 == 0:
                print(f"  Generated {count} structures...")

        print(f"Generated {run_id} in {run_dir} ({count} structures)")

    print("\nGeneration complete.")


if __name__ == "__main__":
    main()
