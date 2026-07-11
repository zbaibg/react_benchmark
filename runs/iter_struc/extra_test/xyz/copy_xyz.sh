#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root and software paths
_SCRIPT_DIR_FOR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_CANDIDATE="$_SCRIPT_DIR_FOR_REPO"
while [ "$_REPO_CANDIDATE" != "/" ] && [ ! -f "$_REPO_CANDIDATE/software.yaml" ]; do
  _REPO_CANDIDATE="$(dirname "$_REPO_CANDIDATE")"
done
if [ ! -f "$_REPO_CANDIDATE/software.yaml" ]; then
  echo "ERROR: could not locate repo root (software.yaml)" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$_REPO_CANDIDATE/tools/repo_env.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p xyz_files cpptraj_logs

ensure_cpptraj() {
    if command -v cpptraj >/dev/null 2>&1; then
        return 0
    fi

    if [[ -f "${HOME}/condainit.sh" ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/condainit.sh"
    fi
    conda activate amber
    command -v cpptraj >/dev/null 2>&1
}

write_zn_closest_waters_xyz() {
    local label="$1"
    local prmtop="$2"
    local rst="$3"
    local n_wat="$4"
    local xyz_out="$5"
    local cpptraj_in="cpptraj_logs/${label}.cpptraj.in"
    local cpptraj_log="cpptraj_logs/${label}.cpptraj.log"
    local closest_out="cpptraj_logs/${label}.closest_waters.dat"
    local closest_prmtop="cpptraj_logs/${label}.closest_${n_wat}wat.prmtop"

    test -s "${prmtop}"
    test -s "${rst}"

    cat > "${cpptraj_in}" <<EOF
parm ${prmtop}
trajin ${rst}
autoimage anchor !:WAT origin
solvent :WAT
closest ${n_wat} @1 closestout ${closest_out} parmout ${closest_prmtop}
trajout ${xyz_out} xyz
run
EOF

    cpptraj -i "${cpptraj_in}" > "${cpptraj_log}"
    test -s "${xyz_out}"
}

write_endpoint_waters_xyz() {
    local label="$1"
    local prmtop="$2"
    local rst="$3"
    local zn_n_wat="$4"
    local imh_h_o_wat="$5"
    local deprot_n_h_wat="$6"
    local xyz_out="$7"
    local safe_label="${label//[^A-Za-z0-9_.-]/_}"
    local zn_cpptraj_in="cpptraj_logs/${safe_label}.zn_closest.cpptraj.in"
    local zn_cpptraj_log="cpptraj_logs/${safe_label}.zn_closest.cpptraj.log"
    local zn_closest_out="cpptraj_logs/${safe_label}.zn_closest_waters.dat"
    local zn_closest_prmtop="cpptraj_logs/${safe_label}.zn_closest_${zn_n_wat}wat.prmtop"
    local image_cpptraj_in="cpptraj_logs/${safe_label}.autoimage.cpptraj.in"
    local image_cpptraj_log="cpptraj_logs/${safe_label}.autoimage.cpptraj.log"
    local pdb_out="cpptraj_logs/${safe_label}.autoimaged.pdb"
    local full_xyz_out="cpptraj_logs/${safe_label}.autoimaged_full.xyz"
    local selection_out="cpptraj_logs/${safe_label}.selected_waters.dat"

    test -s "${prmtop}"
    test -s "${rst}"

    if (( zn_n_wat > 0 )); then
        cat > "${zn_cpptraj_in}" <<EOF
parm ${prmtop}
trajin ${rst}
autoimage anchor !:WAT origin
solvent :WAT
closest ${zn_n_wat} @1 closestout ${zn_closest_out} parmout ${zn_closest_prmtop}
run
EOF

        cpptraj -i "${zn_cpptraj_in}" > "${zn_cpptraj_log}"
        test -s "${zn_closest_out}"
    else
        : > "${zn_closest_out}"
    fi

    cat > "${image_cpptraj_in}" <<EOF
parm ${prmtop}
trajin ${rst}
autoimage anchor !:WAT origin
trajout ${pdb_out} pdb
trajout ${full_xyz_out} xyz
run
EOF

    cpptraj -i "${image_cpptraj_in}" > "${image_cpptraj_log}"
    test -s "${pdb_out}"
    test -s "${full_xyz_out}"

    python - "${pdb_out}" "${full_xyz_out}" "${zn_closest_out}" "${zn_n_wat}" "${imh_h_o_wat}" "${deprot_n_h_wat}" "${xyz_out}" "${selection_out}" <<'PY'
import math
import sys

pdb_path, full_xyz_path, zn_closest_path, zn_count_text, imh_h_o_count_text, deprot_n_h_count_text, xyz_path, selection_path = sys.argv[1:]
zn_count = int(zn_count_text)
imh_h_o_count = int(imh_h_o_count_text)
deprot_n_h_count = int(deprot_n_h_count_text)
zn_bound_cutoff = 2.6


def residue_key(atom):
    return (atom["chain"], atom["resid"], atom["icode"], atom["resname"])


def infer_element(atom_name, element):
    element = element.strip()
    if element:
        return element.upper()
    letters = "".join(ch for ch in atom_name.strip() if ch.isalpha())
    if letters.upper().startswith("ZN"):
        return "ZN"
    if not letters:
        return "X"
    return letters[0].upper()


atoms = []
serial_to_atom = {}
with open(pdb_path, "r", encoding="utf-8") as handle:
    for line in handle:
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        atom = {
            "serial": int(line[6:11]),
            "name": line[12:16].strip(),
            "resname": line[17:20].strip(),
            "chain": line[21].strip(),
            "resid": int(line[22:26]),
            "icode": line[26].strip(),
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
            "element": infer_element(line[12:16], line[76:78] if len(line) >= 78 else ""),
        }
        atoms.append(atom)
        serial_to_atom[atom["serial"]] = atom

if not atoms:
    raise SystemExit(f"No atoms found in {pdb_path}")

with open(full_xyz_path, "r", encoding="utf-8") as handle:
    try:
        xyz_atom_count = int(handle.readline().strip())
    except ValueError as exc:
        raise SystemExit(f"Could not read atom count from {full_xyz_path}") from exc
    handle.readline()
    xyz_lines = [handle.readline() for _ in range(xyz_atom_count)]

if xyz_atom_count != len(atoms):
    raise SystemExit(
        f"Atom count mismatch: {pdb_path} has {len(atoms)}, "
        f"{full_xyz_path} has {xyz_atom_count}"
    )

for atom, line in zip(atoms, xyz_lines):
    fields = line.split()
    if len(fields) < 4:
        raise SystemExit(f"Malformed XYZ line in {full_xyz_path}: {line.rstrip()}")
    atom["x"] = float(fields[-3])
    atom["y"] = float(fields[-2])
    atom["z"] = float(fields[-1])

water_keys = []
selection_rows = []


def add_water_key(key):
    if key not in water_keys:
        water_keys.append(key)


def dist2(a, b):
    return (
        (a["x"] - b["x"]) ** 2
        + (a["y"] - b["y"]) ** 2
        + (a["z"] - b["z"]) ** 2
    )


water_oxygens = [
    atom
    for atom in atoms
    if atom["resname"] == "WAT" and atom["name"].upper() == "O"
]
water_hydrogens = [
    atom
    for atom in atoms
    if atom["resname"] == "WAT" and atom["element"] == "H"
]
ligand_nitrogens = [
    atom
    for atom in atoms
    if atom["resname"] in {"IMH", "IM-"} and atom["name"].upper().startswith("N")
]
zn_atoms = [atom for atom in atoms if atom["resname"] == "ZN" or atom["element"] == "ZN"]


def is_zn_bound(n_atom):
    return bool(zn_atoms) and min(math.sqrt(dist2(n_atom, zn_atom)) for zn_atom in zn_atoms) <= zn_bound_cutoff


def nearest_water_oxygens(center, count):
    ranked = sorted(
        ((dist2(center, atom), atom) for atom in water_oxygens),
        key=lambda item: (item[0], item[1]["serial"]),
    )
    out = []
    seen = set()
    for d2_value, atom in ranked:
        key = residue_key(atom)
        if key in seen:
            continue
        seen.add(key)
        out.append((d2_value, atom))
        if len(out) == count:
            break
    return out


def nearest_water_hydrogens(center, count):
    ranked = sorted(
        ((dist2(center, atom), atom) for atom in water_hydrogens),
        key=lambda item: (item[0], item[1]["serial"]),
    )
    out = []
    seen = set()
    for d2_value, atom in ranked:
        key = residue_key(atom)
        if key in seen:
            continue
        seen.add(key)
        out.append((d2_value, atom))
        if len(out) == count:
            break
    return out


with open(zn_closest_path, "r", encoding="utf-8") as handle:
    for line in handle:
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        first_atom_serial = int(fields[3])
        atom = serial_to_atom.get(first_atom_serial)
        if atom is None:
            raise SystemExit(
                f"Closest-water atom serial {first_atom_serial} not found in {pdb_path}"
            )
        key = residue_key(atom)
        add_water_key(key)
        selection_rows.append(
            ("ZN", "-", atom["resid"], atom["serial"], atom["name"], math.nan)
        )

if len(water_oxygens) < imh_h_o_count:
    raise SystemExit(f"Only found {len(water_oxygens)} WAT oxygen atoms, need {imh_h_o_count}")
if len(water_hydrogens) < deprot_n_h_count:
    raise SystemExit(f"Only found {len(water_hydrogens)} WAT hydrogen atoms, need {deprot_n_h_count}")

protonated_n_serials = set()
if imh_h_o_count > 0:
    hn_atoms = [
        atom
        for atom in atoms
        if atom["resname"] == "IMH" and atom["name"].upper() in {"HN1", "H1"}
    ]
    for hn_atom in hn_atoms:
        same_res_ns = [
            atom
            for atom in ligand_nitrogens
            if residue_key(atom) == residue_key(hn_atom)
        ]
        bonded_n = min(same_res_ns, key=lambda atom: dist2(atom, hn_atom)) if same_res_ns else None
        if bonded_n is not None:
            protonated_n_serials.add(bonded_n["serial"])
        if bonded_n is not None and is_zn_bound(bonded_n):
            selection_rows.append(("SKIP_ZN_BOUND_IMH_H", hn_atom["name"], "-", "-", "-", 0.0))
            continue
        for d2_value, atom in nearest_water_oxygens(hn_atom, imh_h_o_count):
            add_water_key(residue_key(atom))
            selection_rows.append(
                (
                    "IMH_H_O",
                    hn_atom["name"],
                    atom["resid"],
                    atom["serial"],
                    atom["name"],
                    math.sqrt(d2_value),
                )
            )

if deprot_n_h_count > 0:
    for n_atom in ligand_nitrogens:
        if n_atom["serial"] in protonated_n_serials:
            continue
        if is_zn_bound(n_atom):
            selection_rows.append(("SKIP_ZN_BOUND_N", n_atom["name"], "-", "-", "-", 0.0))
            continue
        for d2_value, atom in nearest_water_hydrogens(n_atom, deprot_n_h_count):
            add_water_key(residue_key(atom))
            selection_rows.append(
                (
                    "DEPROT_N_H",
                    n_atom["name"],
                    atom["resid"],
                    atom["serial"],
                    atom["name"],
                    math.sqrt(d2_value),
                )
            )

selected_water_key_set = set(water_keys)

selected_atoms = [
    atom
    for atom in atoms
    if atom["resname"] != "WAT" or residue_key(atom) in selected_water_key_set
]

with open(selection_path, "w", encoding="utf-8") as handle:
    handle.write("# source ligand_atom water_resid water_atom_serial water_atom distance_A\n")
    for source, ligand_atom, water_resid, water_serial, water_atom, distance in selection_rows:
        if math.isnan(distance):
            distance_text = "nan"
        else:
            distance_text = f"{distance:.4f}"
        handle.write(
            f"{source} {ligand_atom} {water_resid} {water_serial} {water_atom} {distance_text}\n"
        )
    handle.write(f"# requested_zn_waters {zn_count}\n")
    handle.write(f"# requested_imh_h_o_waters {imh_h_o_count}\n")
    handle.write(f"# requested_deprot_n_h_waters_per_N {deprot_n_h_count}\n")
    handle.write(f"# unique_waters {len(water_keys)}\n")

with open(xyz_path, "w", encoding="utf-8") as handle:
    handle.write(f"{len(selected_atoms)}\n")
    handle.write(
        f"Zn {zn_count} WAT + IMH H...O {imh_h_o_count} WAT + "
        f"deprotonated N...H {deprot_n_h_count} WAT per unbound N "
        f"from {pdb_path}\n"
    )
    for atom in selected_atoms:
        handle.write(
            f"{atom['element']:<2s} {atom['x']:16.8f} {atom['y']:16.8f} {atom['z']:16.8f}\n"
        )
PY

    test -s "${xyz_out}"
}

write_imh_xyz_cluster_waters() {
    local source_xyz="$1"
    local n_h_wat="$2"
    local nh_o_wat="$3"
    local n_h_xyz_out="$4"
    local nh_o_xyz_out="$5"
    local combined_xyz_out="$6"
    local safe_label
    safe_label="$(basename "${source_xyz%.xyz}")"
    safe_label="${safe_label//[^A-Za-z0-9_.-]/_}"
    local selection_out="cpptraj_logs/${safe_label}.selected_waters.dat"

    test -s "${source_xyz}"

    python - "${source_xyz}" "${n_h_wat}" "${nh_o_wat}" \
        "${n_h_xyz_out}" "${nh_o_xyz_out}" "${combined_xyz_out}" "${selection_out}" <<'PY'
import math
import sys

source_xyz, n_h_count_text, nh_o_count_text, n_h_xyz, nh_o_xyz, combined_xyz, selection_path = sys.argv[1:]
n_h_count = int(n_h_count_text)
nh_o_count = int(nh_o_count_text)
imh_atom_count = 9


def dist2(a, b):
    return (
        (a["x"] - b["x"]) ** 2
        + (a["y"] - b["y"]) ** 2
        + (a["z"] - b["z"]) ** 2
    )


def dist(a, b):
    return math.sqrt(dist2(a, b))


with open(source_xyz, "r", encoding="utf-8") as handle:
    try:
        atom_count = int(handle.readline().strip())
    except ValueError as exc:
        raise SystemExit(f"Could not read atom count from {source_xyz}") from exc
    comment = handle.readline().strip()
    atoms = []
    for serial in range(1, atom_count + 1):
        line = handle.readline()
        fields = line.split()
        if len(fields) < 4:
            raise SystemExit(f"Malformed XYZ line in {source_xyz}: {line.rstrip()}")
        atoms.append(
            {
                "serial": serial,
                "element": fields[0],
                "x": float(fields[1]),
                "y": float(fields[2]),
                "z": float(fields[3]),
            }
        )

if len(atoms) != atom_count:
    raise SystemExit(f"Expected {atom_count} atoms in {source_xyz}, found {len(atoms)}")
if atom_count <= imh_atom_count or (atom_count - imh_atom_count) % 3 != 0:
    raise SystemExit(
        f"{source_xyz} should contain 9 IMH atoms followed by O-H-H water triplets"
    )

imh_atoms = atoms[:imh_atom_count]
water_triplets = [
    atoms[start : start + 3]
    for start in range(imh_atom_count, atom_count, 3)
]

for water_id, water in enumerate(water_triplets, start=1):
    elements = [atom["element"].upper() for atom in water]
    if elements != ["O", "H", "H"]:
        raise SystemExit(
            f"Water {water_id} in {source_xyz} is not ordered as O H H: {elements}"
        )

imh_nitrogens = [atom for atom in imh_atoms if atom["element"].upper() == "N"]
imh_hydrogens = [atom for atom in imh_atoms if atom["element"].upper() == "H"]
if len(imh_nitrogens) != 2:
    raise SystemExit(f"Expected two IMH nitrogens in the first 9 atoms of {source_xyz}")
if not imh_hydrogens:
    raise SystemExit(f"Expected IMH hydrogens in the first 9 atoms of {source_xyz}")

protonated_n, hn_atom, nh_distance = min(
    (
        (n_atom, h_atom, dist(n_atom, h_atom))
        for n_atom in imh_nitrogens
        for h_atom in imh_hydrogens
    ),
    key=lambda item: (item[2], item[0]["serial"], item[1]["serial"]),
)
if nh_distance > 1.3:
    raise SystemExit(
        f"Closest IMH N-H distance in {source_xyz} is {nh_distance:.3f} A; "
        "could not identify protonated N-H"
    )
deprotonated_n = next(
    atom for atom in imh_nitrogens if atom["serial"] != protonated_n["serial"]
)


def nearest_unique_waters_to_atom(center, atom_element, count):
    ranked = []
    for water_id, water in enumerate(water_triplets, start=1):
        for atom in water:
            if atom["element"].upper() == atom_element:
                ranked.append((dist2(center, atom), water_id, atom))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]["serial"]))

    out = []
    seen = set()
    for d2_value, water_id, atom in ranked:
        if water_id in seen:
            continue
        seen.add(water_id)
        out.append((water_id, atom, math.sqrt(d2_value)))
        if len(out) == count:
            break
    if len(out) != count:
        raise SystemExit(f"Only found {len(out)} waters, need {count}")
    return out


n_h_waters = nearest_unique_waters_to_atom(deprotonated_n, "H", n_h_count)
nh_o_waters = nearest_unique_waters_to_atom(hn_atom, "O", nh_o_count)


def selected_atoms(water_ids):
    water_id_set = set(water_ids)
    selected = list(imh_atoms)
    for water_id, water in enumerate(water_triplets, start=1):
        if water_id in water_id_set:
            selected.extend(water)
    return selected


def write_xyz(path, selected, label):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{len(selected)}\n")
        handle.write(f"{label} from {source_xyz} ({comment})\n")
        for atom in selected:
            handle.write(
                f"{atom['element']:<2s} {atom['x']:16.8f} "
                f"{atom['y']:16.8f} {atom['z']:16.8f}\n"
            )


n_h_water_ids = [water_id for water_id, _atom, _distance in n_h_waters]
nh_o_water_ids = [water_id for water_id, _atom, _distance in nh_o_waters]
combined_water_ids = sorted(set(n_h_water_ids + nh_o_water_ids))

write_xyz(
    n_h_xyz,
    selected_atoms(n_h_water_ids),
    f"IMH + deprotonated N...H {n_h_count} WAT",
)
write_xyz(
    nh_o_xyz,
    selected_atoms(nh_o_water_ids),
    f"IMH + protonated N-H...O {nh_o_count} WAT",
)
write_xyz(
    combined_xyz,
    selected_atoms(combined_water_ids),
    f"IMH + selected N...H/N-H...O {len(combined_water_ids)} WAT",
)

with open(selection_path, "w", encoding="utf-8") as handle:
    handle.write("# source imh_atom_serial water_id water_atom_serial water_atom distance_A\n")
    handle.write(
        f"# protonated_N {protonated_n['serial']} HN {hn_atom['serial']} "
        f"distance_A {nh_distance:.4f}\n"
    )
    handle.write(f"# deprotonated_N {deprotonated_n['serial']}\n")
    for water_id, atom, distance_value in n_h_waters:
        handle.write(
            f"N_H {deprotonated_n['serial']} {water_id} "
            f"{atom['serial']} {atom['element']} {distance_value:.4f}\n"
        )
    for water_id, atom, distance_value in nh_o_waters:
        handle.write(
            f"NH_O {hn_atom['serial']} {water_id} "
            f"{atom['serial']} {atom['element']} {distance_value:.4f}\n"
        )
PY

    test -s "${n_h_xyz_out}"
    test -s "${nh_o_xyz_out}"
    test -s "${combined_xyz_out}"
}

ensure_cpptraj

cp "$REPO_ROOT/structures/monomers/IM-.xyz" ./xyz_files/Im-_monomer.xyz
cp "$REPO_ROOT/structures/monomers/ImH.xyz" ./xyz_files/ImH_monomer.xyz
cp "$REPO_ROOT/structures/monomers/Wat.xyz" ./xyz_files/Wat_monomer.xyz
cp "$REPO_ROOT/structures/monomers/Zn.xyz" ./xyz_files/Zn_monomer.xyz
cat > ./xyz_files/H_monomer.xyz <<EOF
1
H+
H     0.000000      0.000000      0.000000

EOF
write_zn_closest_waters_xyz \
    1Zn_1Im-_5Wat \
    "$REPO_ROOT/structures/Net_React/run28/routes/route012_IM-_IM-1__to__bare_Zn/firstshell_leg1/1.000/box_H3.prmtop \
    "$REPO_ROOT/structures/Net_React/run28/routes/route012_IM-_IM-1__to__bare_Zn/firstshell_leg1/1.000/ti_sample_qmmm.rst \
    5 \
    ./xyz_files/1Zn_1Im-_5Wat.xyz

write_endpoint_waters_xyz \
    1Zn_5Wat_1Im-_1Wat \
    "$REPO_ROOT/structures/Net_React/run28/routes/route012_IM-_IM-1__to__bare_Zn/firstshell_leg1/1.000/box_H3.prmtop \
    "$REPO_ROOT/structures/Net_React/run28/routes/route012_IM-_IM-1__to__bare_Zn/firstshell_leg1/1.000/ti_sample_qmmm.rst \
    5 \
    0 \
    1 \
    ./xyz_files/1Zn_5Wat_1Im-_1Wat.xyz

write_zn_closest_waters_xyz \
    1Zn_1ImH_5Wat \
    "$REPO_ROOT/structures/Net_React/run28/routes/route006_IMH_IMH1__to__bare_Zn/firstshell_leg1/1.000/box_H3.prmtop \
    "$REPO_ROOT/structures/Net_React/run28/routes/route006_IMH_IMH1__to__bare_Zn/firstshell_leg1/1.000/ti_sample_qmmm.rst \
    5 \
    ./xyz_files/1Zn_1ImH_5Wat.xyz

write_endpoint_waters_xyz \
    1Zn_5Wat_1ImH_1Wat \
    "$REPO_ROOT/structures/Net_React/run28/routes/route006_IMH_IMH1__to__bare_Zn/firstshell_leg1/1.000/box_H3.prmtop \
    "$REPO_ROOT/structures/Net_React/run28/routes/route006_IMH_IMH1__to__bare_Zn/firstshell_leg1/1.000/ti_sample_qmmm.rst \
    5 \
    1 \
    1 \
    ./xyz_files/1Zn_5Wat_1ImH_1Wat.xyz

write_endpoint_waters_xyz \
    1Im-_2Wat \
    "$REPO_ROOT/structures/Net_React/run28/routes/route902_IM-_free_ligand_solvation/leg1/1.000/box_H3.prmtop \
    "$REPO_ROOT/structures/Net_React/run28/routes/route902_IM-_free_ligand_solvation/leg1/1.000/ti_sample_qmmm.rst \
    0 \
    0 \
    1 \
    ./xyz_files/1Im-_2Wat.xyz

write_endpoint_waters_xyz \
    1ImH_2Wat \
    "$REPO_ROOT/structures/Net_React/run28/routes/route903_IMH_free_ligand_solvation/leg1/1.000/box_H3.prmtop \
    "$REPO_ROOT/structures/Net_React/run28/routes/route903_IMH_free_ligand_solvation/leg1/1.000/ti_sample_qmmm.rst \
    0 \
    1 \
    1 \
    ./xyz_files/1ImH_2Wat.xyz

write_zn_closest_waters_xyz \
    1Zn_0ImH_6Wat \
    "$REPO_ROOT/structures/Net_React/run28/routes/route006_IMH_IMH1__to__bare_Zn/firstshell_leg4/1.000/box_target_stripped.prmtop \
    "$REPO_ROOT/structures/Net_React/run28/routes/route006_IMH_IMH1__to__bare_Zn/firstshell_leg4/1.000/ti_sample_state1.rst \
    6 \
    ./xyz_files/1Zn_0ImH_6Wat.xyz
write_imh_xyz_cluster_waters \
    "$REPO_ROOT/structures/metaD/IMZW64.xyz" \
    2 \
    1 \
    ./xyz_files/IMZW64_1ImH_2Wat_N_H.xyz \
    ./xyz_files/IMZW64_1ImH_1Wat_NH_O.xyz \
    ./xyz_files/IMZW64_1ImH_3Wat_N_H_NH_O.xyz
cp "$REPO_ROOT/structures/metaD/2_Cs.xyz" ./xyz_files/2Wat.xyz
cp ./manual_adjust/* ./xyz_files/
