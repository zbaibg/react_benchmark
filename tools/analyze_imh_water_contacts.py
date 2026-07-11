#!/usr/bin/env python3
import argparse
from collections import Counter
import csv
import math
import re
import sys
from pathlib import Path


VALID_ELEMENTS = {"C", "H", "N", "O"}
DEFAULT_CUTOFF_A = 3.5
DEFAULT_MIN_ANGLE_DEG = 90.0
DEFAULT_IMIDAZOLE_ATOMS_COUNT = 9
DEFAULT_WATER_ATOMS_COUNT = 3
DEFAULT_N3_POSITION = 8
DEFAULT_NH_H_POSITION = 6
DEFAULT_NH_N_POSITION = 9
DEFAULT_NO_HBOND_CLOSE_CUTOFF_A = 4.0
FAR_INTERACTION_LABEL = "far"
CONTACT_LABEL_ALIASES = {
    "NH_O": "NH...O",
    "OH_N3": "OH...N3",
    "CH_O_C1": "CH...O(C1)",
    "CH_O_C2": "CH...O(C2)",
    "CH_O_C3": "CH...O(C3)",
    "OH_PI_C1": "OH...pi(C1)",
    "OH_PI_C2": "OH...pi(C2)",
    "OH_PI_C3": "OH...pi(C3)",
    "OH_PI_N9": "OH...pi(N9)",
    "OH_pi_C1": "OH...pi(C1)",
    "OH_pi_C2": "OH...pi(C2)",
    "OH_pi_C3": "OH...pi(C3)",
    "OH_pi_N9": "OH...pi(N9)",
}


def natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def read_xyz(path):
    with path.open(encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if len(lines) < 2:
        raise ValueError("file is too short for XYZ format")

    try:
        natoms = int(lines[0].split()[0])
    except ValueError as exc:
        raise ValueError("first line is not an atom count") from exc

    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"expected {natoms} atom lines, found {len(atom_lines)}")

    atoms = []
    for atom_index, line in enumerate(atom_lines, start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"bad atom line {atom_index}: {line!r}")
        try:
            xyz = tuple(float(value) for value in parts[1:4])
        except ValueError as exc:
            raise ValueError(f"bad coordinates at atom {atom_index}: {line!r}") from exc
        if parts[0] not in VALID_ELEMENTS:
            if parts[0] in {"0", "El0"} and all(abs(value) < 1e-12 for value in xyz):
                raise ValueError(
                    "placeholder zero-coordinate block from source "
                    f"(atomic number 0 at atom {atom_index})"
                )
            raise ValueError(
                f"invalid element {parts[0]!r} at atom {atom_index}; "
                f"expected one of {', '.join(sorted(VALID_ELEMENTS))}"
            )
        atoms.append(
            {
                "index": atom_index,
                "element": parts[0],
                "xyz": xyz,
            }
        )

    return lines[1], atoms


def water_atom_label(position_in_water):
    if position_in_water == 1:
        return "O"
    return f"H{position_in_water - 1}"


def imidazole_atom_label(position_in_imidazole, atom, n3_position):
    if position_in_imidazole == n3_position:
        return "N3"
    return f"{atom['element']}{position_in_imidazole}"


def parse_index_list(value, option_name):
    if value is None:
        return None
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            index = int(item)
        except ValueError as exc:
            raise ValueError(f"{option_name} must contain comma-separated integers") from exc
        if index <= 0:
            raise ValueError(f"{option_name} indices must be positive")
        indices.append(index)
    return indices


def canonical_interaction_label(label):
    return CONTACT_LABEL_ALIASES.get(label, label)


def parse_contact_cutoff(value, label):
    if isinstance(value, dict):
        distance = value.get("distance_A", value.get("distance", value.get("cutoff")))
        min_angle = value.get(
            "min_angle_deg",
            value.get("min_angle", value.get("angle_deg", value.get("angle"))),
        )
    else:
        try:
            distance, min_angle = value
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"class cutoff for {label!r} must be a (distance_A, min_angle_deg) pair"
            ) from exc

    if distance is None or min_angle is None:
        raise ValueError(
            f"class cutoff for {label!r} must define distance_A and min_angle_deg"
        )
    return float(distance), float(min_angle)


def normalize_class_cutoffs(class_cutoffs):
    if not class_cutoffs:
        return {}
    return {
        canonical_interaction_label(label): parse_contact_cutoff(value, label)
        for label, value in class_cutoffs.items()
    }


def broad_distance_cutoff(default_cutoff, class_cutoffs):
    if not class_cutoffs:
        return default_cutoff
    return max([default_cutoff] + [values[0] for values in class_cutoffs.values()])


def broad_min_angle(default_min_angle, class_cutoffs):
    if not class_cutoffs:
        return default_min_angle
    return min([default_min_angle] + [values[1] for values in class_cutoffs.values()])


def contact_cutoff_for_label(label, default_cutoff, default_min_angle, class_cutoffs):
    return class_cutoffs.get(
        canonical_interaction_label(label),
        (default_cutoff, default_min_angle),
    )


def candidate_passes_class_cutoff(
    record,
    default_cutoff,
    default_min_angle,
    class_cutoffs,
):
    label = interaction_label(record)
    cutoff, min_angle = contact_cutoff_for_label(
        label,
        default_cutoff,
        default_min_angle,
        class_cutoffs,
    )
    if record["distance_A"] >= cutoff:
        return False
    angle = record.get("angle_deg")
    if angle is None or angle <= min_angle:
        return False
    return True


def atom_by_position(atoms, position):
    if position <= 0 or position > len(atoms):
        return None
    return atoms[position - 1]


def angle_degrees(atom1, vertex_atom, atom3):
    v1 = tuple(a - b for a, b in zip(atom1["xyz"], vertex_atom["xyz"]))
    v2 = tuple(a - b for a, b in zip(atom3["xyz"], vertex_atom["xyz"]))
    norm1 = math.sqrt(sum(value * value for value in v1))
    norm2 = math.sqrt(sum(value * value for value in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return None
    cosine = sum(a * b for a, b in zip(v1, v2)) / (norm1 * norm2)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def water_positions_by_element(water_atoms, element):
    return [
        position
        for position, atom in enumerate(water_atoms, start=1)
        if atom["element"] == element
    ]


def imidazole_positions_by_element(imidazole_atoms, element):
    return [
        position
        for position, atom in enumerate(imidazole_atoms, start=1)
        if atom["element"] == element
    ]


def closest_imidazole_position(imidazole_atoms, origin_atom, element):
    best_position = None
    best_distance = None
    for position, atom in enumerate(imidazole_atoms, start=1):
        if atom["element"] != element:
            continue
        distance = math.dist(origin_atom["xyz"], atom["xyz"])
        if best_distance is None or distance < best_distance:
            best_position = position
            best_distance = distance
    return best_position


def imidazole_label_map(imidazole_atoms, n3_position):
    return {
        position: imidazole_atom_label(position, atom, n3_position)
        for position, atom in enumerate(imidazole_atoms, start=1)
    }


def imidazole_atom_notes(imidazole_atoms, n3_position):
    labels = imidazole_label_map(imidazole_atoms, n3_position)
    notes = []

    for position, atom in enumerate(imidazole_atoms, start=1):
        if atom["element"] == "H":
            heavy_position = closest_imidazole_position(
                imidazole_atoms,
                atom,
                "C",
            )
            closest_n_position = closest_imidazole_position(
                imidazole_atoms,
                atom,
                "N",
            )
            nearest_heavy_position = min(
                [
                    candidate
                    for candidate in (heavy_position, closest_n_position)
                    if candidate is not None
                ],
                key=lambda candidate: math.dist(
                    atom["xyz"],
                    atom_by_position(imidazole_atoms, candidate)["xyz"],
                ),
                default=None,
            )
            if nearest_heavy_position is not None:
                notes.append(
                    f"{labels[position]}: H attached to {labels[nearest_heavy_position]}"
                )

    n_labels = [
        labels[position]
        for position, atom in enumerate(imidazole_atoms, start=1)
        if atom["element"] == "N"
    ]
    c_labels = [
        labels[position]
        for position, atom in enumerate(imidazole_atoms, start=1)
        if atom["element"] == "C"
    ]

    return labels, c_labels, n_labels, notes


def build_contact_candidate(
    interaction_type,
    water_position,
    water_atom,
    imidazole_position,
    imidazole_atom,
    n3_position,
):
    return {
        "interaction_type": interaction_type,
        "distance_A": math.dist(water_atom["xyz"], imidazole_atom["xyz"]),
        "water_atom_label": water_atom_label(water_position),
        "water_atom_index": water_atom["index"],
        "water_atom_element": water_atom["element"],
        "imidazole_atom_label": imidazole_atom_label(
            imidazole_position,
            imidazole_atom,
            n3_position,
        ),
        "imidazole_atom_index": imidazole_atom["index"],
        "imidazole_atom_element": imidazole_atom["element"],
    }


def collect_oh_acceptor_candidates(
    interaction_type,
    water_atoms,
    imidazole_atoms,
    acceptor_positions,
    n3_position,
    cutoff,
    min_angle,
):
    candidates = []
    water_h_positions = water_positions_by_element(water_atoms, "H")
    water_o_positions = water_positions_by_element(water_atoms, "O")
    for water_position in water_h_positions:
        water_h = atom_by_position(water_atoms, water_position)
        if water_h is None:
            continue
        oxygen_position = min(
            water_o_positions,
            key=lambda position: math.dist(
                atom_by_position(water_atoms, position)["xyz"],
                water_h["xyz"],
            ),
            default=None,
        )
        if oxygen_position is None:
            continue
        water_o = atom_by_position(water_atoms, oxygen_position)
        for imidazole_position in acceptor_positions:
            imidazole_atom = atom_by_position(imidazole_atoms, imidazole_position)
            if imidazole_atom is None:
                continue
            if math.dist(water_h["xyz"], imidazole_atom["xyz"]) >= cutoff:
                continue
            angle = angle_degrees(water_o, water_h, imidazole_atom)
            if angle is None or angle <= min_angle:
                continue
            candidate = build_contact_candidate(
                interaction_type,
                water_position,
                water_h,
                imidazole_position,
                imidazole_atom,
                n3_position,
            )
            candidate["angle_deg"] = angle
            candidate["angle_atoms"] = (
                f"{water_atom_label(oxygen_position)}-"
                f"{water_atom_label(water_position)}-"
                f"{candidate['imidazole_atom_label']}"
            )
            candidates.append(candidate)
    return candidates


def collect_oh_pi_candidates(
    water_atoms,
    imidazole_atoms,
    ring_heavy_positions,
    n3_position,
    cutoff,
    min_angle,
):
    candidates = []
    water_h_positions = water_positions_by_element(water_atoms, "H")
    water_o_positions = water_positions_by_element(water_atoms, "O")
    for water_position in water_h_positions:
        water_h = atom_by_position(water_atoms, water_position)
        if water_h is None:
            continue
        oxygen_position = min(
            water_o_positions,
            key=lambda position: math.dist(
                atom_by_position(water_atoms, position)["xyz"],
                water_h["xyz"],
            ),
            default=None,
        )
        if oxygen_position is None:
            continue
        water_o = atom_by_position(water_atoms, oxygen_position)
        for imidazole_position in ring_heavy_positions:
            imidazole_atom = atom_by_position(imidazole_atoms, imidazole_position)
            if imidazole_atom is None:
                continue
            if math.dist(water_h["xyz"], imidazole_atom["xyz"]) >= cutoff:
                continue
            angle = angle_degrees(water_o, water_h, imidazole_atom)
            if angle is None or angle <= min_angle:
                continue
            candidate = build_contact_candidate(
                "OH...pi",
                water_position,
                water_h,
                imidazole_position,
                imidazole_atom,
                n3_position,
            )
            candidate["angle_deg"] = angle
            candidate["angle_atoms"] = (
                f"{water_atom_label(oxygen_position)}-"
                f"{water_atom_label(water_position)}-"
                f"{candidate['imidazole_atom_label']}"
            )
            candidates.append(candidate)
    return candidates


def collect_xh_o_candidates(
    interaction_type,
    water_atoms,
    imidazole_atoms,
    h_positions,
    n3_position,
    cutoff,
    min_angle,
    donor_position=None,
    donor_element=None,
):
    candidates = []
    water_o_positions = water_positions_by_element(water_atoms, "O")
    for h_position in h_positions:
        h_atom = atom_by_position(imidazole_atoms, h_position)
        if h_atom is None:
            continue
        actual_donor_position = donor_position
        if actual_donor_position is None and donor_element is not None:
            actual_donor_position = closest_imidazole_position(
                imidazole_atoms,
                h_atom,
                donor_element,
            )
        donor_atom = atom_by_position(imidazole_atoms, actual_donor_position)
        if donor_atom is None:
            continue
        donor_label = imidazole_atom_label(
            actual_donor_position,
            donor_atom,
            n3_position,
        )
        for water_position in water_o_positions:
            water_o = atom_by_position(water_atoms, water_position)
            if water_o is None:
                continue
            if math.dist(h_atom["xyz"], water_o["xyz"]) >= cutoff:
                continue
            angle = angle_degrees(donor_atom, h_atom, water_o)
            if angle is None or angle <= min_angle:
                continue
            candidate = build_contact_candidate(
                interaction_type,
                water_position,
                water_o,
                h_position,
                h_atom,
                n3_position,
            )
            candidate["angle_deg"] = angle
            candidate["angle_atoms"] = (
                f"{donor_label}-{candidate['imidazole_atom_label']}-"
                f"{candidate['water_atom_label']}"
            )
            candidate["donor_atom_label"] = donor_label
            candidates.append(candidate)
    return candidates


def closest_atom_contact(
    water_atoms,
    imidazole_atoms,
    n3_position,
    interaction_type="closest",
):
    best = None
    for water_position, water_atom in enumerate(water_atoms, start=1):
        for imidazole_position, imidazole_atom in enumerate(imidazole_atoms, start=1):
            candidate = build_contact_candidate(
                interaction_type,
                water_position,
                water_atom,
                imidazole_position,
                imidazole_atom,
                n3_position,
            )
            if best is None or candidate["distance_A"] < best["distance_A"]:
                best = candidate
    if best is None:
        return None

    best["angle_atoms"] = ""
    best["angle_deg"] = None
    return best


def closest_non_hbond_contact(water_atoms, imidazole_atoms, cutoff, n3_position):
    best = closest_atom_contact(
        water_atoms,
        imidazole_atoms,
        n3_position,
        interaction_type="noHbond-close",
    )
    if best is None or best["distance_A"] >= cutoff:
        return None
    return best


def far_contact(water_atoms, imidazole_atoms, n3_position):
    record = closest_atom_contact(
        water_atoms,
        imidazole_atoms,
        n3_position,
        interaction_type=FAR_INTERACTION_LABEL,
    )
    if record is None:
        return None

    record["interaction_label"] = FAR_INTERACTION_LABEL
    record["interaction_detail"] = interaction_detail(record)
    record["matched_interactions"] = FAR_INTERACTION_LABEL
    record["matched_interaction_details"] = interaction_summary(record)
    return record


def interaction_detail(record):
    if record["interaction_type"] in {"NH...O", "CH...O"}:
        return f"{record['imidazole_atom_label']}...{record['water_atom_label']}"
    return f"{record['water_atom_label']}...{record['imidazole_atom_label']}"


def interaction_label(record):
    if record["interaction_type"] == FAR_INTERACTION_LABEL:
        return FAR_INTERACTION_LABEL
    if record["interaction_type"] == "OH...pi":
        return f"OH...pi({record['imidazole_atom_label']})"
    if record["interaction_type"] == "CH...O":
        donor_label = record.get("donor_atom_label")
        if donor_label is None:
            donor_label = record.get("angle_atoms", "").split("-", 1)[0]
        return f"CH...O({donor_label})"
    if record["interaction_type"] == "noHbond-close":
        water_element = record["water_atom_element"]
        imidazole_element = record["imidazole_atom_element"]
        imidazole_label = record["imidazole_atom_label"]
        if water_element == "H" and imidazole_element == "H":
            return f"noHbond-close(HH-{imidazole_label})"
        if water_element == "O" and imidazole_element == "H":
            return f"noHbond-close(OH-{imidazole_label})"
        if water_element == "H" and imidazole_element != "H":
            return f"noHbond-close(H-{imidazole_label})"
        return f"noHbond-close({water_element}-{imidazole_label})"
    return record["interaction_type"]


def interaction_count_sort_key(label):
    base_order = {
        "OH...N3": 0,
        "NH...O": 1,
        "CH...O(C1)": 2,
        "CH...O(C2)": 3,
        "CH...O(C3)": 4,
    }
    if label in base_order:
        return (base_order[label], [])
    if label.startswith("OH...pi("):
        natural = [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", label)
        ]
        return (5, natural)
    if label.startswith("noHbond-close("):
        natural = [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", label)
        ]
        return (99, natural)
    if label == FAR_INTERACTION_LABEL:
        return (100, [])
    natural = [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", label)
    ]
    return (6, natural)


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def format_stat(value):
    if value is None:
        return "NA"
    return f"{value:.3f}"


def interaction_stats_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["interaction_label"], []).append(row)

    labels = {"OH...N3", "NH...O", "CH...O(C1)", "CH...O(C2)", "CH...O(C3)"}
    labels.update(grouped)

    stats = []
    for label in sorted(labels, key=interaction_count_sort_key):
        label_rows = grouped.get(label, [])
        distances = [row["distance_A"] for row in label_rows]
        angles = [
            row["angle_deg"]
            for row in label_rows
            if row["angle_deg"] is not None
        ]
        distance_mean, distance_std = mean_std(distances)
        angle_mean, angle_std = mean_std(angles)
        stats.append(
            {
                "label": label,
                "count": len(label_rows),
                "distance_mean": distance_mean,
                "distance_std": distance_std,
                "angle_mean": angle_mean,
                "angle_std": angle_std,
            }
        )
    return stats


def interaction_sort_key(record):
    return record["distance_A"]


def interaction_summary(record):
    if record["interaction_type"] == FAR_INTERACTION_LABEL:
        return (
            f"{FAR_INTERACTION_LABEL}[{record['interaction_detail']} "
            f"d={record['distance_A']:.3f}]"
        )
    if record["interaction_type"] == "noHbond-close":
        return (
            f"noHbond-close[{record['interaction_detail']} "
            f"d={record['distance_A']:.3f}]"
        )
    return (
        f"{interaction_label(record)}[{record['angle_atoms']} "
        f"H...A={record['distance_A']:.3f} angle={record['angle_deg']:.1f}]"
    )


def classify_water_contacts(
    water_atoms,
    imidazole_atoms,
    cutoff,
    n3_position,
    nh_h_position,
    nh_n_position,
    ch_h_positions,
    ring_heavy_positions,
    min_angle,
    include_far=False,
    class_cutoffs=None,
    no_hbond_close_cutoff=None,
):
    class_cutoffs = normalize_class_cutoffs(class_cutoffs)
    candidate_cutoff = broad_distance_cutoff(cutoff, class_cutoffs)
    candidate_min_angle = broad_min_angle(min_angle, class_cutoffs)
    if no_hbond_close_cutoff is None:
        no_hbond_close_cutoff = candidate_cutoff

    candidates = []
    candidates.extend(
        collect_oh_acceptor_candidates(
            "OH...N3",
            water_atoms,
            imidazole_atoms,
            [n3_position],
            n3_position,
            candidate_cutoff,
            candidate_min_angle,
        )
    )
    if nh_h_position:
        candidates.extend(
            collect_xh_o_candidates(
                "NH...O",
                water_atoms,
                imidazole_atoms,
                [nh_h_position],
                n3_position,
                candidate_cutoff,
                candidate_min_angle,
                donor_position=nh_n_position,
            )
        )
    candidates.extend(
        collect_xh_o_candidates(
            "CH...O",
            water_atoms,
            imidazole_atoms,
            ch_h_positions,
            n3_position,
            candidate_cutoff,
            candidate_min_angle,
            donor_element="C",
        )
    )
    candidates.extend(
        collect_oh_pi_candidates(
            water_atoms,
            imidazole_atoms,
            ring_heavy_positions,
            n3_position,
            candidate_cutoff,
            candidate_min_angle,
        )
    )

    matched = [
        candidate
        for candidate in candidates
        if candidate_passes_class_cutoff(
            candidate,
            cutoff,
            min_angle,
            class_cutoffs,
        )
    ]
    if not matched:
        close_contact = closest_non_hbond_contact(
            water_atoms,
            imidazole_atoms,
            no_hbond_close_cutoff,
            n3_position,
        )
        if close_contact is None:
            if include_far:
                return far_contact(water_atoms, imidazole_atoms, n3_position)
            return None
        close_contact["interaction_label"] = interaction_label(close_contact)
        close_contact["interaction_detail"] = interaction_detail(close_contact)
        close_contact["matched_interactions"] = close_contact["interaction_label"]
        close_contact["matched_interaction_details"] = interaction_summary(close_contact)
        return close_contact

    matched.sort(key=interaction_sort_key)
    selected = dict(matched[0])
    selected["interaction_label"] = interaction_label(selected)
    selected["matched_interactions"] = ";".join(
        interaction_label(record) for record in matched
    )
    selected["matched_interaction_details"] = ";".join(
        interaction_summary(record) for record in matched
    )
    selected["interaction_detail"] = interaction_detail(selected)
    return selected


def analyze_xyz(
    path,
    imidazole_atoms_count,
    water_atoms_count,
    cutoff,
    n3_position,
    nh_h_position,
    nh_n_position,
    ch_h_positions,
    ring_heavy_positions,
    min_angle,
    include_far=False,
    class_cutoffs=None,
    no_hbond_close_cutoff=None,
):
    path = Path(path)
    title, atoms = read_xyz(path)
    if len(atoms) < imidazole_atoms_count:
        raise ValueError(
            f"contains {len(atoms)} atoms, fewer than imidazole atom count {imidazole_atoms_count}"
        )

    imidazole_atoms = atoms[:imidazole_atoms_count]
    if atom_by_position(imidazole_atoms, n3_position) is None:
        raise ValueError(f"N3 index {n3_position} is outside the imidazole atom block")

    if nh_h_position and atom_by_position(imidazole_atoms, nh_h_position) is None:
        raise ValueError(
            f"N-H hydrogen index {nh_h_position} is outside the imidazole atom block"
        )
    if nh_h_position and atom_by_position(imidazole_atoms, nh_n_position) is None:
        raise ValueError(
            f"N-H nitrogen index {nh_n_position} is outside the imidazole atom block"
        )

    if ch_h_positions is None:
        ch_h_positions = [
            position
            for position in imidazole_positions_by_element(imidazole_atoms, "H")
            if position != nh_h_position
        ]
    if ring_heavy_positions is None:
        ring_heavy_positions = [
            position
            for position, atom in enumerate(imidazole_atoms, start=1)
            if atom["element"] != "H" and position != n3_position
        ]
    else:
        ring_heavy_positions = [
            position for position in ring_heavy_positions if position != n3_position
        ]

    remaining_atoms = atoms[imidazole_atoms_count:]
    if len(remaining_atoms) % water_atoms_count != 0:
        raise ValueError(
            f"{len(remaining_atoms)} atoms remain after imidazole; "
            f"not divisible by water atom count {water_atoms_count}"
        )

    rows = []
    n_waters = len(remaining_atoms) // water_atoms_count
    for water_index in range(n_waters):
        start = water_index * water_atoms_count
        water_atoms = remaining_atoms[start : start + water_atoms_count]
        contact = classify_water_contacts(
            water_atoms,
            imidazole_atoms,
            cutoff,
            n3_position,
            nh_h_position,
            nh_n_position,
            ch_h_positions,
            ring_heavy_positions,
            min_angle,
            include_far=include_far,
            class_cutoffs=class_cutoffs,
            no_hbond_close_cutoff=no_hbond_close_cutoff,
        )
        if contact:
            rows.append(
                {
                    "file": path.name,
                    "title": title,
                    "n_atoms": len(atoms),
                    "n_waters": n_waters,
                    "water_index": water_index + 1,
                    **contact,
                }
            )
    return rows


def classify_imh_water_xyz(
    path,
    *,
    imidazole_atoms_count=DEFAULT_IMIDAZOLE_ATOMS_COUNT,
    water_atoms_count=DEFAULT_WATER_ATOMS_COUNT,
    cutoff=DEFAULT_CUTOFF_A,
    min_angle=DEFAULT_MIN_ANGLE_DEG,
    n3_position=DEFAULT_N3_POSITION,
    nh_h_position=DEFAULT_NH_H_POSITION,
    nh_n_position=DEFAULT_NH_N_POSITION,
    ch_h_positions=None,
    ring_heavy_positions=None,
    include_far=True,
    class_cutoffs=None,
    no_hbond_close_cutoff=DEFAULT_NO_HBOND_CLOSE_CUTOFF_A,
):
    """
    Return structured IMH-water contact classifications for one XYZ file.

    class_cutoffs may map final labels, such as "NH...O" or
    "OH...pi(C1)", to (distance_A, min_angle_deg). With include_far=True,
    waters without any atom-pair contact inside no_hbond_close_cutoff
    are returned as interaction_label == "far" instead of being omitted.
    """
    return analyze_xyz(
        path,
        imidazole_atoms_count=imidazole_atoms_count,
        water_atoms_count=water_atoms_count,
        cutoff=cutoff,
        n3_position=n3_position,
        nh_h_position=nh_h_position,
        nh_n_position=nh_n_position,
        ch_h_positions=ch_h_positions,
        ring_heavy_positions=ring_heavy_positions,
        min_angle=min_angle,
        include_far=include_far,
        class_cutoffs=class_cutoffs,
        no_hbond_close_cutoff=no_hbond_close_cutoff,
    )


def write_csv(rows, path):
    fieldnames = [
        "file",
        "title",
        "n_atoms",
        "n_waters",
        "water_index",
        "interaction_type",
        "interaction_label",
        "matched_interactions",
        "matched_interaction_details",
        "interaction_detail",
        "angle_atoms",
        "angle_deg",
        "water_atom_label",
        "water_atom_index",
        "water_atom_element",
        "imidazole_atom_label",
        "imidazole_atom_index",
        "imidazole_atom_element",
        "distance_A",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output.pop("donor_atom_label", None)
            output["distance_A"] = f"{row['distance_A']:.6f}"
            output["angle_deg"] = (
                "" if row["angle_deg"] is None else f"{row['angle_deg']:.3f}"
            )
            writer.writerow(output)


def write_markdown(rows, path, imidazole_atoms=None, n3_position=8):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "| File | Water | Interaction | Water atom | "
            "Imidazole atom | H...A Distance (A) | Angle | Angle (deg) | Matched | All matched details |\n"
        )
        handle.write("|---|---:|---|---|---|---:|---|---:|---|---|\n")
        for row in rows:
            water_atom = (
                f"W{row['water_index']}-{row['water_atom_label']} "
                f"({row['water_atom_element']}{row['water_atom_index']})"
            )
            im_atom = (
                f"{row['imidazole_atom_label']} "
                f"(atom {row['imidazole_atom_index']})"
            )
            angle_deg_text = (
                "" if row["angle_deg"] is None else f"{row['angle_deg']:.1f}"
            )
            handle.write(
                f"| {row['file']} | {row['water_index']} | "
                f"{row['interaction_label']} | "
                f"{water_atom} | {im_atom} | {row['distance_A']:.3f} | "
                f"{row['angle_atoms']} | {angle_deg_text} | "
                f"{row['matched_interactions']} | {row['matched_interaction_details']} |\n"
            )

        handle.write("\n## Interaction counts\n\n")
        handle.write(f"Total classified water contacts: {len(rows)}\n\n")
        handle.write(
            "| Interaction | Count | H...A mean (A) | H...A std (A) | "
            "Angle mean (deg) | Angle std (deg) |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for stat in interaction_stats_rows(rows):
            handle.write(
                f"| {stat['label']} | {stat['count']} | "
                f"{format_stat(stat['distance_mean'])} | "
                f"{format_stat(stat['distance_std'])} | "
                f"{format_stat(stat['angle_mean'])} | "
                f"{format_stat(stat['angle_std'])} |\n"
            )

        if imidazole_atoms:
            labels, c_labels, n_labels, h_notes = imidazole_atom_notes(
                imidazole_atoms,
                n3_position,
            )
            handle.write("\n## Atom label definitions\n\n")
            handle.write(
                "IMH atom labels are assigned from the first "
                f"{len(imidazole_atoms)} atoms of each XYZ structure. "
                "The labels below are generated from the first valid XYZ file and "
                "therefore follow the same atom order used by the analysis.\n\n"
            )
            handle.write(f"- N3 is atom {n3_position} in the IMH block.\n")
            other_n_labels = [label for label in n_labels if label != "N3"]
            if other_n_labels:
                handle.write(
                    "- Other ring nitrogen label(s): "
                    f"{', '.join(other_n_labels)}.\n"
                )
            handle.write(
                "- C labels in the ring atom order: "
                f"{', '.join(c_labels)}.\n"
            )
            handle.write(
                "- OH...pi uses ring heavy atoms except N3; therefore possible "
                "pi labels are the C labels above plus any non-N3 ring nitrogen "
                "listed above.\n"
            )
            handle.write(
                "- CH...O(Cx) is assigned by the nearest carbon attached to the "
                "donor H.\n"
            )
            handle.write(
                "- noHbond-close(HH-Hx) means the closest non-H-bond atom pair is "
                "water H to IMH Hx; noHbond-close(OH-Hx) means water O to IMH Hx.\n"
            )
            if h_notes:
                handle.write("\nHydrogen attachments inferred by nearest heavy atom:\n\n")
                for note in h_notes:
                    handle.write(f"- {note}\n")


def write_skipped(rows, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classify imidazole-water D-H...A contacts in XYZ structures by "
            "H...A distance and D-H...A angle cutoffs."
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path("xyz_out"),
        help="Directory containing XYZ files. Default: xyz_out",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("imh_water_contacts.csv"),
        help="CSV output table. Default: imh_water_contacts.csv",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("imh_water_contacts.md"),
        help="Markdown output table. Default: imh_water_contacts.md",
    )
    parser.add_argument(
        "--skipped",
        type=Path,
        default=Path("imh_water_contacts_skipped.csv"),
        help="CSV table for excluded or invalid XYZ files. Default: imh_water_contacts_skipped.csv",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=["all_structures.xyz"],
        help=(
            "XYZ filename to exclude. Can be repeated. "
            "Default excludes all_structures.xyz because it is a concatenated multi-frame file."
        ),
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CUTOFF_A,
        help="H...A distance cutoff in Angstrom for interaction classification.",
    )
    parser.add_argument(
        "--min-angle",
        type=float,
        default=DEFAULT_MIN_ANGLE_DEG,
        help=(
            "Minimum donor-heavy-H-acceptor angle in degrees. "
            "Interactions require angle > this value."
        ),
    )
    parser.add_argument(
        "--imidazole-atoms",
        type=int,
        default=DEFAULT_IMIDAZOLE_ATOMS_COUNT,
        help="Number of leading atoms belonging to imidazole. Default: 9",
    )
    parser.add_argument(
        "--water-atoms",
        type=int,
        default=DEFAULT_WATER_ATOMS_COUNT,
        help="Number of atoms per water molecule. Default: 3",
    )
    parser.add_argument(
        "--n3-index",
        type=int,
        default=DEFAULT_N3_POSITION,
        help="1-based N3 atom index inside the leading imidazole block. Default: 8",
    )
    parser.add_argument(
        "--nh-h-index",
        type=int,
        default=DEFAULT_NH_H_POSITION,
        help=(
            "1-based N-H hydrogen index inside the imidazole block for NH...O. "
            "Use 0 to disable this class. Default: 6"
        ),
    )
    parser.add_argument(
        "--nh-n-index",
        type=int,
        default=DEFAULT_NH_N_POSITION,
        help="1-based donor N index for the NH...O angle. Default: 9",
    )
    parser.add_argument(
        "--ch-h-indices",
        default=None,
        help=(
            "Comma-separated 1-based imidazole C-H hydrogen indices for CH...O. "
            "Default: all imidazole H atoms except --nh-h-index"
        ),
    )
    parser.add_argument(
        "--ring-heavy-indices",
        default=None,
        help=(
            "Comma-separated 1-based imidazole ring heavy-atom indices for OH...pi. "
            "Default: all non-H atoms in the imidazole block except N3"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    try:
        ch_h_positions = parse_index_list(args.ch_h_indices, "--ch-h-indices")
        ring_heavy_positions = parse_index_list(
            args.ring_heavy_indices,
            "--ring-heavy-indices",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    excluded_names = set(args.exclude or [])
    all_xyz_files = sorted(args.input_dir.glob("*.xyz"), key=natural_key)
    xyz_files = [path for path in all_xyz_files if path.name not in excluded_names]
    if not xyz_files:
        raise SystemExit(f"No .xyz files found in {args.input_dir}")

    rows = []
    reference_imidazole_atoms = None
    skipped = [
        {"file": path.name, "reason": "excluded by --exclude"}
        for path in all_xyz_files
        if path.name in excluded_names
    ]
    for path in xyz_files:
        try:
            if reference_imidazole_atoms is None:
                _, reference_atoms = read_xyz(path)
                reference_imidazole_atoms = reference_atoms[: args.imidazole_atoms]
            rows.extend(
                analyze_xyz(
                    path,
                    imidazole_atoms_count=args.imidazole_atoms,
                    water_atoms_count=args.water_atoms,
                    cutoff=args.cutoff,
                    n3_position=args.n3_index,
                    nh_h_position=args.nh_h_index,
                    nh_n_position=args.nh_n_index,
                    ch_h_positions=ch_h_positions,
                    ring_heavy_positions=ring_heavy_positions,
                    min_angle=args.min_angle,
                )
            )
        except Exception as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
            print(f"WARNING: skipping {path.name}: {exc}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.skipped.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output)
    write_markdown(
        rows,
        args.markdown,
        imidazole_atoms=reference_imidazole_atoms,
        n3_position=args.n3_index,
    )
    write_skipped(skipped, args.skipped)

    actual_excluded_count = sum(1 for path in all_xyz_files if path.name in excluded_names)
    invalid_count = len(skipped) - actual_excluded_count
    valid_count = len(xyz_files) - invalid_count

    print(f"Found {len(all_xyz_files)} XYZ files.")
    print(f"Analyzed {valid_count} valid single-structure XYZ files.")
    print(f"Skipped {len(skipped)} files.")
    print(f"Found {len(rows)} classified water contacts with H...A distance < {args.cutoff:g} A.")
    for stat in interaction_stats_rows(rows):
        print(
            f"  {stat['label']}: n={stat['count']}, "
            f"H...A mean={format_stat(stat['distance_mean'])} A, "
            f"H...A std={format_stat(stat['distance_std'])} A, "
            f"angle mean={format_stat(stat['angle_mean'])} deg, "
            f"angle std={format_stat(stat['angle_std'])} deg"
        )
    print(f"CSV table: {args.output}")
    print(f"Markdown table: {args.markdown}")
    print(f"Skipped-file table: {args.skipped}")


if __name__ == "__main__":
    main()
