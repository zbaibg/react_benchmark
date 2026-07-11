#!/usr/bin/env python3
"""Generate ccsdtest.csv from dE.csv.

The table intentionally reports one decimal place for energies and two decimal
places for dE_RMSE and BSSE_RMSE. Both RMSE values are computed from the
full-precision source data over the relative energy columns only, not over the
large absolute total energies.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


INCLUDE_TOTAL_ENERGIES = False

TOTAL_ENERGY_COLUMNS = [
    "E_Zn",
    "E_MImH",
    "E_MIm",
    "E_MeOH",
    "E_1Zn_1MImH_0MeOH",
    "E_1Zn_0MIm_1MeOH",
    "E_1Zn_1MIm_0MeOH",
]

RELATIVE_ENERGY_COLUMNS = [
    "Ebind_1Zn_0MIm_1MeOH",
    "Ebind_1Zn_1MIm_0MeOH",
    "Ebind_1Zn_1MImH_0MeOH",
    "dE(1Zn_0MIm_1MeOH->1Zn_1MIm_0MeOH) +1MIm-1MeOH",
    "dE(1Zn_0MIm_1MeOH->1Zn_1MImH_0MeOH) +1MImH-1MeOH",
    "dE_MImH_MIm_H",
    "dE_1Zn_1MImH_0MeOH_1Zn_1MIm_0MeOH_H",
]

OUTPUT_VALUE_COLUMNS = [
    *(TOTAL_ENERGY_COLUMNS if INCLUDE_TOTAL_ENERGIES else []),
    *RELATIVE_ENERGY_COLUMNS,
]

DISPLAY_COLUMN_NAMES = {
    "E_Zn": "E: Zn (kcal/mol)",
    "E_MImH": "E: MImH (kcal/mol)",
    "E_MIm": "E: MIm (kcal/mol)",
    "E_MeOH": "E: MeOH (kcal/mol)",
    "E_1Zn_1MImH_0MeOH": "E: Zn(MImH) (kcal/mol)",
    "E_1Zn_0MIm_1MeOH": "E: Zn(MeOH) (kcal/mol)",
    "E_1Zn_1MIm_0MeOH": "E: Zn(MIm) (kcal/mol)",
    "Ebind_1Zn_0MIm_1MeOH": "dE: Zn + MeOH -> Zn(MeOH) (kcal/mol)",
    "Ebind_1Zn_1MIm_0MeOH": "dE: Zn + MIm -> Zn(MIm) (kcal/mol)",
    "Ebind_1Zn_1MImH_0MeOH": "dE: Zn + MImH -> Zn(MImH) (kcal/mol)",
    "dE(1Zn_0MIm_1MeOH->1Zn_1MIm_0MeOH) +1MIm-1MeOH": (
        "dE: Zn(MeOH) + MIm -> Zn(MIm) + MeOH (kcal/mol)"
    ),
    "dE(1Zn_0MIm_1MeOH->1Zn_1MImH_0MeOH) +1MImH-1MeOH": (
        "dE: Zn(MeOH) + MImH -> Zn(MImH) + MeOH (kcal/mol)"
    ),
    "dE_MImH_MIm_H": "dE: MImH -> MIm + H+ (kcal/mol)",
    "dE_1Zn_1MImH_0MeOH_1Zn_1MIm_0MeOH_H": (
        "dE: Zn(MImH) -> Zn(MIm) + H+ (kcal/mol)"
    ),
}

TABLE_COLUMNS = [
    "Run",
    "Program",
    "Main setting",
    "Other settings",
    *[DISPLAY_COLUMN_NAMES[column] for column in OUTPUT_VALUE_COLUMNS],
    "dE_RMSE (kcal/mol)",
    "Compared to",
    "BSSE_RMSE (kcal/mol)",
]

VALUE_COLUMNS = TOTAL_ENERGY_COLUMNS + RELATIVE_ENERGY_COLUMNS
RMSE_COLUMNS = RELATIVE_ENERGY_COLUMNS

TABLE_NOTE = (
    "dE_RMSE and BSSE_RMSE are full-precision RMSE values over the 7 relative-energy columns; "
    "total energies are excluded. BSSE_RMSE compares each row with the same row recomputed from BSSE=no data."
)

RAW_SELECTORS = {
    # ORCA F12 rows appear twice in dE.csv; the benchmark table uses the
    # F12-corrected BSSE row for run77.
    "run77": {"F12corr": "yes"},
}

DERIVED_DEFINITIONS = {
    "run62-run36": [(1.0, "run62"), (-1.0, "run36")],
    "run57-run36": [(1.0, "run57"), (-1.0, "run36")],
    "run56-run49": [(1.0, "run56"), (-1.0, "run49")],
    "run50-run49": [(1.0, "run50"), (-1.0, "run49")],
    "run45-run37": [(1.0, "run45"), (-1.0, "run37")],
    "run56-run49+run77": [(1.0, "run56"), (-1.0, "run49"), (1.0, "run77")],
    "run50-run49+run77": [(1.0, "run50"), (-1.0, "run49"), (1.0, "run77")],
    "run125+run62-run57": [(1.0, "run125"), (1.0, "run62"), (-1.0, "run57")],
}


@dataclass(frozen=True)
class RowSpec:
    kind: str
    run: str = ""
    source: str = ""
    method: str = ""
    rmse_ref: str = ""
    display_note: str = ""


def section(title: str) -> RowSpec:
    return RowSpec(kind="section", method=title)


def data(run: str, rmse_ref: str = "", display_note: str = "") -> RowSpec:
    return RowSpec(kind="data", run=run, source=run, rmse_ref=rmse_ref, display_note=display_note)


def expr(
    run: str,
    method: str,
    source: str | None = None,
    rmse_ref: str = "",
    display_note: str = "",
) -> RowSpec:
    return RowSpec(
        kind="expr",
        run=run,
        source=source or run,
        method=method,
        rmse_ref=rmse_ref,
        display_note=display_note,
    )


TABLE_LAYOUT = [
    section("PNO/RI for ECP"),
    data("run57"),
    data("run126", rmse_ref="run57", display_note="vs run57"),
    data("run50", rmse_ref="run57", display_note="vs run57"),
    data("run123", rmse_ref="run57", display_note="vs run57"),
    data("run124", rmse_ref="run57", display_note="vs run57"),
    data("run61", rmse_ref="run57", display_note="vs run57"),
    section("PNO/RI for Non-ECP"),
    data("run36"),
    data("run49", rmse_ref="run36", display_note="vs run36"),
    data("run38", rmse_ref="run36", display_note="vs run36"),
    section("Valence-Core Correlation"),
    data("run50"),
    data("run52", rmse_ref="run50", display_note="vs run50"),
    data("run49"),
    data("run53", rmse_ref="run49", display_note="vs run49"),
    section("Relativity Effect Correction for ORCA F12"),
    data("run36"),
    data("run62"),
    data("run57", rmse_ref="run62", display_note="vs run62"),
    data("run49"),
    data("run56"),
    data("run50", rmse_ref="run56", display_note="vs run56"),
    data("run37"),
    data("run45"),
    expr("run62-run36", "CCSD(T) DKH2 effect"),
    expr(
        "run56-run49",
        "DLPNO-CCSD(T) DKH2 effect",
        rmse_ref="run62-run36",
        display_note="vs run62-run36",
    ),
    expr("run57-run36", "CCSD(T) ECP effect"),
    expr(
        "run50-run49",
        "DLPNO-CCSD(T) ECP effect",
        rmse_ref="run57-run36",
        display_note="vs run57-run36",
    ),
    expr("run45-run37", "DFT-ECP effect", rmse_ref="run57-run36", display_note="vs run57-run36"),
    data("run77"),
    expr("run56-run49+run77", "DLPNO-CCSD(T)-F12 DKH2  corrected"),
    expr(
        "run50-run49+run77",
        "DLPNO-CCSD(T)-F12 ECP  corrected",
        rmse_ref="run56-run49+run77",
        display_note="vs run56-run49+run77",
    ),
    section("Basis Set Limit for ORCA"),
    data("run50", rmse_ref="run50-run49+run77", display_note="vs run50-run49+run77"),
    data("run59", rmse_ref="run50-run49+run77", display_note="vs run50-run49+run77"),
    expr("run50-run49+run77", "DLPNO-CCSD(T)-F12 ECP  corrected"),
    section("Basis Set Limit for Molpro"),
    data("run61", rmse_ref="run60", display_note="vs run60"),
    data("run60"),
    section("PNO-ECP-F12 vs Canonical-ECP-F12"),
    data("run125"),
    expr(
        "run50-run49+run77",
        "DLPNO-CCSD(T)-F12 ECP  corrected",
        rmse_ref="run125",
        display_note="vs run125",
    ),
    data("run60", rmse_ref="run125", display_note="vs run125"),
    section("PNO-ECP/DKH2-F12 vs Canonical-DKH2-F12"),
    expr(
        "run125+run62-run57",
        "Molpro CCSD(T)-F12 avtz-f12 (Zn avtz-pp-f12) + DKH2 correction",
    ),
    expr(
        "run56-run49+run77",
        "DLPNO-CCSD(T)-F12 DKH2  corrected",
        rmse_ref="run125+run62-run57",
        display_note="vs run125+run62-run57",
    ),
    expr(
        "run50-run49+run77",
        "DLPNO-CCSD(T)-F12 ECP  corrected",
        rmse_ref="run125+run62-run57",
        display_note="vs run125+run62-run57",
    ),
    data("run60", rmse_ref="run125+run62-run57", display_note="vs run125+run62-run57"),
    section("Canonical-DKH2-F12 vs Canonical-ECP-F12"),
    data("run125"),
    expr(
        "run125+run62-run57",
        "Molpro CCSD(T)-F12 avtz-f12 (Zn avtz-pp-f12) + DKH2 correction",
        rmse_ref="run125",
        display_note="vs run125",
    ),
]


class BenchmarkBuilder:
    def __init__(
        self,
        source_rows: dict[str, dict[str, str]],
        no_bsse_source_rows: dict[str, dict[str, str]],
    ) -> None:
        self.source_rows = source_rows
        self.no_bsse_source_rows = no_bsse_source_rows
        self.value_cache: dict[tuple[str, bool], dict[str, float | None]] = {}

    def raw_row(self, run: str, include_bsse: bool = True) -> dict[str, str]:
        source_rows = self.source_rows if include_bsse else self.no_bsse_source_rows
        if run not in source_rows:
            raise KeyError(f"Missing source row for {run}")
        return source_rows[run]

    def values(self, key: str, include_bsse: bool = True) -> dict[str, float | None]:
        cache_key = (key, include_bsse)
        if cache_key in self.value_cache:
            return self.value_cache[cache_key]

        if key in DERIVED_DEFINITIONS:
            values = {column: 0.0 for column in VALUE_COLUMNS}
            for coefficient, source_key in DERIVED_DEFINITIONS[key]:
                source_values = self.values(source_key, include_bsse=include_bsse)
                for column in VALUE_COLUMNS:
                    source_value = source_values[column]
                    if source_value is None:
                        values[column] = None
                    elif values[column] is not None:
                        values[column] += coefficient * source_value
            self.value_cache[cache_key] = values
            return values

        row = self.raw_row(key, include_bsse=include_bsse)
        values = {column: parse_float(row.get(column, "")) for column in VALUE_COLUMNS}
        self.value_cache[cache_key] = values
        return values

    def method(self, key: str) -> str:
        return self.raw_row(key)["Method"]

    def build_row(self, spec: RowSpec) -> list[str]:
        if spec.kind == "section":
            return ["", "", spec.method, ""] + [""] * (len(TABLE_COLUMNS) - 4)

        values = self.values(spec.source)
        no_bsse_values = self.values(spec.source, include_bsse=False)
        raw_method = spec.method or self.method(spec.source)
        program = infer_program(raw_method)
        method, settings = split_method_name(raw_method)
        rmse = ""
        if spec.rmse_ref:
            rmse = f"{calculate_rmse(values, self.values(spec.rmse_ref)):.2f}"
        bsse_rmse = f"{calculate_rmse(values, no_bsse_values):.2f}"

        return [
            spec.run,
            program,
            method,
            settings,
            *[format_one_decimal(values[column]) for column in OUTPUT_VALUE_COLUMNS],
            rmse,
            format_compared_to(spec.display_note),
            bsse_rmse,
        ]

    def build_table(self) -> list[list[str]]:
        note_row = ["Note", "", TABLE_NOTE, ""] + [""] * (len(TABLE_COLUMNS) - 4)
        return [TABLE_COLUMNS] + [self.build_row(spec) for spec in TABLE_LAYOUT] + [note_row]


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    return float(value)


def format_one_decimal(value: float | None) -> str:
    if value is None:
        return ""
    rounded = f"{value:.1f}"
    return "0.0" if rounded == "-0.0" else rounded


def infer_program(method: str) -> str:
    return "Molpro" if "Molpro" in method else "ORCA"


def split_method_name(method: str) -> tuple[str, str]:
    replacements = {
        "HCNO_aug-cc-pVTZ_Zn_aug-cc-pVTZ-PP-F12": "avtz-(pp)-f12",
        "HCNO_aug-cc-pVTZ_Zn_aug-cc-pVTZ-PP": "avtz-(pp)",
        "HCNO_cc-pVTZ-F12_Zn_cc-pVTZ-F12-wis": "vtz-f12-wis",
        "aug-cc-p(wC)VTZ-(PP)": "a(wc)vtz-(pp)",
        "aug-cc-p(wC)VTZ": "a(wc)vtz",
        "aug-cc-pVQZ-(PP)": "avqz-(pp)",
        "aug-cc-pVTZ-(PP)": "avtz-(pp)",
        "aug-cc-pVTZ-DK": "avtz-DK",
        "aug-cc-pVTZ": "avtz",
        "avtz-f12 (Zn avtz-pp-f12)": "avtz-(pp)-f12",
        "avtz (Zn avtz-pp)": "avtz-(pp)",
        "M052X": "DFT/M052X",
    }
    for old, new in replacements.items():
        method = method.replace(old, new)

    method = method.replace("Molpro", "")
    ignored_tokens = {"NoAutoStart", "NoAmber", "ORCA-native", "NormalPNO"}
    setting_tokens = {
        "VERYTIGHTSCF",
        "EXTREMESCF",
        "Thresh=1e-16",
        "TCut=1e-18",
        "AutoAux",
        "D3ZERO",
        "DEFGRID3",
        "PCDTrimAuxJ",
    }
    method_parts = []
    settings = []
    for token in method.split():
        if token in ignored_tokens:
            continue
        if token in setting_tokens or token.endswith("/C"):
            settings.append(token)
        else:
            method_parts.append(token)

    return " ".join(method_parts).strip(), "; ".join(settings).strip()


def format_compared_to(display_note: str) -> str:
    return display_note.removeprefix("vs ").strip()


def calculate_rmse(left: dict[str, float | None], right: dict[str, float | None]) -> float:
    squared_errors = []
    for column in RMSE_COLUMNS:
        left_value = left[column]
        right_value = right[column]
        if left_value is None or right_value is None:
            raise ValueError(f"Cannot calculate RMSE with missing value in {column}")
        squared_errors.append((left_value - right_value) ** 2)
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def load_source_rows(path: Path, bsse: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected: dict[str, dict[str, str]] = {}
    needed_runs = sorted(needed_raw_runs(), key=run_sort_key)
    for run in needed_runs:
        selector = {"BSSE": bsse, "F12corr": "no", **RAW_SELECTORS.get(run, {})}
        matches = [
            row
            for row in rows
            if row["Run"] == run and all(row.get(key, "") == value for key, value in selector.items())
        ]
        if not matches:
            fallback_matches = [
                row
                for row in rows
                if row["Run"] == run
                and row.get("BSSE", "") == bsse
                and all(row.get(key, "") == value for key, value in RAW_SELECTORS.get(run, {}).items())
            ]
            matches = fallback_matches
        if len(matches) != 1:
            detail = ", ".join(
                f"BSSE={row.get('BSSE')} F12corr={row.get('F12corr')} MRCC={row.get('MRCCLevel')}"
                for row in matches
            )
            raise ValueError(f"Expected one selected row for {run}, found {len(matches)}: {detail}")
        selected[run] = matches[0]
    return selected


def needed_raw_runs() -> set[str]:
    runs = set()

    def visit(key: str) -> None:
        if key in DERIVED_DEFINITIONS:
            for _, child_key in DERIVED_DEFINITIONS[key]:
                visit(child_key)
        else:
            runs.add(key)

    for spec in TABLE_LAYOUT:
        if spec.kind in {"data", "expr"}:
            visit(spec.source)
            if spec.rmse_ref:
                visit(spec.rmse_ref)
    return runs


def run_sort_key(run: str) -> tuple[int, str]:
    digits = "".join(character for character in run if character.isdigit())
    return (int(digits) if digits else -1, run)


def write_csv_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def check_comparison_notes() -> list[str]:
    messages = []
    for row_number, spec in enumerate(TABLE_LAYOUT, start=2):
        if not spec.rmse_ref:
            continue

        expected_note = f"vs {spec.rmse_ref}"
        if not spec.display_note:
            messages.append(
                f"CSV row {row_number} ({spec.run}) has dE_RMSE vs {spec.rmse_ref!r}, "
                "but the comparison-note column is empty."
            )
        elif spec.display_note != expected_note:
            messages.append(
                f"CSV row {row_number} ({spec.run}) displays {spec.display_note!r}, "
                f"but its dE_RMSE is calculated against {spec.rmse_ref!r}."
            )
    return messages


def print_messages(title: str, messages: list[str], limit: int) -> None:
    print(title)
    if not messages:
        print("  PASS")
        return

    print(f"  FAIL: {len(messages)} issue(s)")
    for message in messages[:limit]:
        print(f"  - {message}")
    if len(messages) > limit:
        print(f"  - ... {len(messages) - limit} more issue(s) omitted")


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_dir / "dE.csv", help="Input dE.csv path")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "ccsdtest.csv",
        help="Output CSV path for the generated benchmark table",
    )
    parser.add_argument("--max-mismatches", type=int, default=20, help="Maximum detailed mismatches to print")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builder = BenchmarkBuilder(
        source_rows=load_source_rows(args.input, bsse="yes"),
        no_bsse_source_rows=load_source_rows(args.input, bsse="no"),
    )
    generated_rows = builder.build_table()

    note_issues = check_comparison_notes()
    if note_issues:
        print_messages("Generated comparison-label logic check:", note_issues, args.max_mismatches)
        return 1

    write_csv_rows(args.output, generated_rows)

    print(f"Loaded source data: {args.input}")
    print(f"Wrote generated table: {args.output}")
    print(f"Generated rows: {len(generated_rows)}")
    print("dE_RMSE/BSSE_RMSE definition: full-precision RMSE over 7 relative-energy columns.")
    print("Generated comparison-label logic check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
