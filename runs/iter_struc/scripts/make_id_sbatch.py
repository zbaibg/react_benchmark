#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_CAND = Path(__file__).resolve().parent
while _REPO_CAND != _REPO_CAND.parent and not (_REPO_CAND / "software.yaml").exists():
    _REPO_CAND = _REPO_CAND.parent
REPO_ROOT = _REPO_CAND
sys.path.insert(0, str(REPO_ROOT / "tools"))
from paths import load_software
_SW = load_software()
_ORCA = _SW["orca"]
_XTB_GAUSSIAN = _SW["xtb_gaussian"]
_CONDAINIT = _SW.get("condainit", str(Path.home() / "condainit.sh"))



@dataclass(frozen=True)
class JobConfig:
    name: str  # "orca", "xtb_gaussian", or "dftbplus"
    gen_dir_name: str
    sbatch_filename: str
    job_name_prefix: str
    preamble_lines: list[str]
    metallogen_calculator: str
    final_relax: int = 0  # metallogen -r (final_relax)
    dftbplus_template: str | None = None  # if set, export DFTB_TEMPLATE and pass --dftbplus_template


ORCA = JobConfig(
    name="orca",
    gen_dir_name="gen_struc_orca",
    sbatch_filename="run_orca.sbatch",
    job_name_prefix="orca",
    preamble_lines=[
        f'source "{_CONDAINIT}"',
        "conda activate metallogen",
        f'export PATH="{_ORCA}:${{PATH}}"',
    ],
    metallogen_calculator="orca",
)

XTB_GAUSSIAN = JobConfig(
    name="xtb_gaussian",
    gen_dir_name="gen_struc_xtb_gaussian",
    sbatch_filename="run_xtb_gaussian.sbatch",
    job_name_prefix="xtbg16",
    preamble_lines=[
        f'source "{_CONDAINIT}"',
        "conda activate metallogen",
        "source /share/apps/gaussian/g16/bsd/g16.profile",
        f'export xtbbin="{_XTB_GAUSSIAN}"',
    ],
    metallogen_calculator="xtb_gaussian",
)

DFTBPLUS_DEFAULT_TEMPLATE = str(
    REPO_ROOT / "runs/deprotonate/MIMH_PBE_STRUC/minimize/run6/1Zn_1MImH_3MeOH/dftb_pin.hsd"
)

DFTBPLUS = JobConfig(
    name="dftbplus",
    gen_dir_name="gen_struc_dftbplus",
    sbatch_filename="run_dftbplus.sbatch",
    job_name_prefix="mg_dftb",
    preamble_lines=[
        f'source "{_CONDAINIT}"',
        "conda activate metallogen",
        '# DFTB+ binary must be on PATH, or set e.g. export DFTBPLUS_COMMAND="/path/to/dftb+"',
    ],
    metallogen_calculator="dftbplus",
    final_relax=1,
    dftbplus_template=DFTBPLUS_DEFAULT_TEMPLATE,
)


DEFAULT_NC = 20


SBATCH_HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={stdout}
#SBATCH --error={stderr}
#SBATCH --time=7-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --exclude=compute-0-[0-40,44]

cd "${{SLURM_SUBMIT_DIR}}"
"""


def _safe_lines_from_tsv(tsv_path: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    with tsv_path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if not header:
            return items
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            id_ = parts[0].strip()
            msmiles = parts[-1].strip()
            if not id_ or not msmiles:
                continue
            items.append((id_, msmiles))
    return items


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _sbatch_for_id(cfg: JobConfig, id_: str, msmiles: str) -> str:
    job_name = f"{cfg.job_name_prefix}_{id_}"
    # Keep log filenames consistent with top-level scripts.
    if cfg.name == "orca":
        stdout = "run_orca.log"
        stderr = "run_orca.err"
    elif cfg.name == "xtb_gaussian":
        stdout = "run_xtb_gaussian.log"
        stderr = "run_xtb_gaussian.err"
    elif cfg.name == "dftbplus":
        stdout = "run_metallogen.log"
        stderr = "run_metallogen.err"
    else:
        stdout = f"{cfg.sbatch_filename}.log"
        stderr = f"{cfg.sbatch_filename}.err"

    preamble = "\n".join(cfg.preamble_lines) + "\n"

    # msmiles is embedded as a single-quoted bash string; escape any single quotes.
    msmiles_escaped = msmiles.replace("'", "'\"'\"'")

    dftb_block = ""
    if cfg.dftbplus_template is not None:
        # Path is fixed ASCII; safe for double-quoted bash.
        dftb_block = f'DFTB_TEMPLATE="{cfg.dftbplus_template}"\n'

    metallogen_tail = f"    -c {cfg.metallogen_calculator}"
    if cfg.dftbplus_template is not None:
        metallogen_tail += " \\\n    --dftbplus_template \"${DFTB_TEMPLATE}\""

    body = f"""
{preamble}
{dftb_block}ID="{id_}"
MSMILES='{msmiles_escaped}'
WORKDIR="${{SLURM_SUBMIT_DIR}}/work_dir"
SAVEDIR="${{SLURM_SUBMIT_DIR}}/save_dir"
mkdir -p "${{WORKDIR}}" "${{SAVEDIR}}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START id=${{ID}} ${{MSMILES}}"
{{
  echo "=== id=${{ID}} metallogen msmiles: ${{MSMILES}}"
  export PYTHONUNBUFFERED=1
  metallogen \\
    -s "${{MSMILES}}" \\
    -wd "${{WORKDIR}}" \\
    -sd "${{SAVEDIR}}" \\
    -r {cfg.final_relax} \\
    -nc {DEFAULT_NC} \\
    --skip_failed_clean \\
    --no_qc_clean_lig_broken_bond \\
    --always_qc \\
{metallogen_tail}
}} > metallogen.log 2>&1
rc=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] END id=${{ID}} exit=${{rc}}"
exit "${{rc}}"
""".lstrip(
        "\n"
    )

    return SBATCH_HEADER.format(job_name=job_name, stdout=stdout, stderr=stderr) + body


def _generate_for_cfg(
    base_dir: Path,
    cfg: JobConfig,
    items: list[tuple[str, str]],
) -> int:
    gen_dir = base_dir / cfg.gen_dir_name
    gen_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for id_, msmiles in items:
        id_dir = gen_dir / id_
        (id_dir / "work_dir").mkdir(parents=True, exist_ok=True)
        (id_dir / "save_dir").mkdir(parents=True, exist_ok=True)

        sbatch_path = id_dir / cfg.sbatch_filename
        _write_text(sbatch_path, _sbatch_for_id(cfg, id_, msmiles))
        try:
            sbatch_path.chmod(0o755)
        except PermissionError:
            pass
        written += 1
    return written


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generate per-id SLURM sbatch scripts under gen_struc_orca/<id>/, "
            "gen_struc_xtb_gaussian/<id>/, and/or gen_struc_dftbplus/<id>/ "
            "from zn_msmiles_enumeration.tsv."
        )
    )
    p.add_argument(
        "--tsv",
        default="zn_msmiles_enumeration.tsv",
        help="Input TSV (default: zn_msmiles_enumeration.tsv)",
    )
    p.add_argument(
        "--base-dir",
        default=".",
        help="Base directory containing gen_struc_* folders (default: .)",
    )
    p.add_argument(
        "--calculator",
        choices=["orca", "xtb_gaussian", "dftbplus", "both", "all"],
        default="dftbplus",
        help=(
            "Which calculator(s) to generate: orca, xtb_gaussian, dftbplus; "
            "both = orca + xtb_gaussian; all = all three (default: orca)"
        ),
    )

    args = p.parse_args()
    base_dir = Path(args.base_dir).resolve()
    tsv_path = Path(args.tsv).resolve()

    items = _safe_lines_from_tsv(tsv_path)
    if not items:
        raise SystemExit(f"No valid rows found in TSV: {tsv_path}")

    total = 0
    if args.calculator in ("orca", "both", "all"):
        total += _generate_for_cfg(base_dir, ORCA, items)
    if args.calculator in ("xtb_gaussian", "both", "all"):
        total += _generate_for_cfg(base_dir, XTB_GAUSSIAN, items)
    if args.calculator in ("dftbplus", "all"):
        total += _generate_for_cfg(base_dir, DFTBPLUS, items)

    print(f"Wrote {total} sbatch scripts under {base_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

