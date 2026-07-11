#!/usr/bin/env python3
"""Print bash export lines for software.yaml. Used by repo_env.sh."""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import REPO_ROOT, load_software  # noqa: E402


def main() -> None:
    sw = load_software()
    print(f"export REPO_ROOT={shlex.quote(str(REPO_ROOT))}")
    mapping = {
        "ORCA_PATH": sw.get("orca"),
        "ORCA_LEGACY_PATH": sw.get("orca_legacy"),
        "MOLPRO_ROOT": sw.get("molpro_root"),
        "MRCC_PATH": sw.get("mrcc"),
        "AMBER_SH": sw.get("ambertools_sh"),
        "AMBER_SH_LEGACY": sw.get("ambertools_sh_legacy"),
        "XTB_GAUSSIAN": sw.get("xtb_gaussian"),
        "CONDAINIT": sw.get("condainit"),
        "SCRATCH_ROOT": sw.get("scratch_root"),
        "APPTAINER_SIF": sw.get("apptainer"),
        "GXTB_HOME": (sw.get("gxtb") or {}).get("home"),
        "GXTB_BINARY": (sw.get("gxtb") or {}).get("binary"),
    }
    for name, val in mapping.items():
        if val is not None:
            print(f"export {name}={shlex.quote(str(val))}")
    for key, val in (sw.get("dftb_skf") or {}).items():
        env = "DFTB_SKF_" + key.replace("-", "_").replace(".", "_")
        print(f"export {env}={shlex.quote(str(val))}")


if __name__ == "__main__":
    main()
