#!/usr/bin/env python3
"""Optional charge post-processing stub.

The original form_charges.py was not available when vendoring tools.
This stub leaves the mol2 unchanged so amber_prep/prepare.sh can run.
Replace with a real implementation if charge reformatting is required.
"""
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("usage: form_charges.py MOL2 [CHARGE]", file=sys.stderr)
        sys.exit(1)
    mol2 = Path(sys.argv[1])
    if not mol2.is_file():
        print(f"ERROR: {mol2} not found", file=sys.stderr)
        sys.exit(1)
    print(f"form_charges stub: leaving {mol2} unchanged")

if __name__ == "__main__":
    main()
