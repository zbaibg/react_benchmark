from __future__ import annotations

import os

import ccrepo as cc


def _merge_basis_dicts(*dicts: dict) -> dict:
    merged: dict = {}
    for d in dicts:
        merged.update(d)
    return merged


def write_set(
    *,
    out_dir: str,
    hcn_basis: str,
    zn_basis: str,
) -> None:
    """
    Write one ORCA-readable basis file for H/C/N/O and Zn.
    The caller passes explicit basis names for each set type
    (main, OptRI, MP2Fit), and this function writes that exact set.
    """
    elements_hcn = ["H", "C", "N", "O"]
    elements_zn = ["Zn"]

    def out_path(basis_name: str) -> str:
        return os.path.join(out_dir, basis_name)

    basis_set = _merge_basis_dicts(
        cc.fetch_basis(elements_hcn, hcn_basis),
        cc.fetch_basis(elements_zn, zn_basis),
    )
    stem = f"HCNO_{hcn_basis}_Zn_{zn_basis}"
    cc.write_basis(basis_set, out_path(stem), "orca")


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Set 1: cc-pVTZ-F12 / cc-pVTZ-PP-F12
    write_set(out_dir=out_dir, hcn_basis="cc-pVTZ-F12", zn_basis="cc-pVTZ-PP-F12")
    write_set(out_dir=out_dir, hcn_basis="cc-pVTZ-F12_OptRI", zn_basis="cc-pVTZ-PP-F12_OptRI")

    # Set 2: aug-cc-pVTZ / aug-cc-pVTZ-PP-F12
    write_set(out_dir=out_dir, hcn_basis="aug-cc-pVTZ", zn_basis="aug-cc-pVTZ-PP-F12")
    write_set(out_dir=out_dir, hcn_basis="aug-cc-pVTZ_OptRI", zn_basis="aug-cc-pVTZ-PP-F12_OptRI")

    # Set 3: cc-pVTZ-F12 / aug-cc-pVTZ-PP
    write_set(out_dir=out_dir, hcn_basis="cc-pVTZ-F12", zn_basis="aug-cc-pVTZ-PP")
    write_set(out_dir=out_dir, hcn_basis="cc-pVTZ-F12_OptRI", zn_basis="aug-cc-pVTZ-PP_OptRI")

    # Set 4: cc-pVTZ-F12 / aug-cc-pVTZ-PP-F12
    write_set(out_dir=out_dir, hcn_basis="cc-pVTZ-F12", zn_basis="aug-cc-pVTZ-PP-F12")
    write_set(out_dir=out_dir, hcn_basis="cc-pVTZ-F12_OptRI", zn_basis="aug-cc-pVTZ-PP-F12_OptRI")
    write_set(out_dir=out_dir, hcn_basis="cc-pVTZ-F12_MP2Fit", zn_basis="aug-cc-pVTZ-PP-F12_MP2Fit")
    write_set(out_dir=out_dir, hcn_basis="cc-pVQZ_JKFit", zn_basis="cc-pVQZ-PP_JKFit")
    write_set(out_dir=out_dir, hcn_basis="cc-pVTZ_JKFit", zn_basis="cc-pVTZ-PP_JKFit")

    # Set 5: cc-pVQZ-F12 / aug-cc-pVQZ-PP-F12
    write_set(out_dir=out_dir, hcn_basis="cc-pVQZ-F12", zn_basis="aug-cc-pVQZ-PP-F12")
    write_set(out_dir=out_dir, hcn_basis="cc-pVQZ-F12_OptRI", zn_basis="aug-cc-pVQZ-PP-F12_OptRI")
    write_set(out_dir=out_dir, hcn_basis="cc-pVQZ-F12_MP2Fit", zn_basis="aug-cc-pVQZ-PP-F12_MP2Fit")
    write_set(out_dir=out_dir, hcn_basis="cc-pV5Z_JKFit", zn_basis="cc-pV5Z-PP_JKFit")

    # Set 5: cc-pVDZ-F12 / aug-cc-pVDZ-PP-F12
    write_set(out_dir=out_dir, hcn_basis="cc-pVDZ-F12", zn_basis="aug-cc-pVDZ-PP-F12")
    write_set(out_dir=out_dir, hcn_basis="cc-pVDZ-F12_OptRI", zn_basis="aug-cc-pVDZ-PP-F12_OptRI")
    write_set(out_dir=out_dir, hcn_basis="cc-pVDZ-F12_MP2Fit", zn_basis="aug-cc-pVDZ-PP-F12_MP2Fit")
if __name__ == "__main__":
    main()
