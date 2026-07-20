from __future__ import annotations

import os

import ccrepo as cc


def _merge_basis_dicts(*dicts: dict) -> dict:
    merged: dict = {}
    for d in dicts:
        merged.update(d)
    return merged


def _write_set(
    *,
    out_dir: str,
    stem: str,
    hcn_basis: str,
    zn_basis: str,
    optri,
    mp2fit
) -> None:
    """
    Write three ORCA-readable basis files:
      - GTO basis
      - AuxC (MP2Fit)
      - CABS (OptRI)
    for elements H/C/N/O with global defaults and Zn override.
    """
    elements_hcn = ["H", "C", "N", "O"]
    elements_zn = ["Zn"]

    def out_path(basis_name: str) -> str:
        return os.path.join(out_dir, basis_name)

    # Main orbital basis (GTO)
    gto = _merge_basis_dicts(
        cc.fetch_basis(elements_hcn, hcn_basis),
        cc.fetch_basis(elements_zn, zn_basis),
    )
    cc.write_basis(gto, out_path(stem), "orca")
    if mp2fit:
        # AuxC / MP2Fit
        auxc = _merge_basis_dicts(
            cc.fetch_basis(elements_hcn, hcn_basis+"_MP2Fit"),
            cc.fetch_basis(elements_zn, zn_basis+"_MP2Fit"),
        )
        # Put suffix on BOTH HCNO and Zn parts for clarity.
        cc.write_basis(
            auxc,
            out_path(f"HCNO_{hcn_basis}_MP2Fit_Zn_{zn_basis}_MP2Fit"),
            "orca",
        )
    if optri:
        cabs = _merge_basis_dicts(
            cc.fetch_basis(elements_hcn, hcn_basis+"_OptRI"),
            cc.fetch_basis(elements_zn, zn_basis+"_OptRI"),
        )
        cc.write_basis(
            cabs,
            out_path(f"HCNO_{hcn_basis}_OptRI_Zn_{zn_basis}_OptRI"),
            "orca",
        )


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Set 1:
    #   H/C/N: cc-pVTZ-F12 (+ MP2Fit / OptRI)
    #   Zn:    cc-pVTZ-PP-F12 (+ MP2Fit / OptRI)
    _write_set(
        out_dir=out_dir,
        stem="HCNO_cc-pVTZ-F12_Zn_cc-pVTZ-PP-F12",
        hcn_basis="cc-pVTZ-F12",
        zn_basis="cc-pVTZ-PP-F12",
        optri=True,
        mp2fit=False,
    )

    # Set 2:
    #   H/C/N: aug-cc-pVTZ-F12 (+ MP2Fit / OptRI)
    #   Zn:    aug-cc-pVTZ-PP-F12 (+ MP2Fit / OptRI)
    _write_set(
        out_dir=out_dir,
        stem="HCNO_aug-cc-pVTZ_Zn_aug-cc-pVTZ-PP-F12",
        hcn_basis="aug-cc-pVTZ",
        zn_basis="aug-cc-pVTZ-PP-F12",
        optri=True,
        mp2fit=False,
    )
    _write_set(
        out_dir=out_dir,
        stem="HCNO_cc-pVTZ-F12_Zn_aug-cc-pVTZ-PP",
        hcn_basis="cc-pVTZ-F12",
        zn_basis="aug-cc-pVTZ-PP",
        optri=True,
        mp2fit=False,
    )


if __name__ == "__main__":
    main()
