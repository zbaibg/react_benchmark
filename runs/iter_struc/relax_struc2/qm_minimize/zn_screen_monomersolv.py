#!/usr/bin/env python3
"""
Rough solvent-aware screening of Zn complexes in methanol.

Key correction versus earlier scripts:
- Compare every candidate to [Zn(MeOH)6]2+ in methanol.
- Treat MeOH as the solvent reservoir: a_MeOH ~ 1 in the concentration term.
- EXPLICITLY include monomer solvation free energies for MeOH and MImH.
- Keep cluster solvation as a rough Born-like term unless user later replaces it.

Recommended use:
1) Compute monomer solvation free energies in methanol (e.g. SMD/CPCM) for:
      MeOH, MImH
   and optionally cluster solvation free energies if available.
2) Plug them into this script.
3) Use rel_global_kcal for coarse screening.

This script is intentionally honest: if monomer solvation inputs are missing,
it will stop instead of silently producing a biased ranking.
"""

import argparse
import math
import re
from pathlib import Path

import pandas as pd

RT_298 = 0.00198720425864083 * 298.15
RTLN10_298 = RT_298 * math.log(10.0)
BORN_CONST_KCAL_A = 166.03185664959605

TOKEN_PATTERNS = {
    "Zn": re.compile(r"(\d+)Zn"),
    "NO3": re.compile(r"(\d+)NO3"),
    "MImH": re.compile(r"(\d+)MImH"),
    "MIm": re.compile(r"(\d+)MIm(?!H)"),
    "MeOH": re.compile(r"(\d+)MeOH"),
    "H2O": re.compile(r"(\d+)H2O"),
}
DIST_COLS = [
    "max_Zn-Nmin_MIM(Å)",
    "max_Zn-O_MeOH(Å)",
    "max_Zn-O_H2O(Å)",
    "max_Zn-O_NO3(Å)",
]

def parse_counts(system: str) -> dict:
    text = str(system)
    out = {k: 0 for k in TOKEN_PATTERNS}
    for key, pat in TOKEN_PATTERNS.items():
        m = pat.search(text)
        if m:
            out[key] = int(m.group(1))
    return out

def born_solvation(charge: float, radius_a: float, eps: float) -> float:
    return -BORN_CONST_KCAL_A * (1.0 - 1.0 / eps) * (charge ** 2) / radius_a

def pick_radius(row: pd.Series, pad: float, default_radius: float = 4.0) -> float:
    vals = []
    for col in DIST_COLS:
        if col in row.index:
            v = pd.to_numeric(row[col], errors="coerce")
            if not pd.isna(v):
                vals.append(float(v))
    return (max(vals) + pad) if vals else default_radius

def logsumexp(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return float("nan")
    vmax = max(vals)
    return vmax + math.log(sum(math.exp(v - vmax) for v in vals))

def effective_deprot_event_kcal(
    c_mimh: float | None,
    c_meoh: float | None,
    pka_mimh_acid_water: float,
    dpka_mimh_acid_meoh: float,
    pka_mimh2_water: float,
    dpka_mimh2_meoh: float,
    pka_meoh2_meoh: float,
):
    """
    Approximate free energy per deprotonation event:
        coordinated MImH -> coordinated MIm- + H+
    with two competing proton acceptors in bulk:
        MImH  and  MeOH
    """
    pka_acid = pka_mimh_acid_water + dpka_mimh_acid_meoh
    pka_mimh2 = pka_mimh2_water + dpka_mimh2_meoh

    dg0_via_mimh = RTLN10_298 * (pka_acid - pka_mimh2)
    dg0_via_meoh = RTLN10_298 * (pka_acid - pka_meoh2_meoh)

    dg_via_mimh = float("nan") if not c_mimh or c_mimh <= 0 else dg0_via_mimh - RT_298 * math.log(c_mimh)
    dg_via_meoh = float("nan") if not c_meoh or c_meoh <= 0 else dg0_via_meoh - RT_298 * math.log(c_meoh)

    xs = []
    labs = []
    for lab, val in [("MImH", dg_via_mimh), ("MeOH", dg_via_meoh)]:
        if math.isfinite(val):
            xs.append(-val / RT_298)
            labs.append(lab)

    out = {
        "dG0_via_MImH_kcal": dg0_via_mimh,
        "dG0_via_MeOH_kcal": dg0_via_meoh,
        "dG_via_MImH_kcal": dg_via_mimh,
        "dG_via_MeOH_kcal": dg_via_meoh,
    }

    if xs:
        lse = logsumexp(xs)
        out["dG_event_effective_kcal"] = -RT_298 * lse
        weights = {lab: math.exp(x - lse) for lab, x in zip(labs, xs)}
    else:
        out["dG_event_effective_kcal"] = float("nan")
        weights = {}
    out["frac_via_MImH"] = weights.get("MImH", 0.0)
    out["frac_via_MeOH"] = weights.get("MeOH", 0.0)
    return out

def main():
    ap = argparse.ArgumentParser(description="Zn screening in methanol with EXPLICIT monomer solvation terms.")
    ap.add_argument("input")
    ap.add_argument("--eps", type=float, default=32.6)
    ap.add_argument("--radius-pad-A", type=float, default=1.5)

    # concentrations
    ap.add_argument("--c-MeOH", type=float, default=24.7)
    ap.add_argument("--c-MImH", type=float, default=0.20)

    # monomer solvation free energies in methanol (kcal/mol) -- REQUIRED
    ap.add_argument("--dgsolv-MeOH", type=float, required=True,
                    help="Monomer solvation free energy of MeOH in methanol, kcal/mol.")
    ap.add_argument("--dgsolv-MImH", type=float, required=True,
                    help="Monomer solvation free energy of neutral MImH in methanol, kcal/mol.")

    # optional direct cluster solvation corrections
    ap.add_argument("--cluster-solv-model", choices=["born"], default="born")
    ap.add_argument("--cluster-solv-offset", type=float, default=0.0,
                    help="Optional constant offset added to all cluster solvation terms; does not affect ranking.")

    # deprotonation proxy
    ap.add_argument("--include-mim-deprot", action="store_true")
    ap.add_argument("--pka-mimh-acid-water", type=float, default=14.5)
    ap.add_argument("--dpka-mimh-acid-meoh", type=float, default=5.4)
    ap.add_argument("--pka-mimh2-water", type=float, default=7.86)
    ap.add_argument("--dpka-mimh2-meoh", type=float, default=0.30)
    ap.add_argument("--pka-meoh2-meoh", type=float, default=0.0)

    ap.add_argument("--assoc-penalty-per-imidazole", type=float, default=0.0,
                    help="Empirical penalty per coordinated imidazole-derived ligand, kcal/mol.")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    inp = Path(args.input)
    df = pd.read_csv(inp, sep="\t")
    df["Etot_last_cycle_kcal_mol"] = pd.to_numeric(df["Etot_last_cycle_kcal_mol"], errors="coerce")

    counts = df["system"].astype(str).apply(parse_counts).apply(pd.Series)
    df = pd.concat([df, counts], axis=1)

    for key in TOKEN_PATTERNS:
        if key not in df:
            df[key] = 0

    df["coord_n"] = df[["MIm", "MImH", "MeOH", "H2O", "NO3"]].sum(axis=1)
    df["charge_est"] = 2 - df["MIm"] - df["NO3"]
    df["lig_imid_total"] = df["MIm"] + df["MImH"]

    energy_by_name = dict(zip(df["system"], df["Etot_last_cycle_kcal_mol"]))
    for required in ["MeOH", "MImH", "1Zn_0MIm_0MImH_6MeOH"]:
        if required not in energy_by_name or pd.isna(energy_by_name[required]):
            raise ValueError(f"Missing required reference energy for {required}")

    E_MeOH = float(energy_by_name["MeOH"])
    E_MImH = float(energy_by_name["MImH"])
    E_ref = float(energy_by_name["1Zn_0MIm_0MImH_6MeOH"])

    ref_row = df.loc[df["system"] == "1Zn_0MIm_0MImH_6MeOH"].iloc[0]
    ref_radius = pick_radius(ref_row, args.radius_pad_A)
    dGsolv_ref_cluster = born_solvation(2.0, ref_radius, args.eps) + args.cluster_solv_offset

    event = effective_deprot_event_kcal(
        args.c_MImH,
        args.c_MeOH,
        args.pka_mimh_acid_water,
        args.dpka_mimh_acid_meoh,
        args.pka_mimh2_water,
        args.dpka_mimh2_meoh,
        args.pka_meoh2_meoh,
    )

    rows = []
    for _, row in df.iterrows():
        if int(row.get("Zn", 0)) != 1 or pd.isna(row["Etot_last_cycle_kcal_mol"]):
            continue

        a = int(row["MIm"])
        b = int(row["MImH"])
        c = int(row["MeOH"])
        L = a + b

        radius = pick_radius(row, args.radius_pad_A)
        dGsolv_cluster = born_solvation(float(row["charge_est"]), radius, args.eps) + args.cluster_solv_offset

        # Solution-phase ligand exchange relative to [Zn(MeOH)6]2+:
        # [Zn(MeOH)6]2+  +  L MImH(sol)
        #   ->  target_cluster  +  (6-c) MeOH(sol)
        #   (+ deprotonation correction for each MIm)
        #
        # This is the key place monomer solvation enters.
        dG_exchange_gas = (
            float(row["Etot_last_cycle_kcal_mol"])
            + (6 - c) * E_MeOH
            - E_ref
            - L * E_MImH
        )
        dG_exchange_solv = (
            dGsolv_cluster
            + (6 - c) * args.dgsolv_MeOH
            - dGsolv_ref_cluster
            - L * args.dgsolv_MImH
        )

        # Only MImH is treated as dilute reagent.
        dG_conc = 0.0
        if L and args.c_MImH and args.c_MImH > 0:
            dG_conc = -RT_298 * L * math.log(args.c_MImH)

        dG_deprot = a * event["dG_event_effective_kcal"] if args.include_mim_deprot else 0.0
        dG_assoc = args.assoc_penalty_per_imidazole * L

        score = dG_exchange_gas + dG_exchange_solv + dG_conc + dG_deprot + dG_assoc

        rec = row.to_dict()
        rec.update(event)
        rec["ref_system"] = "1Zn_0MIm_0MImH_6MeOH"
        rec["dGsolv_cluster_rough_kcal"] = dGsolv_cluster
        rec["dGsolv_ref_cluster_rough_kcal"] = dGsolv_ref_cluster
        rec["dG_exchange_gas_kcal"] = dG_exchange_gas
        rec["dG_exchange_solv_kcal"] = dG_exchange_solv
        rec["dG_conc_kcal"] = dG_conc
        rec["dG_MIm_deprot_kcal"] = dG_deprot
        rec["dG_assoc_penalty_kcal"] = dG_assoc
        rec["screen_score_monomersolv_kcal"] = score
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("screen_score_monomersolv_kcal").reset_index(drop=True)
    out["rel_global_kcal"] = out["screen_score_monomersolv_kcal"] - out["screen_score_monomersolv_kcal"].min()

    prefix = args.out_prefix or (inp.stem + "_screen_monomersolv")
    out_csv = inp.with_name(prefix + ".csv")
    out.to_csv(out_csv, index=False)

    summary = []
    summary.append(f"input = {inp}")
    summary.append("Model = solution-phase ligand exchange relative to [Zn(MeOH)6]2+")
    summary.append("MeOH is treated as the solvent reservoir; MImH is treated as a dilute reagent.")
    summary.append("Monomer solvation terms are EXPLICITLY included:")
    summary.append("  dG_exchange_solv = dGsolv(cluster) + (6-c)*dGsolv(MeOH) - dGsolv(ref) - (a+b)*dGsolv(MImH)")
    summary.append(f"dgsolv(MeOH) = {args.dgsolv_MeOH:.6f} kcal/mol")
    summary.append(f"dgsolv(MImH) = {args.dgsolv_MImH:.6f} kcal/mol")
    summary.append(f"epsilon = {args.eps}")
    summary.append(f"include_mim_deprot = {args.include_mim_deprot}")
    if args.include_mim_deprot:
        summary.append(f"effective deprotonation per MIm = {event['dG_event_effective_kcal']:.6f} kcal/mol")
    summary.append("")
    cols = [
        "system", "coord_n", "charge_est", "MIm", "MImH", "MeOH",
        "dG_exchange_gas_kcal", "dG_exchange_solv_kcal",
        "dG_conc_kcal", "dG_MIm_deprot_kcal",
        "screen_score_monomersolv_kcal", "rel_global_kcal"
    ]
    summary.append("Top 25 species:")
    summary.append(out[cols].head(25).to_string(index=False))
    out_txt = inp.with_name(prefix + "_summary.txt")
    out_txt.write_text("\n".join(summary), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_txt}")

if __name__ == "__main__":
    main()
