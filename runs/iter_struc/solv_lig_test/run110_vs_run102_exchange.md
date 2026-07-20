# run110 vs run102 BSSE=yes — ligand exchange ΔE

**Reaction:** `ZnLSolv5 + Solv → ZnSolv6 + L` (kcal/mol)

| Method | Description                          |
| ------ | ------------------------------------ |
| run110 | mDFTB3-D3 / optimized_run59 (all QM) |
| run102 | Molpro PNO-LCCSD(T)-F12 (BSSE = yes) |

Source: `runs/iter_struc/solv_lig_test/dE.csv`

## Conclusion

**It is a solvent (Wat) problem, not a ligand problem.**

- **MeOH:** MAE \|Δ\| ≈ **1.2** kcal/mol — good for all four ligands.
- **Wat:** MAE \|Δ\| ≈ **7.9** kcal/mol — systematically too low (~−7 to −8) for both charged and neutral ligands.

Charged (Im−, MIm) and neutral (ImH, MImH) ligands both look fine in MeOH and both fail in Wat by nearly the same amount. That points to mDFTB3-D3 mis-describing the water / Zn–Wat exchange relative to LCCSD(T)-F12, not a specific ligand parameterization failure.

**Practical takeaway:** run110 is usable for MeOH ligand-exchange trends; Wat absolute ΔE needs a solvent-side correction or a higher-level SP.

## Error summary (run110 − run102 BSSE)

| Solvent | MAE\|Δ\| | Mean Δ (bias) | Max\|Δ\| | n |
| ------- | --------: | -------------: | --------: | -: |
| MeOH    |      1.18 |         −0.53 |      1.86 | 4 |
| Wat     |      7.93 |         −7.93 |      8.30 | 4 |

## Per-reaction comparison

| Ligand | Solvent | run110 | run102 BSSE | Δ (110−102) | \|Δ\| | Verdict |
| ------ | ------- | -----: | ----------: | ------------: | -----: | ------- |
| Im−   | MeOH    | 180.48 |      179.18 |         +1.29 |   1.29 | good    |
| MIm    | MeOH    | 179.51 |      180.33 |        −0.82 |   0.82 | good    |
| MImH   | MeOH    |  22.92 |       24.79 |        −1.86 |   1.86 | good    |
| ImH    | MeOH    |  22.09 |       22.83 |        −0.74 |   0.74 | good    |
| Im−   | Wat     | 181.14 |      189.39 |        −8.25 |   8.25 | poor    |
| MIm    | Wat     | 181.24 |      189.46 |        −8.23 |   8.23 | poor    |
| ImH    | Wat     |  22.43 |       29.35 |        −6.92 |   6.92 | poor    |
| MImH   | Wat     |  23.33 |       31.63 |        −8.30 |   8.30 | poor    |

- **good:** \|Δ\| &lt; 3 kcal/mol
- **poor:** \|Δ\| ≥ 5 kcal/mol

Negative Δ means run110 **underestimates** the ligand-exchange endothermicity (ligand appears less strongly retained vs LCCSD(T)-F12).

## By ligand (does not explain the error)

**ΔE of mDFTB (run110) − Δ E of CCSD(T) (run102 with BSSE correction)** for the ligand-exchange reaction

`1Zn_1L_5Solv + Solv → 1Zn_6Solv + L` (kcal/mol)

| Ligand L |   MeOH |    Wat |
| -------- | -----: | -----: |
| Im−     |  +1.29 | −8.25 |
| MIm      | −0.82 | −8.23 |
| MImH     | −1.86 | −8.30 |
| ImH      | −0.74 | −6.92 |

Same ligand family fails only when the solvent is Wat → **solvent-driven**, not ligand-driven.
