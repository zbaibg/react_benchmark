# run102 BSSE=yes vs run110.tfermi300 — exchange ΔE with MRE split

**Reaction:** `ZnLSolv5 + Solv → ZnSolv6 + L` (kcal/mol)

| Method | Description |
|--------|-------------|
| run102 \| BSSE=yes | Molpro PNO-LCCSD(T)-F12, counterpoise-corrected |
| run110.tfermi300 | mDFTB3-D3 / optimized_run59 (all QM), `tfermi=300` |

Source: `runs/iter_struc/solv_lig_test_MRE/dE.csv`

## Definitions

Complex energies are split as

\[
E(\mathrm{complex}) = E(\mathrm{solute}) + E(\mathrm{solvent}) + I(\mathrm{solvent},\mathrm{solute})
\]

where solute = Zn (+ ligand if present), solvent = the Solv\(_n\) shell, and \(I\) is the interaction (by difference).

For the exchange reaction `ZnLSolv5 + Solv → ZnSolv6 + L`:

| Component | Definition |
|-----------|------------|
| **dE** | \(E(\mathrm{ZnSolv6}) + E(\mathrm{L}) - E(\mathrm{ZnLSolv5}) - E(\mathrm{Solv})\) |
| **dE(solute)** | \(E_\mathrm{solute}(\mathrm{ZnSolv6}) + E(\mathrm{L}) - E_\mathrm{solute}(\mathrm{ZnLSolv5})\) |
| **dE(solvent)** | \(E_\mathrm{solvent}(\mathrm{ZnSolv6}) - E_\mathrm{solvent}(\mathrm{ZnLSolv5}) - E(\mathrm{Solv})\) |
| **dE(I)** | \(I(\mathrm{ZnSolv6}) - I(\mathrm{ZnLSolv5})\) |

Identity:

\[
\mathrm{dE} = \mathrm{dE(solute)} + \mathrm{dE(solvent)} + \mathrm{dE(I)}
\]

Δ below is **run110.tfermi300 − run102 BSSE=yes**.

## Comparison table

Columns: **CCSD(T)** = run102 BSSE=yes; **mDFTB** = run110.tfermi300; **Δ** = mDFTB − CCSD(T).

### Im− / MeOH

`1Zn_1Im-_5MeOH + MeOH → 1Zn_6MeOH + Im-`

| Component | CCSD(T) | mDFTB | Δ |
|-----------|--------:|------:|--:|
| dE | 179.19 | 180.56 | +1.37 |
| dE(solute) | 382.82 | 413.80 | +30.99 |
| dE(solvent) | 7.55 | 6.49 | −1.06 |
| dE(I) | −211.19 | −239.74 | −28.55 |

### Im− / Wat

`1Zn_1Im-_5Wat + Wat → 1Zn_6Wat + Im-`

| Component | CCSD(T) | mDFTB | Δ |
|-----------|--------:|------:|--:|
| dE | 188.82 | 181.18 | −7.64 |
| dE(solute) | 383.06 | 414.41 | +31.35 |
| dE(solvent) | 6.22 | 10.00 | +3.78 |
| dE(I) | −200.46 | −243.23 | −42.77 |

### ImH / MeOH

`1Zn_1ImH_5MeOH + MeOH → 1Zn_6MeOH + ImH`

| Component | CCSD(T) | mDFTB | Δ |
|-----------|--------:|------:|--:|
| dE | 21.97 | 22.11 | +0.13 |
| dE(solute) | 170.21 | 184.20 | +13.99 |
| dE(solvent) | 4.60 | 5.95 | +1.35 |
| dE(I) | −152.84 | −168.04 | −15.20 |

### ImH / Wat

`1Zn_1ImH_5Wat + Wat → 1Zn_6Wat + ImH`

| Component | CCSD(T) | mDFTB | Δ |
|-----------|--------:|------:|--:|
| dE | 28.90 | 22.36 | −6.54 |
| dE(solute) | 170.45 | 185.55 | +15.10 |
| dE(solvent) | 5.26 | 8.84 | +3.58 |
| dE(I) | −146.80 | −172.03 | −25.23 |

## Compact view (total dE only)

| Ligand | Solvent | CCSD(T) | mDFTB | Δ |
|--------|---------|--------:|------:|--:|
| Im− | MeOH | 179.19 | 180.56 | +1.37 |
| Im− | Wat | 188.82 | 181.18 | −7.64 |
| ImH | MeOH | 21.97 | 22.11 | +0.13 |
| ImH | Wat | 28.90 | 22.36 | −6.54 |
