# DFTB solute SCC convergence tests (run110 Zn+ligand, no solvent shell)

Directory: `runs/iter_struc/solv_lig_test_MRE/dftb_solute_scc_test/`

## Result

| Setting | ImH_Wat (+2) | Im-_MeOH (+1) | ImH_MeOH (+2) |
|---------|--------------|---------------|---------------|
| tfermi=0, BROYDEN | FAIL | (orig FAIL) | (orig FAIL) |
| tfermi=0, DIIS | FAIL | — | — |
| tfermi=0, maxiter=1000 | FAIL | — | — |
| **tfermi=300, BROYDEN** | **OK (-9407.46)** | **OK (-9427.58)** | **OK (-9406.11)** |
| tfermi=300, DIIS | OK (same E) | OK (same E) | — |
| tfermi=1000, BROYDEN | OK (-9409.07) | OK (-9429.93) | — |
| tfermi=3000, BROYDEN | OK (-9415.12) | — | — |

## Conclusion

**`tfermi = 300` is enough** for all three failing bare Zn+ligand solutes.
Changing mixer (DIIS) or raising `maxiter` alone does **not** fix SCC at `tfermi=0`.

## Energy note

Amber `Etot`/`DFTBPLUSESCF` tracks the finite-T electronic free energy, so absolute E
shifts with tfermi. Prefer the smallest working value (`300`) for solute-only retries,
or re-run full/solute/solvent at the same tfermi for strict MRE consistency.
