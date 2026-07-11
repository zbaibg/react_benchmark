# react_benchmark

Standalone reaction / QM-MM benchmark repository, originally copied from `../qmmm_test/react_benchmark`.  
This README documents **what was vendored from where** and **what was changed** during the self-containment reorganization.

I also delete some large files to reduce the size of this folder, see qmmm_test back-up files if these are in need:
**/orc_job.densities
**/orc_job.property.txt
**/orc_job.gbw
**/orc_job.loc
**/orc_job.ges
**/orc_job.bas*
**/run.xml* (molpro)
**/(DAO|DFINT|DFINV|DFINVLOC|DOMAIN|DOMAIN_AO|EXIT|F12INTE|FOCK|FOCK_AO|GENBAS|KEYWD|MINP|MOCOEF|OCCUP|OEINT|OEINT_AO|PRINT|S12MATCABS|S12MATold|SCFDENSITIES|SCHOL|SROOT|SROOT_AO|SYMTRA|TEDAT|VARS|VARS_AO|fort.11|fort.18|fort.19|fort.32|fort.55|fort.56|fort.62|iface)


runs/iter_struc/relax_struc2/qm_minimize/run41/waste/1Zn_5MIm_0MImH_0MeOH/largeprint_remove* 
**/gen_struc_dftbplus/*/{work_dir,save_dir}
## Layout

| Path | Role |
|------|------|
| `software.yaml` | **Only** place for machine-local absolute paths (ORCA / Molpro / MRCC / AmberTools / soft SKF / g-xtb / apptainer / condainit / scratch) |
| `tools/` | Shared scripts and path helpers |
| `structures/` | Vendored geometries (monomers, multi-coord xyz, Net_React prmtop/rst, metaD samples) |
| `params/` | Calculation parameters: `ORCA_basis/`, `skf/run{28,34,36,59}/` |
| `runs/` | Benchmark campaigns (former top-level experiment directories) |
| `ana.ipynb` | Analysis notebook (kept at repo root) |

## Usage

```bash
cd runs/M052X_struc
python generate_workflow.py --mode sp_init --runs run0
```

- `runs/*/generate_workflow.py` is a symlink to `tools/generate_workflow.py`.
- Run it from a `runs/<name>/` directory that contains `run_configs.yaml`.
- Shell scripts usually walk upward to find `software.yaml` and `source tools/repo_env.sh`.

Path conventions:

- In-repo files: relative to the repo root, e.g. `params/skf/run59`, `structures/monomers/ImH.xyz`
- Machine software: only in `software.yaml`; `run_configs.yaml` may use aliases such as `soft:3ob-3-1_modify_O`, `soft:gxtb_home`

**Generated trees left untouched on disk** (re-run `generate_workflow` to refresh job scripts):  
`SP_init/`, `SP_opt/`, `qm_minimize/`, `minimize/`, and already-written job scripts such as `ccsd_test/mrcc*/**/gen_GENBAS.sh`.

### Git tracking policy

Repo initialized on `main` with [`.gitignore`](.gitignore) (no commit yet).

**Tracked (for remote):** `tools/`, `software.yaml`, YAML configs, `structures/`, `params/`, campaign `xyz/` and `amber_prep/`, **`qm_minimize/**/min.xyz` only** (and `minimize/**/min.xyz`), campaign-level `**/*.csv`, analysis PNGs, optimized `best_hsd_inputs/`.

**Not tracked:** entire `SP_init/` and `SP_opt/` trees; everything under `qm_minimize/` except `min.xyz` (including job scripts); QM scratch; `**/cpptraj_logs/`; regenerable `GENBAS`; `*.log`; `old/`/`bk/`/`waste*`; `gen_struc_*/<N>/` workers; `path*.xyz`; caches; slurm logs.

---

## 1. What was copied from where (vendoring)

### 1.1 `tools/` — scripts

| Path in this repo | Source |
|-------------------|--------|
| `tools/zif_meoh_assign_name.py` | `qmmm_test/python_scripts/zif_meoh_assign_name.py` (formerly reached via symlink `qmmm_test/zif_meoh_assign_name.py`) |
| `tools/make_mrcc_genbas.py` | `qmmm_test/python_scripts/make_mrcc_genbas.py` |
| `tools/xyz_to_radical_lib.py` | `qmmm_test/python_scripts/xyz_to_radical_lib.py` |
| `tools/gxtb_trim_xyz.py` | `qmmm_test/python_scripts/gxtb_trim_xyz.py` |
| `tools/analyze_imh_water_contacts.py` | `qmmm_test/metaD_ZnImH_SPCE/EndPointCorrection/IMH_WAT_cluster/IMH_WAT_XYZ/analyze_imh_water_contacts.py` |
| `tools/generate_workflow.py` | Original `react_benchmark/generate_workflow.py` (identical copies existed under many subdirs) |
| `tools/analib.py` | Original root `analib.py` |
| `tools/check_qm_minimize.py` | Original root `check_qm_minimize.py` |
| `tools/atoms_assign_name.py` | Original `single_atoms/atoms_assign_name_py_path.py` |
| `tools/vendor/nbZIFFF-km/tools/` | `/home/zbai29/JR/data/nbZIFFF-km/structures/tools/` (full package: `ZIFFF.py`, `xyz2ZIFFFlmp.py`, `nbZIFFF.py`, `utils.py`, `coord/`, etc.; used by `zif_meoh_assign_name` for `vdwradii_for_bondguess`) |

**Created in this repo (not copied):**

| Path | Notes |
|------|-------|
| `tools/paths.py` | Locate `REPO_ROOT`, load `software.yaml`, resolve relative paths / `soft:` aliases |
| `tools/export_software_env.py` | Emit bash `export` lines from `software.yaml` |
| `tools/repo_env.sh` | Sourced from bash/zsh; exports `REPO_ROOT`, `ORCA_PATH`, `AMBER_SH`, etc. |
| `tools/form_charges.py` | **Stub**: upstream `form_charges.py` / `fix_charges.py` was already missing; `amber_prep/prepare.sh` still calls `fix_charges_py_path`, so a no-op stub was added |
| `software.yaml` | Central machine-path config |

### 1.2 `structures/` — geometries

Only files actually referenced by source scripts were copied (not entire external trees).

#### `structures/monomers/`

| File | Source |
|------|--------|
| `ImH.xyz` | `qmmm_test/coord_ImH/SingleNode_PBE_TZVP/0Zn_1ImH_0Wat/cal.xyz` |
| `IM-.xyz` | `qmmm_test/coord_ImH/.../0Zn_1Im-_0Wat/cal.xyz` |
| `Wat.xyz` | `qmmm_test/coord_ImH/.../0Zn_0ImH_1Wat/cal.xyz` |
| `Zn.xyz` | `qmmm_test/coord_ImH/.../1Zn_0ImH_0Wat/cal.xyz` |
| `MImH.xyz` | `qmmm_test/coord_MImH/SingleNode_PBE_TZVP/MImH/cal.xyz` |
| `NO3.xyz` | `JR/data/nbZIFFF-km/coord/SingleNode_PBE_TZVP/0Zn_0MIm_0MeOH_1N/cal.xyz` |
| `MIM.xyz` | `JR/data/nbZIFFF-km/coord/SingleNode_PBE_TZVP/0Zn_1MIm_0MeOH/cal.xyz` |
| `MeOH.xyz` | `JR/data/nbZIFFF-km/coord/SingleNode_PBE_TZVP/0Zn_0MIm_1MeOH/cal.xyz` |
| `Zn_nbZIFFF.xyz` | `JR/data/nbZIFFF-km/coord/SingleNode_PBE_TZVP/1Zn_0MIm_0MeOH/cal.xyz` |

#### `structures/nbZIFFF-km/`

From `JR/data/nbZIFFF-km/coord/` and `data/nbZIFFF-km/coord/`, only xyz files referenced by `xyz/copy*.sh` / prepare scripts (relative subpaths preserved), e.g.:

- `1coord_PBE_TZVP/...`
- `4coord_uff_PBE_TZVP/...`
- `5coord_uff_PBE_TZVP/...` (including some `init_geo.xyz`)
- `6coord_uff_PBE_TZVP/...` (including some `init_geo_run1.xyz`)

#### `structures/Net_React/`

From `qmmm_test/Net_React/run28/routes/...`, the prmtop/rst pairs used by `iter_struc/extra_test/xyz/copy_xyz.sh` (routes 006 / 012 / 902 / 903, etc.).

#### `structures/metaD/`

| File | Source |
|------|--------|
| `IMZW64.xyz` | `qmmm_test/metaD_ZnImH_SPCE/.../IMH_WAT_XYZ/xyz_out/IMZW64.xyz` |
| `2_Cs.xyz` | `qmmm_test/metaD_ZnImH_SPCE/.../WAT_cluster/xyz/xyz_out/2_Cs.xyz` |

### 1.3 `params/` — basis sets and SKF

| Path in this repo | Source |
|-------------------|--------|
| `params/ORCA_basis/` | `qmmm_test/ORCA_basis/` (full directory) |
| `params/skf/run28/` | `qmmm_test/react_benchmark/refit/_repopt/run28/optimized_skf/` |
| `params/skf/run34/` | same for `run34` |
| `params/skf/run36/` | same for `run36` |
| `params/skf/run59/` | same for `run59` |

Note: standard 3ob / m3ob / g-xtb under `~/soft/...` were **not** vendored; they remain absolute entries in `software.yaml`.

### 1.4 Intentionally not vendored

- `~/soft/3ob-3-1*`, `m3ob-test_*`, `g-xtb` (large; treated as local software installs)
- Real upstream `form_charges.py` / `fix_charges.py` (missing; stub used instead)
- Absolute paths inside generated `SP_init` / `qm_minimize` trees (not rewritten; regenerate instead)

---

## 2. Structural and source changes

### 2.1 Layout moves

- All former top-level campaigns (`M052X_struc`, `Hbondtest`, `ccsd_test`, `iter_struc`, `lig_exchange`, …) moved under `runs/`.
- Removed nested `Hbondtest/.git`.
- Root `generate_workflow.py` / `analib.py` / `check_qm_minimize.py` moved into `tools/`; per-run copies replaced with **relative symlinks** to `tools/generate_workflow.py`, `tools/analib.py`, and `amber_prep/xyz_to_radical_lib.py` → `tools/xyz_to_radical_lib.py`.

### 2.2 `tools/generate_workflow.py`

- Removed hardcoded `_PROJECT_ROOT = '/home/zbai29/data/qmmm_test/'`.
- Imports `zif_meoh_assign_name`, `make_mrcc_genbas`, and `paths` from `tools/`.
- Loads `run_configs.yaml` from the **current working directory** (`runs/<name>/`), not from `tools/`.
- After load, resolves `DFTBPLUS_skroot` (including `soft:`), `GXTBHOME` / `gxtb_path`, and `mrcc_*` basis paths; rewrites legacy absolute `ORCA_basis` strings in `orca_template` to `params/ORCA_basis`.
- Emitted sbatch snippets take ORCA / Molpro / MRCC / condainit / scratch from `software.yaml`.
- gxtb trim script path set to `tools/gxtb_trim_xyz.py`.

### 2.3 `runs/*/run_configs.yaml`

- `/home/zbai29/soft/3ob-3-1*`, `m3ob-*` → `soft:3ob-3-1` / `soft:3ob-3-1_modify_O` / `soft:m3ob_Zn` / `soft:m3ob_prime_Zn`
- g-xtb → `soft:gxtb_home` / `soft:gxtb_binary`
- Former `qmmm_test/react_benchmark/refit/_repopt/runXX/optimized_skf` → `params/skf/runXX`
- Former `qmmm_test/ORCA_basis/...` → `params/ORCA_basis/...`

### 2.4 Top-level `prepare_template.sh` / `amber_prep/prepare.sh` / `xyz/copy*.sh`

- Bootstrap: walk up to `software.yaml`, then `source tools/repo_env.sh`.
- Script paths → `$REPO_ROOT/tools/...`
- Geometry paths → `$REPO_ROOT/structures/...`
- Software → `$ORCA_PATH`, `$AMBER_SH`, `$APPTAINER_SIF`, `$CONDAINIT`, `$SCRATCH_ROOT`, etc.
- Cross-run copies that pointed at `qmmm_test/react_benchmark/...` → `$REPO_ROOT/runs/...`
- `fix_charges_py_path` / form-charges calls → `$REPO_ROOT/tools/form_charges.py` (stub)

### 2.5 Analysis / helper Python

Top-level scripts that still had absolute paths (e.g. `Hbondtest/analyze_dE.py`, `iter_struc/scripts/make_id_sbatch.py`, `plot_react_net.py`, `solv_lig_test/xyz/transform_xyz.py`, `check_qm_minimize.py`) were updated to:

- Bootstrap `REPO_ROOT` / `TOOLS_DIR`
- Point external paths at `structures/`, `runs/`, `tools/`, or `software.yaml`

### 2.6 `params/` relocation

- `ORCA_basis` and `skf` were first placed under `structures/`, then moved to `params/` as requested; yaml / `generate_workflow` / README paths were updated accordingly.

---

## 3. Absolute paths you may still see (expected)

Left on purpose or only for compatibility:

1. **`software.yaml`** — machine-local software paths.
2. **Generated job trees** — `SP_init/`, `qm_minimize/`, `minimize/`, `ccsd_test/mrcc*/**/gen_GENBAS.sh`, etc.; re-run `generate_workflow.py` to refresh.
3. **`waste1/` / `waste2/`** — discarded copies; not systematically cleaned.
4. **`tools/generate_workflow.py`** still contains a compatibility rewrite from legacy absolute `qmmm_test/ORCA_basis/` strings to `params/ORCA_basis`.

---

## 4. Dependency sketch

```text
runs/<campaign>/
  ├── run_configs.yaml     ──soft:/relative──►  software.yaml
  │                                          params/  structures/
  ├── generate_workflow.py ──symlink─────────►  tools/generate_workflow.py
  ├── prepare_template.sh  ──source──────────►  tools/repo_env.sh
  └── amber_prep/prepare.sh ──► tools/*.py + structures/monomers/
```

---

## 5. Symlink repair after the move

After relocating this tree out of `qmmm_test`, many absolute / wrong-depth symlinks broke. They were remapped into the new layout as **relative** links (e.g. old `qmmm_test/react_benchmark/X` → `runs/X`, `python_scripts/` → `tools/`, JR monomers → `structures/monomers/`, plus a few structural substitutes such as gas → `lig_exchange` and `1Zn_6Wat_*` → `1ImH_6Wat_*`).

**Geometry `MOL.xyz` / `amber_prep` links** were then checked and, where needed, retargeted so inputs match the job’s own `MOL.pdb` (within PDB 3-decimal precision):

| Stage | Expected target (general jobs) |
|-------|--------------------------------|
| `SP_init` / `SP_opt` | Prefer `qm_minimize/**/min.xyz` if it exists and matches PDB; else `xyz/xyz_files/...` |
| `qm_minimize` | `xyz/xyz_files/...` (not same-dir `min.xyz`) |
| `minimize` | corresponding `qm_minimize/**/min.xyz` |
| `amber_prep` | geometry that matches same-stem `*.pdb` |

**Restart / hand-edit cases** (`old/`, `initial_run/`, `manually_created_*.xyz`, `waste/`, etc.): only require that `MOL.xyz` agrees with the same-directory `MOL.pdb`; same-dir links are allowed there (as in the original tree).

**Final check:** no broken symlinks; general jobs do not point `MOL.xyz` at same-dir files; targets follow the table above and match same-dir PDB; restart/manual exceptions match PDB (and match old_data link text). One known job has no `MOL.pdb` (`lig_exchange/.../1Zn_5MIm_0MeOH`) and was verified against `path.xyz` instead.
