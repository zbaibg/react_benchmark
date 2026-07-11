#!/usr/bin/env bash
set -euo pipefail
# Resolve repo root and software paths
_SCRIPT_DIR_FOR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_CANDIDATE="$_SCRIPT_DIR_FOR_REPO"
while [ "$_REPO_CANDIDATE" != "/" ] && [ ! -f "$_REPO_CANDIDATE/software.yaml" ]; do
  _REPO_CANDIDATE="$(dirname "$_REPO_CANDIDATE")"
done
if [ ! -f "$_REPO_CANDIDATE/software.yaml" ]; then
  echo "ERROR: could not locate repo root (software.yaml)" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$_REPO_CANDIDATE/tools/repo_env.sh"

# Split multi-frame xyz into xyz_files/<formula>.xyz (formula from comment line).
SRC="${1:-"$REPO_ROOT/runs/iter_struc_NO3/gen_struc_dftbplus/ana/stable_lowE_per_formula.xyz}"
OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/xyz_files"
mkdir -p "$OUTDIR"

awk -v outdir="$OUTDIR" '
/^[0-9]+$/ && NF == 1 {
	n = int($0)
	if (n <= 0) next
	if (getline title <= 0) exit 1
	if (title !~ /formula=/) next
	if (!match(title, /formula=[^[:space:]]+/)) next
	formula = substr(title, RSTART + 8, RLENGTH - 8)
	gsub(/[^A-Za-z0-9_.+-]/, "_", formula)
	outfile = outdir "/" formula ".xyz"
	if ((getline peek < outfile) > 0) {
		close(outfile)
		printf "ERROR: output already exists: %s (formula=%s)\nTitle: %s\n", outfile, formula, title > "/dev/stderr"
		exit 2
	}
	close(outfile)
	printf "%d\n%s\n", n, title > outfile
	for (i = 0; i < n; i++) {
		if (getline <= 0) exit 1
		print $0 > outfile
	}
	close(outfile)
}
' "$SRC"

echo "Wrote frames under ${OUTDIR}"

cp "$REPO_ROOT/runs/lig_exchange/MIMH_PBE_STRUC/qm_minimize/run16/MeOH_monomer/min.xyz $OUTDIR/MeOH_monomer.xyz
cp "$REPO_ROOT/runs/lig_exchange/MIMH_PBE_STRUC/qm_minimize/run16/Zn_monomer/min.xyz $OUTDIR/Zn_monomer.xyz
cat > $OUTDIR/NO3_monomer.xyz <<EOF
4

  O           1.25397506449910      0.19157279908973      0.00003108027085
  O          -0.46107950184386     -1.18176606648537      0.00003108018332
  O          -0.79289490625332      0.99019010628653      0.00003108029671
  N          -0.00000065640193      0.00000316110911      0.00000675924912

EOF