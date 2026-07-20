#!/usr/bin/env bash
mkdir -p xyz_files
for name in MImH_monomer MIm_monomer Zn_monomer MeOH_monomer 1Zn_0MIm_1MeOH 1Zn_1MIm_0MeOH 1Zn_1MImH_0MeOH
do
cp /home/zbai29/data/qmmm_test/react_benchmark/M052X_struc/qm_minimize/run30/$name/min.xyz xyz_files/$name.xyz
done