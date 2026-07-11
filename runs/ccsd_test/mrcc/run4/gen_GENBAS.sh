#!/bin/bash

python /home/zbai29/data/qmmm_test/python_scripts/make_mrcc_genbas.py \
  --basis-file /home/zbai29/data/qmmm_test/ORCA_basis/HCNO_cc-pVTZ-F12_Zn_aug-cc-pVTZ-PP-F12 \
  --optri-file /home/zbai29/data/qmmm_test/ORCA_basis/HCNO_cc-pVTZ-F12_OptRI_Zn_aug-cc-pVTZ-PP-F12_OptRI \
  --mp2fit-file /home/zbai29/data/qmmm_test/ORCA_basis/HCNO_cc-pVTZ-F12_MP2Fit_Zn_aug-cc-pVTZ-PP-F12_MP2Fit \
  --jkfit-file /home/zbai29/data/qmmm_test/ORCA_basis/HCNO_cc-pVTZ_JKFit_Zn_cc-pVTZ-PP_JKFit \
  --ecp-file /home/zbai29/data/qmmm_test/ORCA_basis/ECP10MDF_Zn \
  --output GENBAS