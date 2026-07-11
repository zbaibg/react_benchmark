water locations are manually adjusted for 1Zn_5Wat_1Im-_1Wat (./xyz/manual_adjust)
1Im-_1Wat I changed the dlfind setting to BFGS HDLC tol=4.5d-4 tolE=5d-6, because Residue conversion error in DL-Find happened.
1Zn_5Wat_1Im-_1Wat I manually adjust the structure and use BFGS-HDLC from start, because the original structure with the LBFGS-DLC setting lead the Im-..Wat go to zinc coordinated water. 

Here the Ebind are defined to be the complex energy minus its own deformed monomers rather than the relaxed monomers.
Manually set ifqnt to zero for run110.mmsolvent/*monomer_1/min.in and its Wat_monomer/min.in because the workflow cannot automatically do so.