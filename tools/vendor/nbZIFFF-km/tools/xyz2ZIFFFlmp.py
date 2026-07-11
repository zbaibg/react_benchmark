'''
This is for open boundary condition (implemented with a box padding of 999 Angstrom).
'''
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from tools.coord.utils import *
from tools.ZIFFF import *
from tools.utils import *
import argparse
import os
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process ZIF structure files')
    parser.add_argument('-i', '--xyzfile', required=True, help='Input XYZ file')
    parser.add_argument('-d', '--datafile', required=True, help='Output DATA file')
    parser.add_argument('-p', '--pdbfile', required=True, help='Output PDB file')
    args = parser.parse_args()
    assert args.datafile is not None, "Output DATA file is required"
    assert args.xyzfile is not None, "Input XYZ file is required"
    assert os.path.exists(args.xyzfile), f"Input XYZ file {args.xyzfile} does not exist"
    Universe=get_Universe_from_file(args.xyzfile)
    normalize_2char_atomtype(Universe,inplace=True,remove_number=True)#Convert ZN to Zn
    Universe=create_box_for_OPB(Universe,padding=999) # a box is necessary for MDAnalysis LAMMPS DATA file dumps even though you want to use open boundary condition.
    Universe=move_atoms_to_center(Universe)
    #Process ZIF
    Universe=assign_ZIFFF_to_universe(Universe,atom_type_mode='int')
    Universe.atoms.write(args.datafile,file_format="DATA")
    if args.pdbfile:
        Universe.atoms.write(args.pdbfile,file_format="PDB")