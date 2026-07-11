import argparse
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / 'vendor' / 'nbZIFFF-km'))
import tools.xyz2ZIFFFlmp as xyz2ZIFFFlmp
import MDAnalysis as mda
import numpy as np
import tempfile
import os

# Covalent alcohol O–H is ~0.96 Å; H-bonds / close contacts guessed as bonds are longer.
_MAX_COVALENT_OH_DIST = 1.10

def remove_xp_atoms(filepath: str) -> str:
    '''
    Remove XP dummy atoms from OPC water model in XYZ file.
    Returns path to cleaned file (temporary if XP atoms were found, original otherwise).
    '''
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Check if it's an XYZ file format
    if not lines:
        return filepath
    
    # For XYZ format: first line is atom count, second line is comment
    try:
        natoms = int(lines[0].strip())
    except (ValueError, IndexError):
        # Not a standard XYZ format, return original
        return filepath
    
    # Filter out XP atoms from the coordinate lines
    header = lines[:2]
    coord_lines = lines[2:2+natoms]
    
    filtered_coords = []
    xp_count = 0
    for line in coord_lines:
        parts = line.split()
        if parts and parts[0].upper() == 'XP':
            xp_count += 1
            continue
        filtered_coords.append(line)
    
    # If no XP atoms found, return original file
    if xp_count == 0:
        return filepath
    
    # Create temporary file with filtered coordinates
    new_natoms = natoms - xp_count
    temp_fd, temp_path = tempfile.mkstemp(suffix='.xyz', text=True)
    with os.fdopen(temp_fd, 'w') as f:
        f.write(f"{new_natoms}\n")
        f.write(header[1])  # Keep original comment line
        f.writelines(filtered_coords)
    
    print(f"Removed {xp_count} XP dummy atoms from {filepath}")
    return temp_path

def xyz_to_mda(
    filepath: str,
    expand_nh_oh_radius: bool = False,
    delete_wrong_bonds: bool = False,
):
    '''
    Assign residue name and atom name to the atoms in the universe.
    The input file is a xyz file or pdb file path.
    The output MDAnalysis universe with residue name and atom name assigned.
    The residue name is MIM, IM-, MOH, NO3, H3O, ZN, WAT, etc.
    Note: the Zn related bonds are not assigned.

    delete_wrong_bonds: if True, after guess_bonds remove (1) all H–H bonds and
        (2) O–H bonds longer than _MAX_COVALENT_OH_DIST (spurious H-bond contacts).
    '''
    # Preprocess to remove XP dummy atoms if present
    cleaned_filepath = remove_xp_atoms(filepath)
    temp_file_created = (cleaned_filepath != filepath)
    
    try:
        u=mda.Universe(cleaned_filepath)
        if not hasattr(u.atoms, 'resnames'):
            u.add_TopologyAttr('resnames')
        vdwradii_for_bondguess=xyz2ZIFFFlmp.vdwradii_for_bondguess.copy()
        vdwradii_for_bondguess['Zn']=-999 # set it to a small value to cluster Zn as a seperate residue
        vdwradii_for_bondguess['ZN']=-999 # set it to a small value to cluster Zn as a seperate residue

        # Optional: increase vdW radii for H, N, and O to allow N–H and O–H covalent bonds
        # to be detected up to ~2 Å with MDAnalysis' bond guessing heuristic:
        #   d < fudge_factor * (R1 + R2), with fudge_factor ≈ 0.55.
        # Using the base radii R_N = 1.55 and R_H = 1.10, a uniform scaling factor
        #   s ≈ 1.37
        # gives:
        #   N–H: 0.55 * s * (1.55 + 1.10) ≈ 2.0 Å
        # and similarly extends O–H when O has a radius comparable to 1.52.
        if expand_nh_oh_radius:
            scale = 1.37
            for elem in ('H', 'H2', 'H3', 'N', 'O', 'O3'):
                if elem in vdwradii_for_bondguess:
                    vdwradii_for_bondguess[elem] *= scale

        u.atoms.guess_bonds(vdwradii=vdwradii_for_bondguess)

        if delete_wrong_bonds:
            # Heuristic bond guessing can add spurious H–H edges when hydrogens are
            # close (e.g. between separate ligands); remove them before fragmenting.
            hh_bonds = [
                b for b in u.bonds
                if b[0].element == 'H' and b[1].element == 'H'
            ]
            if hh_bonds:
                u.delete_bonds(hh_bonds)

            # Spurious O–H from hydrogen-bond-like contacts (e.g. ligand N–H ··· O–MeOH).
            oh_spurious = []
            for b in u.bonds:
                a1, a2 = b[0], b[1]
                e1, e2 = a1.element, a2.element
                if (e1 == 'O' and e2 == 'H') or (e1 == 'H' and e2 == 'O'):
                    d = float(np.linalg.norm(a1.position - a2.position))
                    if d > _MAX_COVALENT_OH_DIST:
                        oh_spurious.append(b)
            if oh_spurious:
                u.delete_bonds(oh_spurious)

        u.atoms.fragments
        u.add_TopologyAttr('name')

        for r,frag in enumerate(u.atoms.fragments):

            if len(frag.atoms) == 1:
                element = frag.atoms[0].element
                if element in ['Zn', 'ZN']:
                    frag.atoms[0].name = 'ZN'
                    resname = 'ZN'
                elif element in ['H']:
                    frag.atoms[0].name = 'H1'
                    resname = 'H'
                elif element in ['O']:
                    frag.atoms[0].name = 'O1'
                    resname = 'O'
                else:
                    raise ValueError(f"Unknown single-atom fragment element: {element}")
            elif len(frag.atoms) == 2:
                elements = sorted([atom.element for atom in frag.atoms])
                if elements == ['H', 'O']:
                    for atom in frag.atoms:
                        if atom.element == 'O':
                            atom.name = 'O1'
                        elif atom.element == 'H':
                            atom.name = 'H1'
                    resname = 'OH'
                else:
                    raise ValueError(f"Unknown 2-atom fragment elements: {elements}")
            elif len(frag.atoms) == 6:
                frag.atoms.select_atoms('element C')[0].name='C1'
                frag.atoms.select_atoms('element O')[0].name='O1'
                HO = frag.atoms.select_atoms('element H and bonded element O')
                assert len(HO) >= 1
                HO_atom = HO[0]
                HO_atom.name='HO1'
                Hc = [
                    h for h in frag.atoms.select_atoms('element H and bonded element C')
                    if h.index != HO_atom.index
                ]
                assert len(Hc) == 3
                for i,h in enumerate(Hc):
                    h.name=f'HC{i+1}'
                resname='MOH'
            elif len(frag.atoms) == 7:
                frag.atoms.select_atoms('element C')[0].name='C1'
                frag.atoms.select_atoms('element O')[0].name='O1'
                HO=frag.atoms.select_atoms('element H and bonded element O')
                assert len(HO) == 2
                HO[0].name='HO1'
                HO[1].name='HO2'
                ho_ids = {HO[0].index, HO[1].index}
                Hc = [
                    h for h in frag.atoms.select_atoms('element H and bonded element C')
                    if h.index not in ho_ids
                ]
                assert len(Hc) == 3
                for i,h in enumerate(Hc):
                    h.name=f'HC{i+1}'
                resname='MO+'
            elif len(frag.atoms) == 11:
                N=frag.atoms.select_atoms('element N')
                for i,n in enumerate(N):
                    n.name=f'N{i+1}'
                C1=frag.atoms.select_atoms('element C and (bonded name N1) and (bonded element H)')
                assert len(C1) == 1
                C1[0].name='C1'
                C2=frag.atoms.select_atoms('element C and (bonded name N2) and (bonded element H)')
                assert len(C2) == 1
                C2[0].name='C2'
                C3=frag.atoms.select_atoms('element C and (bonded name N1) and (bonded name N2)')
                assert len(C3) == 1
                C3[0].name='C3'
                C4=frag.atoms.select_atoms('element C and (bonded name C3)')
                assert len(C4) == 1
                C4[0].name='C4'
                HC4=frag.atoms.select_atoms('element H and bonded name C4')
                assert len(HC4) == 3
                for i,h in enumerate(HC4):
                    h.name=f'H{i+1}'
                HC1=frag.atoms.select_atoms('element H and bonded name C1')
                assert len(HC1) == 1
                HC1[0].name='H4'
                HC2=frag.atoms.select_atoms('element H and bonded name C2')
                assert len(HC2) == 1
                HC2[0].name='H5'
                resname='MIM'
            elif len(frag.atoms) == 8:
                # Deprotonated imidazole; keep atom names aligned with IMH where possible.
                N=frag.atoms.select_atoms('element N')
                assert len(N) == 2
                N[0].name='N1'
                N[1].name='N2'
                C1=frag.atoms.select_atoms('element C and (bonded name N1) and (bonded element C)')
                assert len(C1) == 1
                C1[0].name='C1'
                C2=frag.atoms.select_atoms('element C and (bonded name N2) and (bonded element C)')
                assert len(C2) == 1
                C2[0].name='C2'
                C3=frag.atoms.select_atoms('element C and (bonded name N1) and (bonded name N2)')
                assert len(C3) == 1
                C3[0].name='C3'
                HC3=frag.atoms.select_atoms('element H and bonded name C3')
                assert len(HC3) == 1
                HC3[0].name='H6'
                HC1=frag.atoms.select_atoms('element H and bonded name C1')
                assert len(HC1) == 1
                HC1[0].name='H4'
                HC2=frag.atoms.select_atoms('element H and bonded name C2')
                assert len(HC2) == 1
                HC2[0].name='H5'
                resname='IM-'
            elif len(frag.atoms) == 9:
                # imidazole molecule, just for test purpose, try to label the atom names as in MIM as much as possible
                N=frag.atoms.select_atoms('element N')
                N1=frag.atoms.select_atoms('element N and (bonded element H)')
                N2=N-N1
                assert len(N1) == 1
                assert len(N2) == 1
                N1[0].name='N1'
                N2[0].name='N2'
                C1=frag.atoms.select_atoms('element C and (bonded name N1) and (bonded element C)')
                assert len(C1) == 1
                C1[0].name='C1'
                C2=frag.atoms.select_atoms('element C and (bonded name N2) and (bonded element C)')
                assert len(C2) == 1
                C2[0].name='C2'
                C3=frag.atoms.select_atoms('element C and (bonded name N1) and (bonded name N2)')
                assert len(C3) == 1
                C3[0].name='C3'
                HC3=frag.atoms.select_atoms('element H and bonded name C3')
                assert len(HC3) == 1
                HC3[0].name='H6'
                HC1=frag.atoms.select_atoms('element H and bonded name C1')
                assert len(HC1) == 1
                HC1[0].name='H4'
                HC2=frag.atoms.select_atoms('element H and bonded name C2')
                assert len(HC2) == 1
                HC2[0].name='H5'
                HN1=frag.atoms.select_atoms('element H and bonded name N1')
                assert len(HN1) == 1
                HN1[0].name='HN1'
                resname='IMH'
            elif len(frag.atoms) == 12:
                # 2-methyl-imidazole molecule, just for test purpose, try to label the atom names as in MIM as much as possible
                NH=frag.atoms.select_atoms('element N and (bonded element H)')
                assert len(NH) == 1
                NH[0].name='N1'
                N2=frag.atoms.select_atoms('element N and not (bonded element H)')
                assert len(N2) == 1
                N2[0].name='N2'
                C1=frag.atoms.select_atoms('element C and (bonded name N1) and (bonded element H)')
                assert len(C1) == 1
                C1[0].name='C1'
                C2=frag.atoms.select_atoms('element C and (bonded name N2) and (bonded element H)')
                assert len(C2) == 1
                C2[0].name='C2'
                C3=frag.atoms.select_atoms('element C and (bonded name N1) and (bonded name N2)')
                assert len(C3) == 1
                C3[0].name='C3'
                C4=frag.atoms.select_atoms('element C and (bonded name C3)')
                assert len(C4) == 1
                C4[0].name='C4'
                HC4=frag.atoms.select_atoms('element H and bonded name C4')
                assert len(HC4) == 3
                for i,h in enumerate(HC4):
                    h.name=f'H{i+1}'
                HC1=frag.atoms.select_atoms('element H and bonded name C1')
                assert len(HC1) == 1
                HC1[0].name='H4'
                HC2=frag.atoms.select_atoms('element H and bonded name C2')
                assert len(HC2) == 1
                HC2[0].name='H5'
                HN=frag.atoms.select_atoms('element H and (bonded element N)')
                assert len(HN) == 1
                HN[0].name='H6'

                resname='MIH'
            elif len(frag.atoms) == 13:
                # 2-methyl-imidazole molecule, just for test purpose, try to label the atom names as in MIM as much as possible
                NH=frag.atoms.select_atoms('element N and (bonded element H)')
                assert len(NH) == 2
                NH[0].name='N1'
                NH[1].name='N2'
                C1=frag.atoms.select_atoms('element C and (bonded name N1) and (bonded element H)')
                assert len(C1) == 1
                C1[0].name='C1'
                C2=frag.atoms.select_atoms('element C and (bonded name N2) and (bonded element H)')
                assert len(C2) == 1
                C2[0].name='C2'
                C3=frag.atoms.select_atoms('element C and (bonded name N1) and (bonded name N2)')
                assert len(C3) == 1
                C3[0].name='C3'
                C4=frag.atoms.select_atoms('element C and (bonded name C3)')
                assert len(C4) == 1
                C4[0].name='C4'
                HC4=frag.atoms.select_atoms('element H and bonded name C4')
                assert len(HC4) == 3
                for i,h in enumerate(HC4):
                    h.name=f'H{i+1}'
                HC1=frag.atoms.select_atoms('element H and bonded name C1')
                assert len(HC1) == 1
                HC1[0].name='H4'
                HC2=frag.atoms.select_atoms('element H and bonded name C2')
                assert len(HC2) == 1
                HC2[0].name='H5'
                HN1=frag.atoms.select_atoms('element H and (bonded name N1)')
                assert len(HN1) == 1
                HN1[0].name='H6'
                HN2=frag.atoms.select_atoms('element H and (bonded name N2)')
                assert len(HN2) == 1
                HN2[0].name='H7'
                resname='MI+'
            elif len(frag.atoms) == 4:
                n_oxygen = len(frag.atoms.select_atoms('element O'))
                n_nitrogen = len(frag.atoms.select_atoms('element N'))
                n_hydrogen = len(frag.atoms.select_atoms('element H'))
                if n_nitrogen == 1 and n_oxygen == 3:
                    # Nitrate ion
                    for i, O in enumerate(frag.atoms.select_atoms('element O')):
                        O.name = f'O{i+1}'
                    frag.atoms.select_atoms('element N')[0].name = 'N1'
                    resname = 'NO3'
                elif n_oxygen == 1 and n_hydrogen == 3:
                    # Hydronium ion (H3O+)
                    frag.atoms.select_atoms('element O')[0].name = 'O1'
                    H = frag.atoms.select_atoms('element H')
                    for i, h in enumerate(H):
                        h.name = f'H{i+1}'
                    resname = 'H3O'
                else:
                    raise ValueError(
                        f"Unknown 4-atom fragment: O={n_oxygen}, N={n_nitrogen}, H={n_hydrogen}"
                    )
            elif len(frag.atoms) == 3:
                # Water molecule
                assert len(frag.atoms.select_atoms('element O')) == 1
                assert len(frag.atoms.select_atoms('element H')) == 2
                frag.atoms.select_atoms('element O')[0].name='O'
                H = frag.atoms.select_atoms('element H')
                H[0].name='H1'
                H[1].name='H2'
                resname='WAT'
            else:
                raise ValueError(f"Unknown fragment type: atom number {len(frag.atoms)}")
            new_res=u.add_Residue(resname=resname,resid=r+1,resnum=r+1,icode='')
            frag.atoms.residues=new_res
        
        # Because the atoms with the same residue number must be put together in the output pdb file, we need to merge the fragments to reorder the indices.
        # This also removes the default residue containing all atoms.
        res_atoms_list=[mda.Merge(frag.atoms).atoms for frag in u.atoms.fragments]
        u_new=mda.Merge(*res_atoms_list)
        u_new.dimensions=u.dimensions
        return u_new
    finally:
        # Clean up temporary file if it was created
        if temp_file_created and os.path.exists(cleaned_filepath):
            os.unlink(cleaned_filepath)
# backward compatibility
assign_resname_atomname_by_atomnumber=xyz_to_mda

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Assign residue names and atom names for ZIF/methanol structures "
            "from an XYZ or PDB file, then write an MDAnalysis-supported output file."
        )
    )
    parser.add_argument("input_file", help="Input XYZ or PDB file.")
    parser.add_argument("output_file", help="Output file, typically a named PDB file.")
    parser.add_argument(
        "--expand-nh-oh-bond-radius",
        action="store_true",
        help="Expand N/H/O bond-guessing radii to catch long N-H or O-H covalent bonds.",
    )
    parser.add_argument(
        "--delete-wrong-bonds",
        action="store_true",
        help="Delete guessed H-H bonds and long spurious O-H contacts before fragment assignment.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    u = xyz_to_mda(
        args.input_file,
        expand_nh_oh_radius=args.expand_nh_oh_bond_radius,
        delete_wrong_bonds=args.delete_wrong_bonds,
    )
    u.atoms.write(args.output_file)
    print(f"Wrote named structure: {args.output_file}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
