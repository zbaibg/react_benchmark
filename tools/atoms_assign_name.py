import sys
import MDAnalysis as mda
import os
import tempfile

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

def xyz_to_mda(filepath: str):
    '''
    Supports only single-atom files from the periodic table. Assigns atom names as uppercase element symbol followed by 1 (e.g., C1, ZN1), and resname as the uppercase element symbol.
    '''
    # Preprocess to remove XP dummy atoms if present
    cleaned_filepath = remove_xp_atoms(filepath)
    temp_file_created = (cleaned_filepath != filepath)

    try:
        u = mda.Universe(cleaned_filepath)
        # Only allow single-atom files
        if len(u.atoms) != 1:
            raise ValueError("Only single-atom input is supported.")
        if not hasattr(u.atoms, 'resnames'):
            u.add_TopologyAttr('resnames')
        if not hasattr(u.atoms, 'name'):
            u.add_TopologyAttr('name')
        atom = u.atoms[0]
        element = atom.element
        if element is None:
            raise ValueError("Atom is missing 'element' information.")
        element_uc = element.upper()
        atom.name = f"{element_uc}1"
        # Assign residue name from Atom.residue.resname
        atom.residue.resname = element_uc
        # Assign a unique residue to this atom
        new_res = u.add_Residue(resname=element_uc, resid=1, resnum=1, icode='')
        atom.residues = new_res

        # Return the Universe (single atom, no merge or split needed)
        return u
    finally:
        # Cleanup
        if temp_file_created and os.path.exists(cleaned_filepath):
            os.unlink(cleaned_filepath)

assign_resname_atomname_by_atomnumber = xyz_to_mda

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python assign_name.py <input_pdb_file> <output_pdb_file>")
        sys.exit(1)
    input_pdb_file = sys.argv[1]
    output_pdb_file = sys.argv[2]
    u = xyz_to_mda(input_pdb_file)
    u.atoms.write(output_pdb_file)
