
import MDAnalysis
import os
from .utils import assign_charges, convert_atom_types, convert_topology_types_to_numeric, normalize_2char_atomtype, add_empty_topology_if_none, init_topology, write_universe_to_normalized_pdb_format

fudge_factor_for_bondguess=0.55 # bond_guess_length<fudge_factor*(R1+R2)
vdwradii_for_bondguess={
    "H": 1.10,
    "C": 1.70,
    "N": 1.55,
    "Zn": 1.39*1.6317,#Zn and N forms coordination bond, thus use a larger radius for Zn
    "C1": 1.70,
    "C2": 1.70,
    "C3": 1.70,
    "H2": 1.10,
    "H3": 1.10,
}

atom_type_map = {'C1': 1, 'C2': 2, 'C3': 3, 'H2': 4, 'H3': 5, 'Zn': 6, 'N': 7}


# Update bond types
bond_type_map = {
    ("C3", "H3"): 1,
    ("C1", "C3"): 2,
    ("C1", "N"): 3,
    ("Zn", "N"): 4,
    ("C2", "H2"): 5,
    ("C2", "N"): 6,
    ("C2", "C2"): 7,
}

# Update angle types
angle_type_map = {
    ("C3", "C1", "N"): 1,
    ("C1", "N", "C2"): 2,
    ("N", "C1", "N"): 3,
    ("N", "C2", "C2"): 4,
    ("N", "C2", "H2"): 5,
    ("C2", "C2", "H2"): 6,
    ("Zn", "N", "C2"): 7,
    ("Zn", "N", "C1"): 8,
    ("N", "Zn", "N"): 9,
    ("C1", "C3", "H3"): 10,
    ("H3", "C3", "H3"): 11,
}

# Update dihedral types
dihedral_type_map = {
    ("H2", "C2", "C2", "N"): 1,
    ("H3", "C3", "C1", "N"): 2,
    ("H2", "C2", "N", "Zn"): 3,
    ("C2", "N", "C1", "C3"): 4,
    ("C2", "N", "C1", "N"): 5,
    ("H2", "C2", "C2", "H2"): 6,
    ("N", "C2", "C2", "N"): 7,
    ("C2", "C2", "N", "Zn"): 8,
    ("C2", "N", "Zn", "N"): 9,
    ("C1", "N", "C2", "H2"): 10,
    ("C1", "N", "C2", "C2"): 11,
    ("C3", "C1", "N", "Zn"): 12,
    ("Zn", "N", "C1", "N"): 13,
    ("C1", "N", "Zn", "N"): 14,
}

# Update improper dihedral types
#improper_type_map = {
#    ("C2", "N", "H2", "C2"): 1,
#    ("C1", "C3", "N", "N"): 2,
#    ("N", "Zn", "C1", "C2"): 3,
#}
improper_type_map = {
    ( "N","C2", "H2", "C2"): 1,
    ( "C3","C1", "N", "N"): 2,
    ("Zn", "N", "C2", "C1"): 3,
}

charges = {
        "C1": 0.4375,
        "C2": -0.0662,
        "C3": -0.4606,
        "H2": 0.1141,
        "H3": 0.1381,
        "N": -0.4203,
        "Zn": 0.7072
    }

def find_improper_central_atom(improper):
    """Find the central atom in an improper dihedral.
    
    The central atom in an improper dihedral is the one that is bonded to all other three atoms.
    If you use MDAnalysis guess_improper_dihedrals function to guess the impropers, this atom should be either the first or last in the 4-tuple.
    
    Parameters
    ----------
    improper : MDAnalysis.Improper
        The improper dihedral to analyze
        
    Returns
    -------
    MDAnalysis.Atom
        The central atom of the improper dihedral
        
    Raises
    ------
    AssertionError
        If no single central atom is found
    """
    # Find atoms that are bonded to all other three atoms
    central_atoms = []
    for potential_center in improper.atoms:
        other_atoms = set(improper.atoms) - {potential_center}
        if other_atoms.issubset(set(potential_center.bonded_atoms)):
            central_atoms.append(potential_center)
            
    # Verify we found exactly one central atom
    if len(central_atoms) != 1:
        raise AssertionError(
            f"Found {len(central_atoms)} atoms bonded to all others in improper {improper.indices}. Expected 1."
        )
        
    central_atom = central_atoms[0]
    
    return central_atom
def find_improper_common_egde_atoms(improper):
    """Find the atoms that are on the common edge of an improper dihedral.
    """
    return improper.atoms[1],improper.atoms[2]
def sort_atoms_by_type_order(atoms: MDAnalysis.AtomGroup, type_order: list[str]) -> MDAnalysis.AtomGroup:
    """Sort atom objects according to a specified type order.
    
    Parameters
    ----------
    atoms : list
        List of MDAnalysis Atom objects to be sorted
    type_order : list or tuple 
        Desired order of atom types
        
    Returns
    -------
    MDAnalysis.AtomGroup
        AtomGroup with atoms sorted according to type_order
        
    Raises
    ------
    AssertionError
        If number of atoms doesn't match type_order length
        If atom types don't match required types
        If there are duplicate atoms
    """
    # Verify input lengths match
    assert len(atoms) == len(type_order), \
        f"Number of atoms ({len(atoms)}) does not match length of type order ({len(type_order)})"
    
    # Get atom types and verify they match required types
    atom_types = [atom.type for atom in atoms]
    assert set(atom_types) == set(type_order), \
        f"Atom types {set(atom_types)} do not match required types {set(type_order)}"
    
    # Check for duplicates
    assert len(set(atoms)) == len(atoms), \
        f"There are duplicate atom indices in {atoms}"

    # Sort atoms according to type order
    remaining_atoms = list(atoms)
    sorted_atoms = []
    for target_type in type_order:
        for atom in remaining_atoms:
            if atom.type == target_type:
                sorted_atoms.append(atom)
                remaining_atoms.remove(atom)
                break
                
    return MDAnalysis.AtomGroup(sorted_atoms)

def classify_atoms_ZIFFF(universe: MDAnalysis.Universe) -> MDAnalysis.Universe:
    """Make a initial guess of bonds for classification of atom types.
    Takes a universe with basic atom types (C, H, N, Zn) and classifies atoms based on bonding:
    
    - Carbon atoms are classified as:
        C1: Connected to 2 nitrogens and 1 carbon
        C2: Connected to 1 nitrogen, 1 carbon, and 1 hydrogen (aromatic)
        C3: Connected to 1 carbon and 3 hydrogens (methyl)
    - Hydrogen atoms are classified as:
        H2: Bonded to aromatic carbon (C2)
        H3: Bonded to methyl carbon (C3)
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing atoms to classify
        
    Returns
    -------
    MDAnalysis.Universe
        New universe with classified atom types
    """
    # Work on a copy to preserve original
    classified_universe = universe.copy()
    # Normalize atom types
    normalize_2char_atomtype(classified_universe,inplace=True)
    allowed_types = set(['C', 'H', 'N', 'Zn'])
    assert set(classified_universe.atoms.types).issubset(allowed_types),f"Atom types {set(classified_universe.atoms.types)} is not a subset of required types {allowed_types}"
    # make a initial guess of bonds for classification of atom types
    for topology in ["bonds", "angles", "dihedrals", "impropers"]:
        if hasattr(classified_universe, topology):
            getattr(classified_universe, f"delete_{topology}")(getattr(classified_universe, topology))
    classified_universe.atoms.guess_bonds(
        vdwradii=vdwradii_for_bondguess, fudge_factor=fudge_factor_for_bondguess
    )
    # Process carbon atoms
    for carbon in classified_universe.select_atoms("type C"):
        bonded = carbon.bonded_atoms
        # Count bonded atoms by type
        num_hydrogens = sum(1 for atom in bonded if atom.type == "H")
        num_nitrogens = sum(1 for atom in bonded if atom.type == "N")
        num_carbons = sum(1 for atom in bonded if atom.type in ["C", "C1", "C2", "C3"])
        
        # Classify based on bonding pattern
        if num_hydrogens == 3 and num_carbons == 1 and num_nitrogens == 0:
            carbon.type = "C3"  # Methyl group carbon
        elif num_nitrogens == 2 and num_carbons == 1 and num_hydrogens == 0:
            carbon.type = "C1"  # Carbon between nitrogens
        elif num_hydrogens == 1 and num_nitrogens == 1 and num_carbons == 1:
            carbon.type = "C2"  # Aromatic ring carbon
        else:
            raise ValueError(
                f"Carbon atom {carbon.index} has invalid bonding: "
                f"{num_hydrogens}H, {num_nitrogens}N, {num_carbons}C"
            )
    
    # Process hydrogen atoms
    for hydrogen in classified_universe.select_atoms("type H"):
        bonded = hydrogen.bonded_atoms
        
        # Verify hydrogen has exactly one bond
        if len(bonded) != 1:
            raise ValueError(
                f"Hydrogen atom {hydrogen.index} has {len(bonded)} bonds, "
                f"expected 1. Bonds: {bonded.indices}"
            )
        
        # Get the carbon it's bonded to
        carbon = bonded[0]
        if carbon.type not in ["C2", "C3"]:
            raise ValueError(
                f"Hydrogen atom {hydrogen.index} is bonded to {carbon.type}, "
                f"expected C2 or C3. Bond: {carbon.index}"
            )
            
        # Classify based on carbon type
        hydrogen.type = "H2" if carbon.type == "C2" else "H3"
    assert set(classified_universe.atoms.types).issubset(set(atom_type_map.keys())),f"Atom types {set(classified_universe.atoms.types)} is not a subset of required types {set(atom_type_map.keys())}"
    return classified_universe
def build_topology_ZIFFF(universe: MDAnalysis.Universe) -> None:
    """
    Build filter the topology objects for ZIFFF. Then convert topology types to numeric types
    """
    # rebuild all topology objects and filter wanted topology objects
    # rebuild all topology objects and filter wanted topology objects
    new_u=universe.copy()
    new_u=init_topology(new_u,vdwradii_for_bondguess,fudge_factor_for_bondguess,filter=True,bond_type_map=bond_type_map,angle_type_map=angle_type_map,dihedral_type_map=dihedral_type_map,improper_type_map=improper_type_map)
    new_u=convert_topology_types_to_numeric(new_u,bond_type_map,angle_type_map,dihedral_type_map,improper_type_map)
    # Further filter wanted impropers
    keep_half_C3C1NN_improper_for_ZIFFF(new_u)
    keep_impropers_common_egde_on_bonds(new_u)
    return new_u
def keep_half_C3C1NN_improper_for_ZIFFF(universe: MDAnalysis.Universe) -> None:
    """Keep half amount of C1-C3-N-N improper dihedrals.
    This is specific for ZIFFF, because MDanalysis could identify two C1-C3-N-N improper dihedrals for one MeIm ligand,
    but in the forcefield, only one of them is keeped (although it is asymmetric)
    """
    for c1 in universe.select_atoms("type C1"):
        num=0
        bonded_atoms=c1.bonded_atoms
        assert len(bonded_atoms)==3, f"C1 atom {c1.index} has {len(bonded_atoms)} bonded atoms, expected 3"
        C3_C1_N_N_impropers=[]
        for improper in c1.impropers:
            if set(improper.atoms)==set(bonded_atoms+c1):
                C3_C1_N_N_impropers.append(improper)
        assert len(C3_C1_N_N_impropers)==2, f"C1 atom {c1.index} has {len(C3_C1_N_N_impropers)} C3-C1-N-N impropers, expected 2"
        universe.delete_impropers([C3_C1_N_N_impropers[0]])
def keep_impropers_common_egde_on_bonds(universe: MDAnalysis.Universe) -> None:
    """Keep half amount of impropers with common edge on bonds.
    This is specific for ZIFFF. As there are two possible impropers ( "N","C2", "H2", "C2"), 
    one of which has a common edge on bonds, and the other has a common edge not on bonds. We only want the former one.
    """
    for improper in universe.impropers:
        if improper.atoms[1] not in improper.atoms[2].bonded_atoms:
            universe.delete_impropers([improper])
def check_topology_numbers_for_ZIFFF(universe: MDAnalysis.Universe) -> None:
    """Check the number of topology elements for ZIFFF.
    """
    Zn_num=len(universe.select_atoms("type Zn"))
    # Check that the number of bonds is correct: 4 bonds per Zn, 2 ligands for each Zn, 11 bonds per ligand
    assert len(universe.bonds) == Zn_num*4+Zn_num*2*11,f"Zn_num={Zn_num}, len(universe.bonds)={len(universe.bonds)}"
    # Check that the number of angles is correct: 6 angles per Zn, 3 angles per N, 3 angles per C2 or C1, 6 angles per C3
    assert len(universe.angles) == Zn_num*(6+2*(2*3+3*3+1*6)),f"Zn_num={Zn_num}, len(universe.angles)={len(universe.angles)}"
    # Check that the number of dihedrals is correct: 6 per ZN-N,C1-C3, 4 per N-C1,C2-C2,N-C2,
    # (4 Zn-N, 2*5 N-C1,C2-C2,N-C2, 2*1 C1-C3) per Zn
    assert len(universe.dihedrals) == Zn_num*(6*6+10*4),f"Zn_num={Zn_num}, len(universe.dihedrals)={len(universe.dihedrals)}"
    # Check that the number of impropers is correct: 5 impropers per ligand
    assert len(universe.impropers) == Zn_num*2*5,f"Zn_num={Zn_num}, len(universe.impropers)={len(universe.impropers)}"

def adapt_to_ZIFFF(Universe: MDAnalysis.Universe,output_folder: str):
    '''Convert a VASP CONTCAR file to ZIFFF LAMMPS data format (ZIFFF.data).
    Also generates two PDB files for visualization:
    - ZIFFF.pdb: Contains atoms classified by ZIFFF atom types
    
    - structure.pdb: Contains atoms with original atom types (unclassified)
    
    Parameters:
    ----------
    Universe : MDAnalysis.Universe
        MDAnalysis Universe containing the structure.
        It should be a universe with basic atom types (C, H, N, Zn)
    output_folder : str
        Path to output directory.
        Output files include:
        - ZIFFF.data: LAMMPS data file
        - ZIFFF.pdb: PDB file with ZIFFF atom types and topologies
        - structure.pdb: PDB file with original atom types for visualization
        
    Notes
    -----
    The function performs the following steps:
    1. Creates output directory if it doesn't exist
    2. Writes structure.pdb with original atom types for visualization
    3. Guesses topology based on atom radii, classifies atoms into ZIFFF types, 
    4. Removes extra topology objects which are not in the forcefield, 
    5. Converts topology types to numeric indices.
    6. Keeps only half of C1-C3-N-N improper dihedrals as designed in ZIFFF
    7. Assigns charges according to ZIFFF
    8. Writes ZIFFF.data, ZIFFF.pdb
    9. Validates that number of bonds, angles, dihedrals and impropers match ZIFFF requirements (This is only for bulk structure)
    '''
    u_new=Universe.copy()
    add_empty_topology_if_none(u_new)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    #Write structure.pdb with original atom types for visualization
    write_universe_to_normalized_pdb_format(u_new,os.path.join(output_folder, 'structure.pdb'))
    #Classify atoms
    u_new=classify_atoms_ZIFFF(u_new)
    #Build topology
    u_new=build_topology_ZIFFF(u_new)
    #Charge
    assign_charges(u_new,charges)
    #Output lammps data file
    u_new=convert_atom_types(u_new, "int",atom_type_map)
    u_new.atoms.write(os.path.join(output_folder, 'ZIFFF.data'),file_format="DATA")
    #Output ZIFFF.pdb file    
    u_new=convert_atom_types(u_new, "str",atom_type_map)
    write_universe_to_normalized_pdb_format(u_new,os.path.join(output_folder, 'ZIFFF.pdb'))
    #Check topology element numbers
    check_topology_numbers_for_ZIFFF(u_new)
    
def assign_ZIFFF_to_universe(Universe: MDAnalysis.Universe, atom_type_mode:str='int'):
    """Assign ZIFFF to a universe. atom_type_mode can be 'int' or 'str'. The atom_type_map is used to convert the atom types to ZIFFF types to either int or str.
    """
    u_new=Universe.copy()
    #Classify atoms
    u_new=classify_atoms_ZIFFF(u_new)
    #Build topology
    u_new=build_topology_ZIFFF(u_new)
    #Charge
    assign_charges(u_new,charges)
    #Output lammps data file
    if atom_type_mode=='int':
        u_new=convert_atom_types(u_new, "int",atom_type_map)
        return u_new
    elif atom_type_mode=='str':
        return u_new
    else:
        raise ValueError(f"Invalid atom_type_mode: {atom_type_mode}")

