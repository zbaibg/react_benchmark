import os
from typing import List, Tuple

import MDAnalysis
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

from tools.utils import add_empty_topology_if_none

from .utils import (
    add_block_to_lmpdata,
    assign_charges,
    convert_atom_types,
    convert_topology_types_to_numeric,
    init_topology,
    modify_block_to_lmpdata,
    modify_typenums_in_lmpdata,
    normalize_2char_atomtype,
    write_universe_to_normalized_pdb_format,
)
from . import ZIFFF
vdwradii_for_bondguess = ZIFFF.vdwradii_for_bondguess.copy()
vdwradii_for_bondguess["N1"] = ZIFFF.vdwradii_for_bondguess["N"]
vdwradii_for_bondguess["C8"] = ZIFFF.vdwradii_for_bondguess["C"]
vdwradii_for_bondguess["H5"] = ZIFFF.vdwradii_for_bondguess["H"]
vdwradii_for_bondguess["O"] = 1.52
vdwradii_for_bondguess["O3"] = vdwradii_for_bondguess["O"]
DIS_Zn_He = 0.9
DIS_N1_Ne = 0.5
fudge_factor_for_bondguess = ZIFFF.fudge_factor_for_bondguess
vdwradii_for_bondguess["He"] = max(
    DIS_Zn_He / fudge_factor_for_bondguess - vdwradii_for_bondguess["Zn"] + 0.2, 1.4 
    #For bond number correctness check, I set it larger so that He could be connected to other He near Zn. 
)
vdwradii_for_bondguess["Ne"] = (
    DIS_N1_Ne / fudge_factor_for_bondguess - vdwradii_for_bondguess["N1"] + 0.2
)


atom_type_map = {
    "C8": 1,
    "O3": 2,
    "H5": 3,
    "Zn": 4,
    "He": 5,
    "H3": 6,
    "C2": 7,
    "H2": 8,
    "N1": 9,
    "C1": 10,
    "C3": 11,
    "Ne": 12,
}
# Other topology objects that do not appear in bond_type_map,angle_type_map,dihedral_type_map,improper_type_map
# will be automatically removed after guessing topology
bond_type_map = {
    ("C8", "O3"): 1,
    ("O3", "H5"): 2,
    ("Zn", "He"): 3,
    ("He", "He"): 4,
    ("H3", "C3"): 5,
    ("C2", "H2"): 6,
    ("C2", "N1"): 7,
    ("C2", "C2"): 8,
    ("N1", "C1"): 9,
    ("N1", "Ne"): 10,
    ("C1", "C3"): 11,
}

angle_type_map = {
    ("C8", "O3", "H5"): 1,
    ("He", "Zn", "He"): 2,
    ("Zn", "He", "He"): 3,
    ("He", "He", "He"): 4,
    ("H2", "C2", "N1"): 5,
    ("H2", "C2", "C2"): 6,
    ("N1", "C2", "C2"): 7,
    ("C2", "N1", "C1"): 8,
    ("C2", "N1", "Ne"): 9,
    ("C1", "N1", "Ne"): 10,
    ("N1", "C1", "N1"): 11,
    ("N1", "C1", "C3"): 12,
    ("H3", "C3", "H3"): 13,
    ("H3", "C3", "C1"): 14,
}

dihedral_type_map = {
    # ("Zn", "He", "He", "He"): 1,
    # ("He", "Zn", "He", "He"): 2,
    # ("He", "He", "He", "He"): 3,
    ("H3", "C3", "C1", "N1"): 4,
    ("C2", "N1", "C1", "N1"): 5,
    ("C2", "N1", "C1", "C3"): 6,
    ("C2", "C2", "N1", "C1"): 7,
    ("C2", "C2", "N1", "Ne"): 8,
    ("H2", "C2", "N1", "C1"): 9,
    ("H2", "C2", "N1", "Ne"): 10,
    ("H2", "C2", "C2", "H2"): 11,
    ("H2", "C2", "C2", "N1"): 12,
    ("N1", "C2", "C2", "N1"): 13,
    ("N1", "C1", "N1", "Ne"): 14,
    ("C3", "C1", "N1", "Ne"): 15,
}

improper_type_map = {
    # ("Zn", "He", "He", "He"): 1,
    # ("He", "Zn", "He", "He"): 2,
    # ("He", "He", "He", "He"): 3,
    # ("C2", "H2", "N1", "C2"): 4,
    ("N1", "C2", "C1", "Ne"): 5,
    ("C1", "N1", "N1", "C3"): 6,
    # ("C3", "H3", "H3", "C1"): 7,
    # ("C3", "H3", "H3", "H3"): 8,
}

charges = {
    "C8": 0.2650000,
    "O3": -0.7000000,
    "H5": 0.4350000,
    "Zn": 0.353600,
    "He": 0.088400,
    "H3": 0.138100,
    "C2": -0.066200,
    "H2": 0.114100,
    "N1": 0.000000,
    "C1": 0.437500,
    "C3": -0.460600,
    "Ne": -0.420300,
}

masses = {
    "C8": 12.0107002,
    "O3": 15.9989996,
    "H5": 1.0079401,
    "Zn": 64.3720551,
    "He": 1.0079401,
    "H3": 1.0079401,
    "C2": 12.0107002,
    "H2": 1.0079401,
    "N1": 12.9987593,
    "C1": 12.0107002,
    "C3": 12.0107002,
    "Ne": 1.0079401,
}
# typenum_in_lmpfile is the number of types written in the head of the lmpdata file.
# It should be consistent with the number of types in the topo_coeff_block.
typenum_in_lmpfile = {
    "atom types": len(atom_type_map),
    "bond types": 11,
    "angle types": 14,
    "dihedral types": 15,
    "improper types": 8,
}

masses_block = """Masses
  
   1     12.0107002  # C8  
   2     15.9989996  # O3  
   3      1.0079401  # H5  
   4     64.3720551  # Zn  
   5      1.0079401  # He  
   6      1.0079401  # H3  
   7     12.0107002  # C2  
   8      1.0079401  # H2  
   9     12.9987593  # N1  
  10     12.0107002  # C1  
  11     12.0107002  # C3  
  12      1.0079401  # Ne
"""
topo_coeff_block = """Bond Coeffs
  
   1   12.5592232       1.42999995      # harmonic    # C8  O3  
   2   8.66882277      0.944999993      # harmonic    # O3  H5  
   3   23.4144001      0.899999976      # harmonic    # Zn  He  
   4   23.4144001       1.47000003      # harmonic    # He  He  
   5   13.9519463       1.10200000      # harmonic    # H3  C3  
   6   16.0323601       1.08800006      # harmonic    # C2  H2  
   7   12.5592232       1.38600004      # harmonic    # C2  N1  
   8   17.4445953       1.37699997      # harmonic    # C2  C2  
   9   14.6235933       1.35500002      # harmonic    # N1  C1  
  10   23.4144001      0.500000000      # harmonic    # N1  Ne  
  11   9.78331661       1.49800003      # harmonic    # C1  C3  
  
Angle Coeffs
  
   1   2.3867509    108.5000000      0.0000000      0.0000000  # charmm      # C8  O3  H5  
   2   2.3848000    109.5000000      0.0000000      0.0000000  # charmm      # He  Zn  He  
   3   2.3848000     35.2500000      0.0000000      0.0000000  # charmm      # Zn  He  He  
   4   2.3848000     60.0000000      0.0000000      0.0000000  # charmm      # He  He  He  
   5   1.3693087    121.3170013      0.8858448      2.1600001  # charmm      # H2  C2  N1  
   6   1.5175999    130.0339966      0.6408608      2.2360001  # charmm      # H2  C2  C2  
   7   1.4560288    107.9950027      4.2948079      2.2349999  # charmm      # N1  C2  C2  
   8   2.0088689    106.2519989      4.8411441      2.1930001  # charmm      # C2  N1  C1  
   9   0.4925696    126.9499969      0.0000000      0.0000000  # charmm      # C2  N1  Ne  
  10   0.6256848    126.8499985      0.0000000      0.0000000  # charmm      # C1  N1  Ne  
  11   1.4022623    111.1689987      4.6555634      2.2360001  # charmm      # N1  C1  N1  
  12   1.6940751    124.1969986      1.3272495      2.5220001  # charmm      # N1  C1  C3  
  13   1.1785247    107.7409973      0.8095312      1.7790000  # charmm      # H3  C3  H3  
  14   1.5744016    110.9629974      0.8307776      2.1530001  # charmm      # H3  C3  C1  
  
Dihedral Coeffs
  
   1   none # Zn  He  He  He  
   2   none # He  Zn  He  He  
   3   none # He  He  He  He  
   4   charmm  0.0117072   2 180      0.0000000 # H3  C3  C1  N1  
   5   charmm  0.4669872   2 180      0.0000000 # C2  N1  C1  N1  
   6   charmm  0.1578304   2 180      0.0000000 # C2  N1  C1  C3  
   7   charmm  0.2879104   2 180      0.0000000 # C2  C2  N1  C1  
   8   charmm  0.0613978   2 180      0.0000000 # C2  C2  N1  Ne  
   9   charmm  0.1582640   2 180      0.0000000 # H2  C2  N1  C1  
  10   charmm  0.0457882   2 180      0.0000000 # H2  C2  N1  Ne  
  11   charmm  0.0147424   2 180      0.0000000 # H2  C2  C2  H2  
  12   charmm  0.1539280   2 180      0.0000000 # H2  C2  C2  N1  
  13   charmm  0.6647088   2 180      0.0000000 # N1  C2  C2  N1  
  14   charmm  0.0266230   2 180      0.0000000 # N1  C1  N1  Ne  
  15   charmm  0.0098427   2 180      0.0000000 # C3  C1  N1  Ne  
  
Improper Coeffs
  
   1   none # Zn  He  He  He  
   2   none # He  Zn  He  He  
   3   none # He  He  He  He  
   4   none # C2  H2  N1  C2  
   5   cvff  0.0024282  -1   2 # cvff      # N1  C2  C1  Ne  
   6   cvff  0.1517600  -1   2 # cvff      # C1  N1  N1  C3  
   7   none # C3  H3  H3  C1  
   8   none # C3  H3  H3  H3  
"""


def add_dummy_atoms(universe: MDAnalysis.Universe) -> MDAnalysis.Universe:
    """Add dummy atoms (He, Ne) around Zn and N atoms based on coordination."""
    new_u = universe.copy()
    assert 'He' not in new_u.atoms.types and 'Ne' not in new_u.atoms.types, "Dummy He and Ne atoms should be removed before running guess_topology_and_classify_atoms_nbZIFFF, because the dummy atoms are added in this function"
    normalize_2char_atomtype(new_u, inplace=True)
    def create_dummy_atoms(
        universe: MDAnalysis.Universe,
        position_list: List[np.ndarray],
        atom_type_list: List[str],
    ) -> MDAnalysis.core.groups.Atom:
        assert len(position_list) == len(atom_type_list)
        inserted = MDAnalysis.Universe.empty(
            n_atoms=len(position_list), trajectory=True
        )
        add_empty_topology_if_none(inserted)
        inserted.load_new(np.array([position_list]), order="fac", format=MemoryReader)
        inserted.add_TopologyAttr("type", atom_type_list)
        res = MDAnalysis.Merge(universe.atoms, inserted.atoms)
        res.dimensions = universe.dimensions
        return res

    def unit_vector(vector: np.ndarray) -> np.ndarray:
        return vector / np.linalg.norm(vector)
    def create_tetrahedron(Zn_pos: np.ndarray, bonded_atom1_pos: np.ndarray, bonded_atom2_pos: np.ndarray) -> np.ndarray:
        tet_positions = np.zeros((4,3))
        vec1 = unit_vector(bonded_atom1_pos - Zn_pos)
        vec2 = unit_vector(bonded_atom2_pos - Zn_pos)
        
        # First vertex along first bond
        tet_positions[0] = Zn_pos + vec1 * DIS_Zn_He
        # Second vertex in plane of bonds but 109.47° from first
        normal = np.cross(vec1, vec2)
        normal = unit_vector(normal)
        rot_angle = np.arccos(-1/3)  # Tetrahedral angle ≈ 109.47°
        vec_in_plane = unit_vector(np.cross(normal, vec1))
        vec_in_plane=-vec_in_plane if np.dot(vec_in_plane,vec2)<0 else vec_in_plane
        tet_positions[1] = Zn_pos + (vec1 * np.cos(rot_angle) + vec_in_plane * np.sin(rot_angle)) * DIS_Zn_He
        
        # Rotate around vec1 axis using Rodrigues rotation formula
        rot_axis = vec1
        vec_to_rotate = tet_positions[1] - Zn_pos
        # Create rotation matrix around rot_axis using Rodrigues formula
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                        [rot_axis[2], 0, -rot_axis[0]], 
                        [-rot_axis[1], rot_axis[0], 0]])
        I = np.eye(3)
        # Rotation by 120° (2π/3)
        theta = 2*np.pi/3
        R = I + np.sin(theta)*K + (1-np.cos(theta))*np.matmul(K,K)
        vec3 = np.matmul(R, vec_to_rotate)
        vec4 = np.matmul(R, vec3)
        tet_positions[2] = Zn_pos + vec3
        tet_positions[3] = Zn_pos + vec4
        return tet_positions
    # Add He atoms around Zn
    for zn in new_u.select_atoms("type Zn"):
        # Get bonded atoms to Zn
        bonded_atoms = new_u.atoms.select_atoms(
            f"point {zn.position[0]} {zn.position[1]} {zn.position[2]} 2.5 and not type Zn and not type He and not type Ne"
        )
        if len(bonded_atoms) >= 2:
            tet_positions = create_tetrahedron(zn.position, bonded_atoms[0].position, bonded_atoms[1].position)
        elif len(bonded_atoms) == 1:
            vec = bonded_atoms[0].position - zn.position            
            # Find any vector not collinear with vec
            if vec[0] != 0 or vec[1] != 0:
                vec2 = np.array([-vec[1], vec[0], 0])
            else:
                vec2 = np.array([1, 0, 0])
            tet_positions = create_tetrahedron(zn.position, bonded_atoms[0].position, vec2+zn.position)
        else:
            tet_positions = create_tetrahedron(zn.position, zn.position+np.array([0,0,1]), zn.position+np.array([0,1,0]))
        new_u = create_dummy_atoms(new_u, tet_positions, ["He"] * 4)

    # Add Ne atoms around N
    for n in new_u.select_atoms("type N"):
        c_neighbors = new_u.atoms.select_atoms(
            f"point {n.position[0]} {n.position[1]} {n.position[2]} 1.5 and type C"
        )
        assert len(c_neighbors) == 2
        direction = unit_vector(
            n.position - (c_neighbors[0].position + c_neighbors[1].position) / 2
        )
        dummy_pos = n.position + DIS_N1_Ne * direction
        new_u = create_dummy_atoms(new_u, [dummy_pos], ["Ne"])

    return new_u


def classify_atoms_nbZIFFF(
    universe: MDAnalysis.Universe,
) -> MDAnalysis.Universe:
    """Make a initial guess of bonds for claasification of aotm types.
    Takes a universe with basic atom types (C, H, N, Zn, O, He (dummy), Ne (dummy)) and classifies atoms based on bonding:
    C1 C2 C3 H2 H3 are determined by invoking guess_topology_and_classify_atoms_ZIFFF()
    This function determines the following atom types additionally:
        C8: Only bonded to 1 oxygens and 1 carbon
        H5: Only bonded to carbonyl carbon in MeOH(C8)
        O3: Only bonded to carbonyl carbon in MeOH(C8)
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing atoms to classify

    Returns
    -------
    MDAnalysis.Universe
        New universe with classified atom types and rebuilt topology
    """
    new_u = universe.copy()
    normalize_2char_atomtype(new_u, inplace=True)
    allowed_types = set(['C', 'H', 'N', 'Zn', 'O', 'He', 'Ne'])
    assert set(new_u.atoms.types).issubset(allowed_types), f"Atom types {set(new_u.atoms.types)} is not a subset of required types {allowed_types}"
    # make a initial guess of bonds for classification of atom types
    for topology in ["bonds", "angles", "dihedrals", "impropers"]:
        if hasattr(new_u, topology):
            getattr(new_u, f"delete_{topology}")(getattr(new_u, topology))
    new_u.atoms.guess_bonds(
        vdwradii=vdwradii_for_bondguess, fudge_factor=fudge_factor_for_bondguess
    )
    # Deal with MeOH
    conditions = [
        "type C and bonded type O",
        "type O and bonded type C8",
        "type H and bonded type O3",
    ]
    types = ["C8", "O3", "H5"]
    neighbor_num_list = [1, 2, 1]
    for condition, atomtype, neighbor_num in zip(conditions, types, neighbor_num_list):
        atomgroup = new_u.select_atoms(condition)
        atomgroup.types = [atomtype] * len(atomgroup)
    MeOH_atomgroup = new_u.select_atoms("type C8 or type O3 or type H5")
    # Deal with Dummy atoms
    Dummy_atomgroup = new_u.select_atoms("type He or type Ne")
    # Deal as ZIFFF
    grp=new_u.select_atoms("not (type C8 or type O3 or type H5 or type He or type Ne)")
    if len(grp) > 0:
        ZIFFF_Universe = MDAnalysis.Merge(grp)
        ZIFFF_Universe.dimensions = new_u.dimensions
        ZIFFF_Universe = ZIFFF.classify_atoms_ZIFFF(ZIFFF_Universe)
        ZIFFF_atomgroup = ZIFFF_Universe.atoms
    else:
        ZIFFF_atomgroup = grp  # assign the empty atomgroup to ZIFFF_atomgroup
    # Merge then make some adjustments
    merge_list = []
    for atomgroup in [MeOH_atomgroup, ZIFFF_atomgroup, Dummy_atomgroup]:
        if len(atomgroup) > 0:
            merge_list.append(atomgroup)
    new_u = MDAnalysis.Merge(*merge_list)
    new_u.dimensions = universe.dimensions
    # N->N1
    new_u.atoms.select_atoms("type N").types = "N1"
    assert set(new_u.atoms.types).issubset(
        set(atom_type_map.keys())
    ), f"Atom types {set(new_u.atoms.types)} is not a subset of required types {set(atom_type_map.keys())}"

    
    return new_u
def build_topology_nbZIFFF(universe: MDAnalysis.Universe) -> None:
    """
    Build filter the topology objects for nbZIFFF. Then convert topology types to numeric types
    """
    # rebuild all topology objects and filter wanted topology objects
    new_u=universe.copy()
    new_u=init_topology(new_u,vdwradii_for_bondguess,fudge_factor_for_bondguess,filter=True,bond_type_map=bond_type_map,angle_type_map=angle_type_map,dihedral_type_map=dihedral_type_map,improper_type_map=improper_type_map)
    new_u=convert_topology_types_to_numeric(new_u,bond_type_map,angle_type_map,dihedral_type_map,improper_type_map)
    #nbZIFFF used C1-N1-N1-C3 improper instead of C1-C3-N1-N1. 
    # The former type has only 1 improper per MIm, so no need to keep half of them.
    #ZIFFF.keep_half_C3C1NN_improper_for_ZIFFF(new_u)
    
    #nbZIFFF do not use impropers with common edge on bonds. so no need to keep_impropers_common_egde_on_bonds()
    #ZIFFF.keep_impropers_common_egde_on_bonds(new_u)
    return new_u

def assign_mass_nbZIFFF(universe: MDAnalysis.Universe) -> None:
    if not hasattr(universe.atoms, "mass"):
        universe.add_TopologyAttr("mass")
    for atom in universe.atoms:
        atom.mass = masses[atom.type]

def assign_resnames_nbZIFFF(universe: MDAnalysis.Universe) -> None:
    """This determine the fragments based on topology connection. So it should have a well defined topology before running this function."""
    if hasattr(universe.atoms,'icodes'):
        universe.del_TopologyAttr("icodes")
    if not hasattr(universe.atoms, "resname"):
        universe.add_TopologyAttr("resname")
    if not hasattr(universe.atoms, "resid"):
        universe.add_TopologyAttr("resid")
    for i, frag in enumerate(universe.atoms.fragments):
        if len(frag.atoms) == 5:
            resname = "Zn"
        elif len(frag.atoms) == 3:
            resname = "MeOH"
        elif len(frag.atoms) == 13:
            resname = "MIm"
        else:
            resname = "Other"
        new_residue = universe.add_Residue(
            segment=universe.segments[-1], resname=resname, resid=i + 1, resnum=i + 1
        )
        frag.residues = new_residue


def check_topology_numbers_for_nbZIFFF(universe: MDAnalysis.Universe) -> None:
    """Check the number of topology elements for nbZIFFF."""
    assign_resnames_nbZIFFF(universe)
    Zn_num = len(universe.select_atoms("resname Zn").fragments)
    MeOH_num = len(universe.select_atoms("resname MeOH").fragments)
    MIm_num = len(universe.select_atoms("resname MIm").fragments)
    bond_num = len(universe.bonds) if hasattr(universe, "bonds") else 0
    angle_num = len(universe.angles) if hasattr(universe, "angles") else 0
    dihedral_num = len(universe.dihedrals) if hasattr(universe, "dihedrals") else 0
    improper_num = len(universe.impropers) if hasattr(universe, "impropers") else 0
    error_msg = f"Zn_num={Zn_num}, MeOH_num={MeOH_num}, MIm_num={MIm_num}, bond_num={bond_num}, angle_num={angle_num}, dihedral_num={dihedral_num}, improper_num={improper_num}"
    # The He atoms around one Zn atom are also connected to each other. They have bond and angle forces.
    assert bond_num == Zn_num * 10 + MeOH_num * 2 + MIm_num * 13, error_msg
    assert angle_num == Zn_num * 30 + MeOH_num * 1 + MIm_num * 21, error_msg
    assert dihedral_num == MIm_num * 26, error_msg
    assert improper_num == MIm_num * 3, error_msg


def adapt_to_nbZIFFF(Universe: MDAnalysis.Universe, output_folder: str,outputted_pdb_filename: str='nbZIFFF_vis.pdb',outputted_data_filename: str='nbZIFFF_sorted.data',dummy_atom_existed: bool=False):
    """Convert a Universe with atom types (C, H, N, Zn, O) to nbZIFFF LAMMPS data format (nbZIFFF_sorted.data).
    Also generates two PDB files for visualization:
    - nbZIFFF_vis.pdb: Contains atoms classified by nbZIFFF atom types

    - structure.pdb: Contains atoms with original atom types (unclassified)

    Parameters:
    ----------
    Universe : MDAnalysis.Universe
        MDAnalysis Universe containing the structure.
        It should be a universe with basic atom types (C, H, N, Zn, O), if dummy_atom_existed is True, it could also contain He and Ne.
    output_folder : str
        Path to output directory.
        Output files include:
        - nbZIFFF_sorted.data: LAMMPS data file
        - nbZIFFF_vis.pdb: PDB file with nbZIFFF atom types and topologies
    outputted_pdb_filename : str
        Name of the outputted PDB file. If None, the PDB file will not be outputted.
    outputted_data_filename : str
        Name of the outputted LAMMPS data file. If None, the LAMMPS data file will not be outputted.
    dummy_atom_existed : bool
        Whether dummy atoms are already added in the universe.
    Notes
    -----
    The function performs the following steps:
    1. Creates output directory if it doesn't exist
    2. Adds dummy atoms around Zn and N atoms
    3. Guesses topology based on atom radii, classifies atoms into nbZIFFF types,
    4. Removes extra topology objects which are not in the forcefield,
    5. Converts topology types to numeric indices.
    5. Keeps only half of C1-C3-N-N improper dihedrals as designed in nbZIFFF
    6. Assigns charges according to nbZIFFF
    7. Writes nbZIFFF.data, nbZIFFF.pdb
    8. Validates that number of bonds, angles, dihedrals and impropers match nbZIFFF requirements
    """
    u_new = Universe.copy()
    add_empty_topology_if_none(u_new)
    allowed_types = set(['C', 'H', 'N', 'Zn', 'O']) if not dummy_atom_existed else set(['C', 'H', 'N', 'Zn', 'O', 'He', 'Ne'])
    assert set(u_new.atoms.types).issubset(allowed_types),f"Atom types {set(u_new.atoms.types)} is not a subset of required types {allowed_types}"
    normalize_2char_atomtype(u_new, inplace=True)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # Add dummy atoms
    if not dummy_atom_existed:
        u_new = add_dummy_atoms(u_new)
    # Classify atoms
    u_new = classify_atoms_nbZIFFF(u_new)
    # Build topology
    u_new = build_topology_nbZIFFF(u_new)
    # Charge
    assign_charges(u_new,charges)
    # Check topology element numbers
    check_topology_numbers_for_nbZIFFF(u_new)
    # Output lammps data file
    if outputted_data_filename is not None:
        u_new = convert_atom_types(u_new, "int", atom_type_map)
        u_new.atoms.write(
            os.path.join(output_folder, outputted_data_filename), file_format="DATA"
        )
        modify_block_to_lmpdata(
            os.path.join(output_folder, outputted_data_filename),
            masses_block,
            section_name="Masses",
        )
        add_block_to_lmpdata(
            os.path.join(output_folder, "nbZIFFF_sorted.data"), topo_coeff_block
        )
        modify_typenums_in_lmpdata(
            os.path.join(output_folder, "nbZIFFF_sorted.data"),
            atom_type_num=typenum_in_lmpfile["atom types"],
            bond_type_num=typenum_in_lmpfile["bond types"],
            angle_type_num=typenum_in_lmpfile["angle types"],
            dihedral_type_num=typenum_in_lmpfile["dihedral types"],
            improper_type_num=typenum_in_lmpfile["improper types"],
        )
    # Output nbZIFFF_vis.pdb file
    if outputted_pdb_filename is not None:
        u_new = convert_atom_types(u_new, "str", atom_type_map)
        write_universe_to_normalized_pdb_format(
            u_new, os.path.join(output_folder, outputted_pdb_filename)
        )

