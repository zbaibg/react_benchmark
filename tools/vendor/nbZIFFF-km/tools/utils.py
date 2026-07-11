from typing import List
import MDAnalysis
import MDAnalysis.core.groups
import nglview
import numpy as np
from ase import io
import os
import re
from IPython.display import display
import pandas as pd
import itertools


def add_bonds_to_view(
    view: nglview.NGLWidget, atomgroups: MDAnalysis.AtomGroup
) -> None:
    """
    add bonds one by one to the view
    """
    for bond in atomgroups.bonds:
        index1 = bond[0].index
        index2 = bond[1].index
        indexhere1 = np.where(atomgroups.atoms.indices == index1)[0][0]
        indexhere2 = np.where(atomgroups.atoms.indices == index2)[0][0]
        view.add_representation(
            "ball+stick", selection=f"@{indexhere1} or @{indexhere2}"
        )


def view_atomgroups_with_correct_bonds(
    atomgroups: MDAnalysis.AtomGroup, auto_bond: bool = False
) -> None:
    """
    bug: if the bonds are too long, the ball+stick representation will not be able to show the bonds.
    view the atomgroups in nglview

    This will reload the trajectory for the atomgroups.
    So make sure you have an in-memory trajectory if you unwrap the atoms out of the cell.
    Or it will wrap the atoms back into the cell.

    auto_bond: bool
        If True, the bonds will be guessed by atom distances.
        If False, the bonds will be the ones in the atomgroups.bonds.
    """
    view = nglview.show_mdanalysis(atomgroups, default_representation=False)
    # view.add_representation('ball+stick', selection='@80 or @145', color='red',radius=0.2)
    if auto_bond:
        view.add_representation("ball+stick", selection="*", radius=0.3)
    else:
        view.add_representation("spacefill", selection="*", radius=0.3)
        if hasattr(atomgroups, "bonds"):
            if len(atomgroups.bonds) > 0:
                add_bonds_to_view(view, atomgroups)
    overwrite_nglview_default(view)
    view.center()
    display(view)
    # return view


def lookup_typotype_from_map(object, type_map, return_none=False):
    """Get the numeric topology type for a topology object.
    Looks up the type in type_map using the atom types of the object.
    Tries different possible order of atom types since some identical topology objects have nonunique representations.
    """
    types = object.atoms.types
    if isinstance(object, MDAnalysis.core.topologyobjects.ImproperDihedral):
        type1 = type_map.get(tuple(types))
        type2 = type_map.get(tuple(types[i] for i in [0, 2, 1, 3]))
        type3 = type_map.get(tuple(types[i] for i in [3, 1, 2, 0]))
        type4 = type_map.get(tuple(types[i] for i in [3, 2, 1, 0]))
        final = type1 or type2 or type3 or type4
    else:
        type1 = type_map.get(tuple(types))
        type2 = type_map.get(tuple(types[::-1]))
        final = type1 or type2

    if final is None:
        if return_none:
            return None
        else:
            raise ValueError(
                f"{object} with atom indices {object.indices}:{types} is not listed in relevant type_map"
            )
    else:
        return final


def convert_topology_types_to_numeric(
    universe: MDAnalysis.Universe,
    bond_type_map: dict,
    angle_type_map: dict,
    dihedral_type_map: dict,
    improper_type_map: dict,
) -> MDAnalysis.Universe:
    """Convert topology types to numeric indices for a Universe.

    Creates a new Universe with the same atoms but converts topology types (bonds,
    angles, dihedrals, impropers) from atom type strings to numeric indices based on
    predefined type maps. For each topology element, looks up the numeric type using
    the atom types of the atoms involved.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing atoms and topology to convert

    Returns
    -------
    MDAnalysis.Universe
        New Universe with topology types converted to numeric indices
    """
    u_new = universe.copy()
    for topology in ["bonds", "angles", "dihedrals", "impropers"]:
        if hasattr(u_new, topology):
            getattr(u_new, f"delete_{topology}")(getattr(u_new, topology))
    if hasattr(universe, "bonds"):
        for bond in universe.bonds:
            u_new.add_bonds(
                [bond.indices], types=[lookup_typotype_from_map(bond, bond_type_map)]
            )

    if hasattr(universe, "angles"):
        for angle in universe.angles:
            u_new.add_angles(
                [angle.indices], types=[lookup_typotype_from_map(angle, angle_type_map)]
            )

    if hasattr(universe, "dihedrals"):
        for dihedral in universe.dihedrals:
            u_new.add_dihedrals(
                [dihedral.indices],
                types=[lookup_typotype_from_map(dihedral, dihedral_type_map)],
            )

    if hasattr(universe, "impropers"):
        for improper in universe.impropers:
            u_new.add_impropers(
                [improper.indices],
                types=[lookup_typotype_from_map(improper, improper_type_map)],
            )

    return u_new

def add_empty_topology_if_none(universe: MDAnalysis.Universe) -> None:
    """Add an empty topology attribute to a universe if it doesn't exist. 
    Some functions of MDAnalysis require the topology attributes to exist. Eg: write to LAMMPS data file.
    """
    for topology in ["bonds", "angles", "dihedrals", "impropers"]:
        if not hasattr(universe, topology):
            universe.add_TopologyAttr(topology, [])

def remove_extra_topology_objects_from_universe(
    universe: MDAnalysis.Universe,
    bond_type_map: dict = None,
    angle_type_map: dict = None,
    dihedral_type_map: dict = None,
    improper_type_map: dict = None,
) -> MDAnalysis.Universe:
    """Remove extra topology objects from a universe.

    This function removes extra topology objects from a universe based on the type maps provided.
    It removes bonds, angles, dihedrals, and impropers that are not listed in the type maps.

    If any map is not provided, it will not remove any topology objects for that type.
    """
    u_new = universe.copy()
    if bond_type_map is not None:
        if hasattr(u_new, "bonds"):
            for bond in u_new.bonds:
                if lookup_typotype_from_map(bond, bond_type_map, return_none=True) is None:
                    u_new.delete_bonds([bond])
    if angle_type_map is not None:
        if hasattr(u_new, "angles"):
            for angle in u_new.angles:
                if (
                    lookup_typotype_from_map(angle, angle_type_map, return_none=True)
                    is None
                ):
                    u_new.delete_angles([angle])
    if dihedral_type_map is not None:
        if hasattr(u_new, "dihedrals"):
            for dihedral in u_new.dihedrals:
                if (
                    lookup_typotype_from_map(dihedral, dihedral_type_map, return_none=True)
                    is None
                ):
                    u_new.delete_dihedrals([dihedral])
    if improper_type_map is not None:
        if hasattr(u_new, "impropers"):
            for improper in u_new.impropers:
                if (
                    lookup_typotype_from_map(improper, improper_type_map, return_none=True)
                    is None
                ):
                    u_new.delete_impropers([improper])
    return u_new


def filter_impropers(universe: MDAnalysis.Universe, improper_type_map: dict) -> None:
    """
    Deprecated. Use remove_extra_topology_objects_from_universe instead.

    Choose the wanted improper dihedral from guessed impropers and delete the rest in the inputed universe.

    The improper diherals guessed by MDAnalysis are not duplicated but we only want a part of them.
    This function takes a universe containing guessed impropers and choose the wanted ones.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing guessed impropers

    Returns
    -------
    None
    """
    original_impropers = list(universe.impropers)

    for improper in original_impropers:
        sequences = [[0, 1, 2, 3], [0, 2, 1, 3], [3, 1, 2, 0], [3, 2, 1, 0]]
        wanted = False
        for seq in sequences:
            improper_atomtypes = tuple(improper.atoms.types[i] for i in seq)
            if improper_atomtypes in improper_type_map.keys():
                wanted = True
                break
        if not wanted:
            universe.delete_impropers([improper])
    return


def get_universe_from_vasp_file(
    vasp_filename: str, in_memory: bool = True
) -> MDAnalysis.Universe:
    """
    Convert a VASP format structure file to LAMMPS data format without topology information.

    Args:
        vasp_filename (str): Path to input VASP structure file (POSCAR or CONTCAR)
    Returns:
        None.
    """
    # Read VASP structure using ASE
    structure = io.read(vasp_filename, format="vasp")

    tmp_xyz_file = os.path.join(
        os.path.dirname(vasp_filename),
        os.path.basename(vasp_filename).replace(".", "_") + "_tmp.pdb",
    )
    io.write(
        tmp_xyz_file,
        structure,
        format="proteindatabank",
    )
    u = MDAnalysis.Universe(tmp_xyz_file, format="pdb", in_memory=in_memory)
    normalize_2char_atomtype(u, inplace=True)
    return u


def write_universe_to_normalized_pdb_format(
    universe: MDAnalysis.Universe, pdb_filename: str
) -> None:
    """
    Write a universe to a PDB format structure file.
    assume the atom types are Element Name of 1 or 2 characters (optionallywith a Number).
    Then set the "name" to be the same as the atom types, and the "element" to be the letter part of the atom types.
    """
    normalize_2char_atomtype(universe, inplace=True)
    namelist = universe.atoms.types
    elementlist = [
        re.match(r"^([A-Za-z]+)(\d*)$", type_str).groups()[0] for type_str in namelist
    ]
    if hasattr(universe.atoms, "name"):
        universe.delete_TopologyAttr("name")
    if hasattr(universe.atoms, "element"):
        universe.delete_TopologyAttr("element")
    universe.add_TopologyAttr("name", namelist)
    universe.add_TopologyAttr("element", elementlist)
    universe.atoms.write(pdb_filename, file_format="PDB")


def read_universe_from_pdb_and_normalize(pdb_filename: str) -> MDAnalysis.Universe:
    """
    Read a PDB file and normalize the atom types.
    Assume the atom types are Element Name of 1 or 2 characters (optionallywith a Number).
    Then set the "name" to be the same as the atom types, and the "element" to be the letter part of the atom types.
    """
    u = MDAnalysis.Universe(pdb_filename, format="PDB")
    normalize_2char_atomtype(u, inplace=True)
    namelist = u.atoms.types
    elementlist = [
        re.match(r"^([A-Za-z]+)(\d*)$", type_str).groups()[0] for type_str in namelist
    ]
    if hasattr(u.atoms, "name"):
        u.delete_TopologyAttr("name")
    if hasattr(u.atoms, "element"):
        u.delete_TopologyAttr("element")
    u.add_TopologyAttr("name", namelist)
    u.add_TopologyAttr("element", elementlist)
    return u


def normalize_2char_atomtype(
    universe: MDAnalysis.Universe, inplace: bool = False, remove_number: bool = False
) -> MDAnalysis.Universe:
    """
    Normalize the "types" property of the universe.atoms with 2 letters.
    Make the first character uppercase and the second character lowercase.
    Assume the atom type format is Element Name of 1 or 2 characters (optionallywith a Number).
    Examples:
    ZN -> Zn
    ZN1 -> Zn1
    ZN12 -> Zn12
    """
    if inplace:
        u_new = universe
    else:
        u_new = universe.copy()
    for type_str in set(u_new.atoms.types):
        # Split into letters and numbers
        match = re.match(r"^([A-Za-z]+)(\d*)$", type_str)
        if not match:
            raise ValueError(
                f"Atom type {type_str} is not in the format of Element Name of 1 or 2 characters (optionally with a Number)"
            )
        letters, numbers = match.groups()
        assert len(letters) in [
            1,
            2,
        ], f"Element Name {type_str} is not of 1 or 2 characters"
        # Keep first letter uppercase, rest lowercase
        if remove_number:
            type_str_new = letters[0].upper() + letters[1:].lower()
        else:
            type_str_new = letters[0].upper() + letters[1:].lower() + numbers
        u_new.atoms.select_atoms(f"type {type_str}").types = type_str_new
    if inplace:
        return None
    else:
        return u_new


def convert_atom_types(
    universe: MDAnalysis.Universe,
    target_format: str,
    atom_type_map: dict,
    str_without_number: bool = False,
) -> MDAnalysis.Universe:
    """Convert atom types between string and integer representations.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing atoms whose types should be converted
    target_format : str
        Format to convert to - either 'str' or 'int'
    str_without_number: bool
        If True, when the target_format is 'str', the string atom types will be converted to the string atom types without numbers.
        This is useful when the wanted atom types are string without numbers, but the atom types in the atom_type_map are string with numbers.
    atom_type_map: dict
        Dictionary mapping the string atom types to their corresponding integer values
    Returns
    -------
    MDAnalysis.Universe
        Universe with converted atom types
    """
    u_new = universe.copy()
    inv_atom_type_dict = {str(v): k for k, v in atom_type_map.items()}
    if target_format == "int":
        assert set(u_new.atoms.types).issubset(
            set(atom_type_map.keys())
        ), f"Atom types {set(u_new.atoms.types)} are not a subset of {set(atom_type_map.keys())}"
        u_new.atoms.types = [str(atom_type_map[t]) for t in u_new.atoms.types]
    elif target_format == "str":
        assert set(u_new.atoms.types).issubset(
            set(inv_atom_type_dict.keys())
        ), f"Atom types {set(u_new.atoms.types)} are not a subset of {set(inv_atom_type_dict.keys())}"
        if str_without_number:
            u_new.atoms.types = [
                re.sub(r"\d+", "", inv_atom_type_dict[str(t)])
                for t in u_new.atoms.types
            ]
        else:
            u_new.atoms.types = [inv_atom_type_dict[str(t)] for t in u_new.atoms.types]
    else:
        raise ValueError(f"Unknown target format: {target_format}")

    return u_new


def init_topology(
    universe: MDAnalysis.Universe,
    vdwradii_for_bondguess: dict,
    fudge_factor_for_bondguess: float,
    filter: bool = True,
    bond_type_map: dict = None,
    angle_type_map: dict = None,
    dihedral_type_map: dict = None,
    improper_type_map: dict = None,
) -> MDAnalysis.Universe:
    """
    Initialize the topology attributes of a universe.
    Parameters:
    -----------
    universe: MDAnalysis.Universe
        The universe to initialize the topology attributes.
    vdwradii_for_bondguess: dict
        The vdwradii for bond guess.
    fudge_factor_for_bondguess: float
        The fudge factor for bond guess.
    filter: bool
        If True, the topology objects will be filtered to keep the wanted types within the bond_type_map, angle_type_map, dihedral_type_map, improper_type_map.
    bond_type_map: dict
        The map for bond types.
    angle_type_map: dict
        The map for angle types.
    dihedral_type_map: dict
        The map for dihedral types.
    improper_type_map: dict
        The map for improper types.
    """
    for topology in ["bonds", "angles", "dihedrals", "impropers"]:
        if hasattr(universe, topology):
            getattr(universe, f"delete_{topology}")(getattr(universe, topology))
    universe.atoms.guess_bonds(
        vdwradii=vdwradii_for_bondguess, fudge_factor=fudge_factor_for_bondguess
    )
    universe = guess_all_possible_impropers(
        universe, filter=True, improper_type_map=improper_type_map
    )
    if filter:
        # Improper is already filtered by guess_all_possible_impropers, so we only need to filter the bonds, angles, dihedrals here
        universe = remove_extra_topology_objects_from_universe(
            universe,
            bond_type_map=bond_type_map,
            angle_type_map=angle_type_map,
            dihedral_type_map=dihedral_type_map,
        )
    return universe


def add_block_to_lmpdata(
    lmp_data: str, block_to_add: str, before_section_name: str = "Atoms"
) -> None:
    """
    Assert the block before one specific section in the LAMMPS data file.
    This could be used to add the topology coefficients before the Atoms block.
    Parameters:
    -----------
    lmp_data: str
        The path to the lmpdata file.
    block_to_add: str
        The block to add.
    before_section_name: str
        The name of the section before which to add the block.
        eg. before_section_name='Atoms'
    """
    block_lines = [line + "\n" for line in block_to_add.strip().split("\n")]
    block_lines.append("\n")
    with open(lmp_data, "r") as f:
        lines_read = f.readlines()
        line_mark = None
        for i in range(len(lines_read)):
            if lines_read[i].startswith(before_section_name):
                line_mark = i
                break
        assert line_mark is not None
    with open(lmp_data, "w") as f:
        f.writelines(lines_read[0:line_mark] + block_lines + lines_read[line_mark:])


def modify_block_to_lmpdata(
    lmp_data: str, block_to_modify: str, section_name: str = "Masses"
) -> None:
    """
    Modify the block of one specific section in the LAMMPS data file.
    This could be used to modify the Masses block in the LAMMPS data file.
    Parameters:
    -----------
    lmp_data: str
        The path to the lmpdata file.
    block_to_modify: str
        The block to modify.
    section_name: str
        The name of the section to modify.
    """
    block_lines = [line + "\n" for line in block_to_modify.strip().split("\n")]
    block_lines.append("\n")
    with open(lmp_data, "r") as f:
        lines_read = f.readlines()
        line_mark = None
        line_end = None
        for i in range(len(lines_read)):
            if lines_read[i].startswith(section_name):
                line_mark = i
                break
        for i in range(line_mark + 2, len(lines_read)):
            if lines_read[i].strip() == "":
                line_end = i
                break
        assert line_mark is not None and line_end is not None
    with open(lmp_data, "w") as f:
        f.writelines(lines_read[0:line_mark] + block_lines + lines_read[line_end + 1 :])


def modify_typenums_in_lmpdata(
    lmp_data: str,
    atom_type_num: int,
    bond_type_num: int,
    angle_type_num: int,
    dihedral_type_num: int,
    improper_type_num: int,
) -> str:
    """
    Modify the type numbers in the head of the LAMMPS data file.
    """
    assert isinstance(atom_type_num, int)
    assert isinstance(bond_type_num, int)
    assert isinstance(angle_type_num, int)
    assert isinstance(dihedral_type_num, int)
    assert isinstance(improper_type_num, int)
    with open(lmp_data, "r") as f:
        lines_read = f.readlines()
        content = "".join(lines_read)
        # Use regex to find and replace the type numbers
        content = re.sub(
            r"(\s*)(\d+)(\s+atom types\s*\n)",
            r"\g<1>" + str(atom_type_num) + r"\g<3>",
            content,
            count=1,
        )
        content = re.sub(
            r"(\s*)(\d+)(\s+bond types\s*\n)",
            r"\g<1>" + str(bond_type_num) + r"\g<3>",
            content,
            count=1,
        )
        content = re.sub(
            r"(\s*)(\d+)(\s+angle types\s*\n)",
            r"\g<1>" + str(angle_type_num) + r"\g<3>",
            content,
            count=1,
        )
        content = re.sub(
            r"(\s*)(\d+)(\s+dihedral types\s*\n)",
            r"\g<1>" + str(dihedral_type_num) + r"\g<3>",
            content,
            count=1,
        )
        content = re.sub(
            r"(\s*)(\d+)(\s+improper types\s*\n)",
            r"\g<1>" + str(improper_type_num) + r"\g<3>",
            content,
            count=1,
        )

    with open(lmp_data, "w") as f:
        f.write(content)


def overwrite_nglview_default(widget: nglview.NGLWidget):
    """
    Copied from mbuild.utils.jsutils import overwrite_nglview_default

    Change the default visualization in nglview.

    This method takes in a nglview.NGLWidget and changes the default hover
    behaviour of the widget to add the atom index when it is hovered over
    the atom. It also overwrites the click signal from the stage to include
    extra information(atom index) in the text display, whenever an atom or
    bond is clicked.

    Parameters:
    ----------
    widget: nglview.NGLWidget, the ipython widget view.
    Returns:
    --------
    None
    Raises:
    ------
    TypeError: If widget is not of type nglview.NGLWidget
    """
    if not isinstance(widget, nglview.NGLWidget):
        raise TypeError(
            "The argument widget can only be of type nglview.NGLWidget not {}".format(
                type(widget)
            )
        )
    tooltip_js = """
                    this.stage.mouseControls.add('hoverPick', (stage, pickingProxy) => {
                        let tooltip = this.stage.tooltip;
                        if(pickingProxy && pickingProxy.atom && !pickingProxy.bond){
                            let atom = pickingProxy.atom;
                            tooltip.innerText = "ATOM: " + atom.qualifiedName() + ", Index: " + atom.index;
                        }
                    });
                 """

    infotext_js = """
                    this.stage.signals.clicked.removeAll();
                    this.stage.signals.clicked.add((pickingProxy) => {
                            if(pickingProxy){
                               let pickingText = null;
                               this.model.set('picked', {});
                               this.touch();
                               let currentPick = {};
                               if(pickingProxy.atom){
                                    currentPick.atom1 = pickingProxy.atom.toObject();
                                    currentPick.atom1.name = pickingProxy.atom.qualifiedName();
                                    pickingText = "Atom: " + currentPick.atom1.name + ", Index: " 
                                                  + pickingProxy.atom.index;
                               }
                               else if(pickingProxy.bond){
                                    currentPick.bond = pickingProxy.bond.toObject();
                                    currentPick.atom1 = pickingProxy.bond.atom1.toObject();
                                    currentPick.atom1.name = pickingProxy.bond.atom1.qualifiedName();
                                    currentPick.atom2 = pickingProxy.bond.atom2.toObject();
                                    currentPick.atom2.name = pickingProxy.bond.atom2.qualifiedName();
                                    pickingText = "Bond: " + currentPick.atom1.name + 
                                                    `(${pickingProxy.bond.atom1.index})` +
                                                    " - " + currentPick.atom2.name    +
                                                    `(${pickingProxy.bond.atom2.index})`;
                               }
                               
                               if(pickingProxy.instance){
                                    currentPick.instance = pickingProxy.instance;
                               }
                               var nComponents = this.stage.compList.length;
                               for(let i = 0; i < nComponents; i++){
                                    let comp = this.stage.compList[i];
                                    if(comp.uuid == pickingProxy.component.uuid){
                                        currentPick.component = i;
                                    }
                               }
                               this.model.set('picked', currentPick);
                               this.touch();
                               this.$pickingInfo.text(pickingText);
                            }
                    });
                """
    widget._js(tooltip_js)
    widget._js(infotext_js)


def guess_all_possible_impropers(
    universe: MDAnalysis.Universe, filter: bool = True, improper_type_map: dict = None
) -> MDAnalysis.Universe:
    """
    MDanalaysis only consider 1-3, 3-4, 1-4 as the common edge between the two half-planes when guessing
    impropers (See figure below). This function will guess all 6 possible impropers including 1-2, 2-3, 2-4 as the common edge,
    in addition to the 3 impropers mentioned above.
    
    It will first clean the existing impropers. Then guess the impropers based on the bonds of the universe.
    
       1
        \
        2 -- 3
        /
       4

    Parameters:
    -----------
    universe: MDAnalysis.Universe
        The universe to guess the impropers.
    filter: bool
        If True, the impropers will be filtered to keep the wanted types within the improper_type_map.
    improper_type_map: dict
        The map for improper types.
    Returns:
    --------
    MDAnalysis.Universe
        The universe with the guessed impropers.
    """
    if hasattr(universe, "impropers"):
        universe.delete_impropers(universe.impropers)
    #Make sure at least a empty impropers attribute is added to the universe
    universe.add_TopologyAttr("impropers", [])
    def add_impropers_for_4atoms(
        universe: MDAnalysis.Universe, four_atoms_ag: MDAnalysis.AtomGroup
    ) -> None:
        assert (
            len(set(four_atoms_ag)) == 4
        ), f"The four atoms are not unique: {four_atoms_ag}"
        for common_egde in itertools.combinations([0, 1, 2, 3], 2):
            other_indices = list(set([0, 1, 2, 3]) - set(common_egde))
            other_indices.sort()
            ag = four_atoms_ag[
                [other_indices[0], common_egde[0], common_egde[1], other_indices[1]]
            ]
            universe.add_impropers([ag])

    atomtype_not_to_search = set()
    if filter:
        possible_improper_atomtypes = set(
            atom_type
            for improper_type in improper_type_map.keys()
            for atom_type in improper_type
        )
        atomtype_not_to_search = set(universe.atoms.types) - possible_improper_atomtypes
    for atom in universe.atoms:
        if (
            atom.type not in atomtype_not_to_search
        ):  # Only look around those possible atoms to speed up
            for four_atoms in itertools.combinations(atom.bonded_atoms + atom, 4):
                four_atoms_ag = MDAnalysis.AtomGroup(four_atoms)
                if filter and sorted(list(four_atoms_ag.types)) not in [
                    sorted(atomtypes) for atomtypes in improper_type_map.keys()
                ]:
                    # Only add those possible impropers to speed up
                    continue
                add_impropers_for_4atoms(universe, four_atoms_ag)
    # Strictly remove those unwanted impropers
    universe = remove_extra_topology_objects_from_universe(
        universe, improper_type_map=improper_type_map
    )
    return universe


def check_topologies_and_positions_are_consistent(
    universe1: MDAnalysis.Universe,
    universe2: MDAnalysis.Universe,
    raise_error: bool = True,
    tolerance: float = 1e-4,
) -> tuple[bool, list[str]]:
    """
    Check if the topologies and atom positions from two different universes are consistent.
    This can be used to check if two lammps data files are consistent.

    Returns:
    --------
    bool: True if the topologies and atom positions are consistent, False otherwise.
    list[str]: A list of error messages if the topologies and atom positions are inconsistent.
    """
    # Create DataFrames for atoms from both universes
    df_sorted_atom_u1 = pd.DataFrame(
        {
            "x": universe1.atoms.positions[:, 0],
            "y": universe1.atoms.positions[:, 1],
            "z": universe1.atoms.positions[:, 2],
            "type": universe1.atoms.types,
            "mass": universe1.atoms.masses,
            "charge": universe1.atoms.charges,
        },
        index=universe1.atoms.indices,
    )

    df_sorted_atom_u2 = pd.DataFrame(
        {
            "x": universe2.atoms.positions[:, 0],
            "y": universe2.atoms.positions[:, 1],
            "z": universe2.atoms.positions[:, 2],
            "type": universe2.atoms.types,
            "mass": universe2.atoms.masses,
            "charge": universe2.atoms.charges,
        },
        index=universe2.atoms.indices,
    )
    df_sorted_atom_u1 = df_sorted_atom_u1.sort_values(by=["x", "y", "z"])
    df_sorted_atom_u2 = df_sorted_atom_u2.sort_values(by=["x", "y", "z"])
    U1_atom_sortedID_to_oldID = np.array(df_sorted_atom_u1.index)
    U2_atom_sortedID_to_oldID = np.array(df_sorted_atom_u2.index)
    U1_atom_oldID_to_sortedID = np.argsort(U1_atom_sortedID_to_oldID)
    U2_atom_oldID_to_sortedID = np.argsort(U2_atom_sortedID_to_oldID)
    df_sorted_atom_u1.reset_index(drop=True, inplace=True)
    df_sorted_atom_u2.reset_index(drop=True, inplace=True)
    #'df_sorted_atom_u1' and df_sorted_atom_u2's index now is the sorted atom indices
    error_message_list = []
    if df_sorted_atom_u1.shape != df_sorted_atom_u2.shape:
        error_message_list.append(
            f"#ATOM NUM INCONSISTENT#\n"+ \
            f"Atom number in Universe1: {df_sorted_atom_u1.shape} != Atom number in Universe2: {df_sorted_atom_u2.shape}"
        )
    pos_consistent = np.isclose(
        df_sorted_atom_u1[["x", "y", "z"]],
        df_sorted_atom_u2[["x", "y", "z"]],
        rtol=0,
        atol=1e-4,
    ).all(axis=1)
    for atom_sorted_ID in np.where(pos_consistent == False)[0]:
        error_message = f"#ATOM POSITION INCONSISTENT#\n"+ \
        f"ATOM Sorted_ID {atom_sorted_ID} Universe1_ID: Position {df_sorted_atom_u1.iloc[atom_sorted_ID][['x','y','z']]}\n" + \
        f"ATOM Sorted_ID {atom_sorted_ID} Universe2_ID: Position {df_sorted_atom_u2.iloc[atom_sorted_ID][['x','y','z']]}\n" + \
        error_message_list.append(error_message)
    mass_charge_consistent = np.isclose(
        df_sorted_atom_u1[["mass", "charge"]],
        df_sorted_atom_u2[["mass", "charge"]],
        rtol=0,
        atol=1e-4,
    ).all(axis=1)
    for atom_sorted_ID in np.where(mass_charge_consistent == False)[0]:
        error_message = f"#ATOM MASS AND CHARGE INCONSISTENT#\n"+ \
        f"ATOM Sorted_ID {atom_sorted_ID} Universe1_ID: Mass {df_sorted_atom_u1.iloc[atom_sorted_ID]['mass']}, Charge {df_sorted_atom_u1.iloc[atom_sorted_ID]['charge']}\n" + \
        f"ATOM Sorted_ID {atom_sorted_ID} Universe2_ID: Mass {df_sorted_atom_u2.iloc[atom_sorted_ID]['mass']}, Charge {df_sorted_atom_u2.iloc[atom_sorted_ID]['charge']}\n" + \
        error_message_list.append(error_message)
    type_consistent = df_sorted_atom_u1["type"] == df_sorted_atom_u2["type"]
    for atom_sorted_ID in np.where(type_consistent == False)[0]:
        error_message = f"#ATOM TYPE INCONSISTENT#\n"+ \
        f"ATOM Sorted_ID {atom_sorted_ID} Universe1_ID: Type {df_sorted_atom_u1.iloc[atom_sorted_ID]['type']}\n" + \
        f"ATOM Sorted_ID {atom_sorted_ID} Universe2_ID: Type {df_sorted_atom_u2.iloc[atom_sorted_ID]['type']}\n" + \
        error_message_list.append(error_message)

    # Switch atom1 and atom2 if atom1 is larger than atom2
    def check_topology_consistency(name: str, u1topo, u2topo) -> list[str]:
        """
        Check if the topologies of two universes are consistent.
        Returns:
        --------
        list[str]: A list of error messages if the topologies are inconsistent.
        """
        error_message_list = []
        assert name in ["bonds", "angles", "dihedrals", "impropers"]
        # if name=='bonds':
        # equivalence_sequences=[[0,1],[1,0]]
        # elif name=='angles':
        # equivalence_sequences=[[0,1,2],[2,1,0]]
        # elif name=='dihedrals':
        # equivalence_sequences=[[0,1,2,3],[3,2,1,0]]
        # elif name=='impropers':
        # equivalence_sequences=[[0,1,2,3],[0,2,1,3],[3,1,2,0],[3,2,1,0]]
        colname_atoms = [
            "atom" + str(i)
            for i in range(
                {"bonds": 2, "angles": 3, "dihedrals": 4, "impropers": 4}[name]
            )
        ]
        colname_all = colname_atoms + ["type"]

        def sort_func(atom_indices_to_rank: pd.Series):
            # to_sort=pd.DataFrame(np.array([atom_indices_to_rank.iloc[equivalence_sequences[i]] for i in range(len(equivalence_sequences))]),columns=colname_atoms)
            # to_sort=to_sort.sort_values(by=colname_atoms)
            # chosen_seq=to_sort.iloc[0]
            if name in ["bonds", "angles", "dihedrals"]:
                if atom_indices_to_rank.iloc[0] > atom_indices_to_rank.iloc[-1]:
                    atom_indices_to_rank = atom_indices_to_rank.iloc[::-1]
            elif name in ["impropers"]:
                if atom_indices_to_rank.iloc[0] > atom_indices_to_rank.iloc[3]:
                    atom_indices_to_rank.iloc[[0, 3]] = atom_indices_to_rank.iloc[
                        [3, 0]
                    ]
                if atom_indices_to_rank.iloc[1] > atom_indices_to_rank.iloc[2]:
                    atom_indices_to_rank.iloc[[1, 2]] = atom_indices_to_rank.iloc[
                        [2, 1]
                    ]
            return tuple(atom_indices_to_rank)

        def get_df_from_topo(topo, U_atom_oldID_to_sortedID):
            """
            Get the dataframe of the topology with sorted topology indices and atom indices.

            Returns:
            --------
            df: pd.DataFrame
                The dataframe of the topology
                index: sorted topology indices
                atom1, atom2, atom3, atom4....: sorted atom indices
                type: the topology's type
            U_topo_sortedID_to_oldID: np.ndarray
                The mapping from sorted topology indices to old topology indices
            U_topo_oldID_to_sortedID: np.ndarray
                The mapping from old topology indices to sorted topology indices
            """
            atom_num_in_topo = topo.indices.shape[1]
            df = pd.DataFrame(
                topo.indices, columns=["atom" + str(i) for i in range(atom_num_in_topo)]
            )
            for i in range(atom_num_in_topo):
                df["atom" + str(i)] = U_atom_oldID_to_sortedID[df["atom" + str(i)]]
            df["type"] = [t.type for t in topo]
            df[colname_atoms] = df[colname_atoms].apply(
                sort_func, axis=1, result_type="expand"
            )
            df = df.sort_values(by=["type"] + colname_atoms)
            U_topo_sortedID_to_oldID = np.array(df.index)
            U_topo_oldID_to_sortedID = np.argsort(U_topo_sortedID_to_oldID)
            df.reset_index(drop=True, inplace=True)
            return df, U_topo_sortedID_to_oldID, U_topo_oldID_to_sortedID

        def get_old_atomindices_from_sorted_indices(
            sorted_atom_indices, U_atom_sortedID_to_oldID
        ):
            return U_atom_sortedID_to_oldID[np.array(sorted_atom_indices, dtype=int)]

        df_sorted_topo_u1, U1_topo_sortedID_to_oldID, U1_topo_oldID_to_sortedID = (
            get_df_from_topo(u1topo, U1_atom_oldID_to_sortedID)
        )
        df_sorted_topo_u2, U2_topo_sortedID_to_oldID, U2_topo_oldID_to_sortedID = (
            get_df_from_topo(u2topo, U2_atom_oldID_to_sortedID)
        )
        if df_sorted_topo_u1.shape != df_sorted_topo_u2.shape:
            error_message_list.append(
                f"#{name.upper()} NUM INCONSISTENT#\n"+ \
                f"{name.upper()} number in Universe1: {df_sorted_topo_u1.shape} != {name.upper()} number in Universe2: {df_sorted_topo_u2.shape}"
            )
        topo_type_consistent = df_sorted_topo_u1["type"] == df_sorted_topo_u2["type"]
        for topo_sorted_ID in np.where(topo_type_consistent == False)[0]:
            error_message = f"#{name.upper()} TYPE INCONSISTENT#\n"+ \
            f"{name.upper()} Sorted_ID {topo_sorted_ID} Universe1_ID {U1_topo_sortedID_to_oldID[topo_sorted_ID]}: Topo type {df_sorted_topo_u1.iloc[topo_sorted_ID]['type']}\n" +\
            f"{name.upper()} Sorted_ID {topo_sorted_ID} Universe2_ID {U2_topo_sortedID_to_oldID[topo_sorted_ID]}: Topo type {df_sorted_topo_u2.iloc[topo_sorted_ID]['type']}\n"
            error_message_list.append(error_message)
            
        topo_atomindices_consistent = np.isclose(
            df_sorted_topo_u1[colname_atoms], df_sorted_topo_u2[colname_atoms]
        ).all(axis=1)
        for topo_sorted_ID in np.where(topo_atomindices_consistent == False)[0]:
            sorted_atom_indices_u1 = df_sorted_topo_u1.iloc[topo_sorted_ID][
                colname_atoms
            ].astype(int).tolist()
            sorted_atom_indices_u2 = df_sorted_topo_u2.iloc[topo_sorted_ID][
                colname_atoms
            ].astype(int).tolist()
            old_atomindices_u1 = get_old_atomindices_from_sorted_indices(
                sorted_atom_indices_u1, U1_atom_oldID_to_sortedID
            )
            old_atomindices_u2 = get_old_atomindices_from_sorted_indices(
                sorted_atom_indices_u2, U2_atom_oldID_to_sortedID
            )
            error_message = f"#{name.upper()} ATOM INDICES INCONSISTENT#\n"+ \
            f"{name.upper()} Sorted_ID {topo_sorted_ID} Universe1_ID {U1_topo_sortedID_to_oldID[topo_sorted_ID]}: AtomTypes {list(df_sorted_atom_u1['type'].iloc[sorted_atom_indices_u1])}, Universe1_Indices {old_atomindices_u1}, Sorted_Indices {sorted_atom_indices_u1}\n" +\
            f"{name.upper()} Sorted_ID {topo_sorted_ID} Universe2_ID {U2_topo_sortedID_to_oldID[topo_sorted_ID]}: AtomTypes {list(df_sorted_atom_u2['type'].iloc[sorted_atom_indices_u2])}, Universe2_Indices {old_atomindices_u2}, Sorted_Indices {sorted_atom_indices_u2}\n"
            error_message_list.append(error_message)

        return error_message_list

    # error_message_list.extend(check_topology_consistency('bonds',universe1.bonds,universe2.bonds))
    # error_message_list.extend(check_topology_consistency('angles',universe1.angles,universe2.angles))
    # error_message_list.extend(check_topology_consistency('dihedrals',universe1.dihedrals,universe2.dihedrals))
    error_message_list.extend(
        check_topology_consistency(
            "impropers", universe1.impropers, universe2.impropers
        )
    )
    if raise_error:
        assert len(error_message_list) == 0, "\n".join(error_message_list)
    return len(error_message_list) == 0, error_message_list

def assign_charges(universe: MDAnalysis.Universe,charges_map: dict,by_what:str='type') -> None:
    """Assign charges to the atoms in the universe.
    charges_map is a dictionary of atom string types and their charges.
    by_what is the attribute to assign the charges by. Either 'type' or 'name'
    """
    # Loop through all atoms and assign charges
    if not hasattr(universe.atoms,"charge"):
        universe.add_TopologyAttr("charge")
    for atom in universe.atoms:
        if by_what == 'type':
            atom.charge = charges_map[atom.type]
        elif by_what == 'name':
            atom.charge = charges_map[atom.name]
        else:
            raise ValueError(f"Invalid by_what: {by_what}")