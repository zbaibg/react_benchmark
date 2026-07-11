"""
Utility functions for xZn-yMIm-zMeOH complex calculation and analysis
Files structures:
    - 5coord(any name you like)/
        - xZn_yMIm_zMeOH/
            - cal.in
            - init_geo.xyz/pdb/gjf
            - cal_trj.xyz
            ...
"""

import os
import re
from typing import List
import MDAnalysis as mda
from ..utils import view_atomgroups_with_correct_bonds
def gen_structurename_list(coord_num:int):
    '''
    generate the structure name list
    coord_num: the coordination number
    '''
    if coord_num>0:
        return [f'1Zn_{i}MIm_{coord_num-i}MeOH' for i in range(0,coord_num+1)]
    else:
        return ['1Zn_0MIm_0MeOH','0Zn_1MIm_0MeOH','0Zn_0MIm_1MeOH']



def get_structure_folder_or_file(coord_path:str=None,file_name:str=None,print_not_exist:bool=True):
    '''if file_name is assigned, return the path list of the file in each structure folder
    
    '''
    reslist=[]
    subfolders=os.listdir(coord_path)
    for subfolder in subfolders:
        if 'Zn' in subfolder and 'MeOH' in subfolder and 'MIm' in subfolder:
            path=f'{coord_path}/{subfolder}/{file_name}' if file_name is not None else f'{coord_path}/{subfolder}'
            if os.path.exists(path):
                reslist.append(path)
            else:
                if print_not_exist:
                    print(f'{path} does not exist')
    return reslist

def get_structure_str(f):
    '''
    get the structure name from the path
    f: the path of files inside the structure folder, ot the folder itself
    return: the structure name
    '''
    if os.path.isfile(f):
        return os.path.basename(os.path.dirname(f))
    else:
        return os.path.basename(f)

def get_node_num(name) -> dict:
    '''
    get the number of each node in the structure name
    name: the name of the structure
    return: a dictionary with the number of each node
    '''
    assert name is not None, "name is None"
    res={}
    for node in ['MIm','MeOH','Zn']:
        if node in name:
            num = int(re.search(r'(\d+)'+node, name).group(1))
            res[node]=num
        else:
            res[node]=0
    return res

def get_geo_path_list(coord_path:str,files_to_find:list):
    '''
    get the geometry path list from the coordination path
    coord_path: the path of the coordination
    return: the geometry path list
    '''
    assert isinstance(files_to_find,list),'files_to_find must be a list'
    struc_folder_list=get_structure_folder_or_file(coord_path,print_not_exist=False)
    struc_list=[get_structure_str(path) for path in struc_folder_list]
    reslist=[]
    for file_to_find in files_to_find:
        geo_list= get_structure_folder_or_file(coord_path,file_name=file_to_find,print_not_exist=False)
        reslist.extend(geo_list)
    struc_list_with_geos=[get_structure_str(path) for path in reslist]
    for struc,folder in zip(struc_list,struc_folder_list):
        if struc not in struc_list_with_geos:
            print(f'does not find any supported geometry files for {folder}. Try to find {files_to_find}')
    reslist.sort()
    return reslist

def turn_gjf_to_xyz(gjf_path:str,xyz_path:str):
    '''
    turn the gjf file to xyz file
    gjf_path: the path of the gjf file
    xyz_path: the path of the xyz file
    '''
    copy_content=[]
    with open(gjf_path,'r') as f:
        for _ in range(6):
            next(f)
        for line in f:
            if line.strip()=='':
                break
            copy_content.append(line)
    with open(xyz_path,'w') as f:
        f.write(str(len(copy_content))+'\n\n')
        for line in copy_content:
            f.write(line)
def get_Universe_from_file(file_path:str,in_memory:bool=True):
    '''
    get the universe from the file
    file_path: the path of the file
    return: the universe object
    '''
    if file_path.endswith('.gjf'):
        turn_gjf_to_xyz(file_path,file_path.replace('.','_')+'.xyz')
        file_path=file_path.replace('.','_')+'.xyz'
    return mda.Universe(file_path,in_memory=in_memory)
def get_mda_universes_from_paths(geofile_path_list:list,in_memory:bool=True):
    '''
    get the geometries from the geofile_path_list. Support xyz pdb or gjf format
    geofile_path_list: the path of the geofile
    return: the mdanalysis universe object
    '''
    return [get_Universe_from_file(f,in_memory=in_memory) for f in geofile_path_list]
def show_mda_universe(Universe_list:List[mda.Universe],title_list:List[str]=None,auto_bond:bool=False):
    if title_list is not None:
        assert len(Universe_list)==len(title_list),'the length of Universe_list and title_list must be the same'
    for uni,title in zip(Universe_list,title_list):
        print(title)
        view_atomgroups_with_correct_bonds(uni.atoms,auto_bond=auto_bond)
        
def show_init_geos_for_coord(coord_folder:str):
    geofile_path_list=get_geo_path_list(coord_folder,files_to_find=['init_geo.xyz','init_geo.pdb','init_geo.gjf'])
    uni_list=get_mda_universes_from_paths(geofile_path_list)
    show_mda_universe(uni_list,title_list=geofile_path_list)
    
def show_final_geos_for_coord(coord_folder:str):
    geofile_path_list=get_geo_path_list(coord_folder,files_to_find=['cal.xyz'])
    uni_list=get_mda_universes_from_paths(geofile_path_list)
    show_mda_universe(uni_list,title_list=geofile_path_list)

def show_traj_for_coord(coord_folder:str):
    geofile_path_list=get_geo_path_list(coord_folder,files_to_find=['cal_trj.xyz'])
    uni_list=get_mda_universes_from_paths(geofile_path_list)
    show_mda_universe(uni_list,title_list=geofile_path_list)
    
def tile_universe(universe, n_x, n_y, n_z):
    box = universe.dimensions[:3]
    copied = []
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                u_ = universe.copy()
                move_by = box*(x, y, z)
                u_.atoms.translate(move_by)
                copied.append(u_.atoms)

    new_universe = mda.Merge(*copied)
    new_box = box*(n_x, n_y, n_z)
    new_universe.dimensions = list(new_box) + [90]*3
    return new_universe

def create_box_for_OPB(Universe,padding:float=10):
    '''
    assign a box with a padding to the universe. This aims to create box mimicking the open boundary condition with padding of interaction cutoff distance.
    A box is necessary for MDAnalysis LAMMPS DATA file dumps even though you want to use open boundary condition.
    '''
    mol_dim=Universe.atoms.positions.max(axis=0)-Universe.atoms.positions.min(axis=0)
    Universe.dimensions = list(mol_dim+padding)+[90,90,90]
    return Universe
def move_atoms_to_center(Universe):
    atom_center=Universe.atoms.center_of_mass()
    cell_center=Universe.dimensions[:3]/2
    Universe.atoms.translate(cell_center-atom_center)
    return Universe