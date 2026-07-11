'''
This module is used to generate the DFT input files for the specific coordination number xZn-yMIm-zMeOH complex
It can generate the 3D configuration using UFF or restart from the reference coordination number.

Files structures:
    - 5coord(any name you like)/
        - xZn_yMIm_zMeOH/
            - cal.in
            - init_geo.xyz/pdb/gjf
            - cal_trj.xyz
            ...
            
'''
from .utils import *
import shutil
from openbabel import pybel
def create_Zn_complex(MIm_num:int,MeOH_num:int) -> pybel.Molecule:
    Zn_smiles = "[Zn+2]"
    MIm_smiles = "([N-]1=C(N=C[CH]1)C)"
    MeOH_smiles = "([OH]C)"
    smiles = Zn_smiles + MIm_smiles*MIm_num + MeOH_smiles*MeOH_num
    mol = pybel.readstring("smi", smiles)
    mol.make3D('UFF',steps=2000)
    return mol

def write_cal_in(filename:str,head:str,cpu_core:int,geom_MaxIter:int,charge:int,multiplicity:int,xyz_file_path_in_cal_in:str) -> str:
    pal=f'%pal\n    nprocs {cpu_core}\nend\n'
    geom=f'%geom\n    MaxIter {geom_MaxIter}\nend\n'
    
    xyz=f'* xyzfile {charge} {multiplicity} {xyz_file_path_in_cal_in}\n'
    
    cal_in=head+'\n'+pal+'\n'+geom+'\n'+xyz
    with open(filename,'w') as f:
        f.write(cal_in)


def gen_coord(coord_num:int,coord_folder:str,init_geo_method:str='UFF',path_for_restart_coord:str=None,cal_in_head:str='! PBE Def2-TZVP TightSCF TightOpt D3BJ\n'):
    os.makedirs(coord_folder,exist_ok=True)
    if not os.path.exists(f'{coord_folder}/run.sh'):
        os.symlink('../run.sh',f'{coord_folder}/run.sh')
    for structure_name in gen_structurename_list(coord_num):
        MIm_num=get_node_num(structure_name)['MIm']
        MeOH_num=get_node_num(structure_name)['MeOH']
        Zn_num=get_node_num(structure_name)['Zn']
        charge=Zn_num*2+MIm_num*(-1)+MeOH_num*0
        structure_folder=f'{coord_folder}/{structure_name}'
        cal_in_path=f'{structure_folder}/cal.in'
        init_geo_xyz_path=f'{structure_folder}/init_geo.xyz'
        os.makedirs(structure_folder,exist_ok=True)
        if not os.path.exists(structure_folder+'/run.sh'):
            os.symlink('../run.sh',structure_folder+'/run.sh')
        if init_geo_method=='UFF':
            mol=create_Zn_complex(MIm_num,MeOH_num)
            mol.write('xyz',init_geo_xyz_path,overwrite=True)
        elif init_geo_method=='restart':
            cal_xyz_to_copy=f'{path_for_restart_coord}/{structure_name}/cal.xyz'
            assert os.path.exists(cal_xyz_to_copy),f'{cal_xyz_to_copy} does not exist'
            shutil.copy(cal_xyz_to_copy,init_geo_xyz_path)
        write_cal_in(cal_in_path,head=cal_in_head,
                     cpu_core=20,
                     geom_MaxIter=150,
                     charge=charge,
                     multiplicity=1,
                     xyz_file_path_in_cal_in='init_geo.xyz')

def create_coord_uff_PBE_TZVP(coord_num:int,new_coord_folder:str):
    os.makedirs(new_coord_folder,exist_ok=True)
    cal_in_head='! PBE Def2-TZVP TightSCF TightOpt D3BJ\n'
    gen_coord(coord_num,new_coord_folder,init_geo_method='UFF',cal_in_head=cal_in_head)

def create_coord_restart_aug_b3(coord_num:int,new_coord_folder:str,ref_coord_folder:str):
    '''copy the cal.xyz files from the reference coord folder to the new coord folder as init_geo.xyz'''
    assert os.path.exists(ref_coord_folder),f'{ref_coord_folder} does not exist'
    assert new_coord_folder!=ref_coord_folder,f'new_coord_folder and ref_coord_folder are the same'
    os.makedirs(new_coord_folder,exist_ok=True)
    cal_in_head='!B3LYP def2-TZVPPD VeryTightSCF VeryTightOpt D3BJ NORI\n'
    gen_coord(coord_num,new_coord_folder,init_geo_method='restart',path_for_restart_coord=ref_coord_folder,cal_in_head=cal_in_head)
#create_jobs_for_coord(5)
#create_jobs_for_coord(6)
#create_aug_b3_jobs_for_coord(4)
#create_coord_restart_aug_b3(0,new_coord_folder='SingleNode_restart_aug_b3',ref_coord_folder='SingleNode')