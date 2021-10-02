# -*- coding: utf-8 -*-
"""
ToDo
"""
import os
import aiida
import pandas as pd
from aiida.orm import StructureData
from aiida.plugins import DataFactory
aiida.load_profile()

class MagnData(StructureData):
    """MagnData class from StructureData"""

    def __init__(self,structure, **kwargs):
        """What we need for calculation is :
            1. A posfile 
            2. jfile 
            3. 
        """
        super(MagnData,self).__init__(**kwargs)

    def magn_atoms(self,posfile = None,target_atoms=None):
        if posfile == None and target_atoms != None:
            ase_struc = self.get_ase()#get ase structure from AiiDA structure datatype
            chemical_symbles = ase_struc.get_chemical_symbols()
            position_array = ase_struc.get_positions()
            #find out the position matrix for target_atom
            #like Sr2NbIrO6  ['Sr', 'Sr', 'Nb', 'Ir', 'O', 'O', 'O', 'O', 'O', 'O'] with target_atoms = ["Ir","Nb"]
            #position_indicator = [2,3]
            position_indicator = []
            atom_type_indicator = []
            for t in target_atoms:
                atom_type_indicator.append(target_atoms.index(t)+1)
                position_indicator = position_indicator + [i for i, cs in enumerate(chemical_symbles) if cs == t]
            
            posfile_array = position_array[position_indicator]
            #TODO : here I am confused with Anders' PDF and UppASD's user guide so just use user guides as standerd answer.
            #See user guide page 11 version 2017
            posfile_df = pd.DataFrame({'atom_index':list(range(1,len(posfile_array)+1)),
                                       'atom_type': atom_type_indicator,
                                       'position_x': posfile_array[:,0],
                                       'position_y': posfile_array[:,1],
                                       'position_z': posfile_array[:,2]})
            
    def map_Jij(self,Jij_file):
        
        
        
        
        



#test build supercell from ase 
from ase.io import read, write
a = read('Fe.cif')


