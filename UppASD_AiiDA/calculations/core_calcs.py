"""
Calculations provided by aiida_diff.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""
from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.orm import SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
from aiida.plugins import DataFactory, CalculationFactory
import numpy as np
from os import listdir
from os.path import isfile, join
import os
# registed in setup.json file

def find_out_files(filepath,except_files):
    if ".DS_Store" not in except_files:
        except_files.append(".DS_Store")
    filenames = [f for f in listdir(filepath) if (isfile(join(filepath, f)) and f not in except_files)]  
    return filenames


class UppASD(CalcJob):
    """
    AiiDA calculation plugin wrapping the SD executable (from UppASD packages).
    Basic 
    """
    @classmethod
    def define(cls, spec):  # cls this is the reference of the class itself and is mandatory for any class method, spec which is the ‘specification’
        """Define inputs and outputs of the calculation."""
        # yapf: disable
        # replace the class name cls with the name of UppASD calculation job
        super(UppASD, cls).define(spec)
        # input file sections :
        # Core data types: Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData.  Please classify UppASD's need(in furture :-) ) into those and update plugins
        spec.input('prepared_file_folder', valid_type=Str,
                   help='path to prepared_file_folder')
        spec.input('except_filenames', valid_type=List,
                   help='list of excepted filenames')
        spec.input('inpsd', valid_type=Dict,
                   help='the dict of inpsd.dat', required=False,default=lambda:Dict(dict={}))
        spec.input('retrieve_list_name', valid_type=List,
                   help='list of output file name')
        # output sections:
        # the instance that defined here should be used in parser
        spec.output('totenergy', valid_type=ArrayData,
                    help='all data that stored in totenergy.out')
        spec.output('coord', valid_type=ArrayData,
                    help='all data that stored in coord.out')
        spec.output('qpoints', valid_type=ArrayData,
                    help='all data that stored in qpoints.out')
        spec.output('averages', valid_type=ArrayData,
                    help='all data that stored in averages.out')
        spec.output('qm_sweep', valid_type=ArrayData,
                    help='all data that stored in qm_sweep.out')
        spec.output('qm_minima', valid_type=ArrayData,
                    help='all data that stored in qm_minima.out')
        
        spec.output('mom_states_traj', valid_type=ArrayData,
                    help='all data that stored in moment.out')
        
        # exit code section
        spec.exit_code(100, 'ERROR_MISSING_OUTPUT_FILES',
                       message='Calculation did not produce all expected output files.')

    def prepare_for_submission(self, folder):
        """
        Create input file: inpsd.dat

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files needed by
            the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        calcinfo = datastructures.CalcInfo()
        
        #input_filenames = find_out_files(self.inputs.prepared_file_folder.value,self.inputs.except_filenames.get_list())
        auto_name = globals()
        input_filenames = find_out_files(self.inputs.prepared_file_folder.value,self.inputs.except_filenames.get_list())

        for name in input_filenames:
            auto_name[name] = SinglefileData(
                                file=os.path.join(self.inputs.prepared_file_folder.value,name)).store() 
        if 'inpsd.dat' not in input_filenames:#nothing in inpsd dict
            #note that we take the inpsd.dat first, that means if we have both inpsd dict and inpsd.dat file we only use inpsd.dat file 
            # Create input file: inpsd.dat
            with folder.open(self.options.input_filename, 'a+') as f:
                for flag in self.inputs.inpsd.attributes_keys():
                    f.write(f'{flag}'+f'    {self.inputs.inpsd[flag]}\n')
                for name in input_filenames:
                    f.write(f'{name}    ./{name}\n')               
        local_list = []        
        for name in input_filenames:
            #I believe all our user are kind people, they will not do evail things with eval() function :-) , ToDo: replace eval() here to make sure the safity
            local_list.append((eval(name).uuid,eval(name).filename,eval(name).filename))
        calcinfo.local_copy_list = local_list

        input_retrieve_list_name = self.inputs.retrieve_list_name
        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = []  # note that nothing need here for SD
        codeinfo.code_uuid = self.inputs.code.uuid
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = input_retrieve_list_name.get_list()
        return calcinfo

