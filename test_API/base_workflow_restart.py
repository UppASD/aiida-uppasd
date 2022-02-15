# -*- coding: utf-8 -*-
"""Base workchain"""
from aiida import orm
import aiida
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import  run,submit,ToContext, if_, while_, BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode
from aiida.plugins import CalculationFactory, GroupFactory
from aiida.orm import Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
from aiida_uppasd.workflows.base_restart import ASDBaseRestartWorkChain
import os
aiida.load_profile()


input_uppasd = {
    'inpsd_dict' :orm.Dict(dict={
        'simid': Str('sky'),
        'ncell': Str('30 30 1'),
        'BC': Str('P         P         0 '),
        'cell': Str('''1.00000 0.00000 0.00000
                0.00000 1.00000 0.00000
                0.00000 0.00000 1.00000'''),
        'do_prnstruct':      Int(1),
        'maptype': Int(2),
        'SDEalgh': Int(1),
        'Initmag': Int(3),
        'ip_mode': Str('Y'),
        'qm_svec': Str('1   -1   0 '),
        'qm_nvec': Str('0  0  1'),
        'mode': Str('S'),
        'temp': Float(300.000),
        'damping': Float(0.500),
        'Nstep': Int(1000),
        'timestep': Str('1.000d-16'),
        'hfield': Str('0.0 0.0 -150.0 '),
        'skyno': Str('Y'),
        'qpoints': Str('F'),
        'plotenergy': Int(1),
        'do_avrg': Str('Y'),
        #new added flags
        'do_tottraj':Str('Y'),
        'tottraj_step': Int(1),
    }),
    'exchange' : orm.Dict(dict={
    '1':Str('1 1  1.0       0.0       0.0      1.00000'),
    '2':Str('1 1 -1.0       0.0       0.0      1.00000'),
    '3':Str('1 1  0.0       1.0       0.0      1.00000'),
    '4':Str('1 1  0.0      -1.0       0.0      1.00000'),
    }),
    'num_machines' :orm.Int(1),
    'num_mpiprocs_per_machine' :orm.Int(12),
    'max_wallclock_seconds' :orm.Int(30),
    #'code' :Code.get_from_string('uppasd_dev@uppasd_local'),
    'code' :Code.get_from_string('uppasd_nsc_2021_test@nsc_uppasd_2021'),
    'input_filename' : orm.Str('inpsd.dat'),
    'parser_name' :orm.Str('UppASD_core_parsers'),
    'label' : orm.Str('uppasd_base_workflow_demo'),
    'description' :orm.Str('Test base workflow'),
    'prepared_file_folder' :Str(os.path.join(os.getcwd(),'demo3_input')),
    'except_filenames':List(list = []),
    'retrieve_list_name':List(list=[('*.out','.', 0)]),
}


process = submit(ASDBaseRestartWorkChain,**input_uppasd)
print("Submitted ASDBaseRestartWorkchain with PK {}".format(process.pk))

