# -*- coding: utf-8 -*-
"""Base workchain"""
import os

from aiida import load_profile, orm
from aiida.engine import submit

from aiida_uppasd.workflows.base_restart import ASDBaseRestartWorkChain

load_profile()

input_uppasd = {
    'inpsd_dict':
    orm.Dict(
        dict={
            'simid':
            orm.Str('sky'),
            'ncell':
            orm.Str('30 30 1'),
            'BC':
            orm.Str('P         P         0 '),
            'cell':
            orm.Str(
                '''1.00000 0.00000 0.00000
                0.00000 1.00000 0.00000
                0.00000 0.00000 1.00000'''
            ),
            'do_prnstruct':
            orm.Int(1),
            'maptype':
            orm.Int(2),
            'SDEalgh':
            orm.Int(1),
            'Initmag':
            orm.Int(3),
            'ip_mode':
            orm.Str('Y'),
            'qm_svec':
            orm.Str('1   -1   0 '),
            'qm_nvec':
            orm.Str('0  0  1'),
            'mode':
            orm.Str('S'),
            'temp':
            orm.Float(300.000),
            'damping':
            orm.Float(0.500),
            'Nstep':
            orm.Int(1000),
            'timestep':
            orm.Str('1.000d-16'),
            'hfield':
            orm.Str('0.0 0.0 -150.0 '),
            'skyno':
            orm.Str('Y'),
            'qpoints':
            orm.Str('F'),
            'plotenergy':
            orm.Int(1),
            'do_avrg':
            orm.Str('Y'),
            #new added flags
            'do_tottraj':
            orm.Str('Y'),
            'tottraj_step':
            orm.Int(1),
        }
    ),
    'exchange':
    orm.Dict(
        dict={
            '1': orm.Str('1 1  1.0       0.0       0.0      1.00000'),
            '2': orm.Str('1 1 -1.0       0.0       0.0      1.00000'),
            '3': orm.Str('1 1  0.0       1.0       0.0      1.00000'),
            '4': orm.Str('1 1  0.0      -1.0       0.0      1.00000'),
        }
    ),
    'num_machines':
    orm.Int(1),
    'num_mpiprocs_per_machine':
    orm.Int(12),
    'max_wallclock_seconds':
    orm.Int(30),
    #'code' :Code.get_from_string('uppasd_dev@uppasd_local'),
    'code':
    orm.Code.get_from_string('uppasd@localhost'),
    'input_filename':
    orm.Str('inpsd.dat'),
    'parser_name':
    orm.Str('uppasd.uppasd_parser'),
    'label':
    orm.Str('uppasd_base_workflow_demo'),
    'description':
    orm.Str('Test base workflow'),
    'prepared_file_folder':
    orm.Str(os.path.join(os.getcwd(), 'demo3_input')),
    'except_filenames':
    orm.List(list=[]),
    'retrieve_list_name':
    orm.List(list=[('*.out', '.', 0)]),
}

process = submit(ASDBaseRestartWorkChain, **input_uppasd)
print(f'Submitted ASDBaseRestartWorkchain with PK {process.pk}')
