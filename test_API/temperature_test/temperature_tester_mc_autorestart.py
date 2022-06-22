# -*- coding: utf-8 -*-
"""Base workchain"""
import os
from aiida import orm, load_profile
from aiida.engine import submit
from aiida_uppasd.workflows.temperature_restart import UppASDTemperatureRestartWorkflow

load_profile()

input_uppasd = {
    'inpsd_temp':
    orm.Dict(
        dict={
            'simid':
            orm.Str('bccFe100'),
            'ncell':
            orm.Str('24 24 24'),
            'BC':
            orm.Str('P         P         P '),
            'cell':
            orm.Str(
                """1.00000 0.00000 0.00000
                0.00000 1.00000 0.00000
                0.00000 0.00000 1.00000"""
            ),
            'sym':
            orm.Int(1),
            'maptype':
            orm.Int(1),
            'Initmag':
            orm.Int(3),
            'alat':
            orm.Float(2.87e-10),
        }
    ),
    'num_machines':
    orm.Int(1),
    'num_mpiprocs_per_machine':
    orm.Int(1),
    'max_wallclock_seconds':
    orm.Int(5),
    #'code' :Code.get_from_string('uppasd_dev@uppasd_local'),
    'code':
    orm.Code.get_from_string('uppasd_nsc_2021_test@nsc_uppasd_2021'),
    'input_filename':
    orm.Str('inpsd.dat'),
    'parser_name':
    orm.Str('UppASD_core_parsers'),
    'label':
    orm.Str('uppasd_base_workflow_demo'),
    'description':
    orm.Str('Test base workflow'),
    'prepared_file_folder':
    orm.Str(os.path.join(os.getcwd(), 'task1_input')),
    'except_filenames':
    orm.List(list=[]),
    'retrieve_list_name':
    orm.List(list=[('*.out', '.', 0), ('*.json', '.', 0)]),
    'tasks':
    orm.List(list=['mc', 'thermodynamics']),
    'temperatures':
    orm.List(list=[
        0.001,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        850,
        900,
        950,
        1000,
        1100,
        1200,
        1300,
        1400,
        1500,
    ]),
}

process = submit(UppASDTemperatureRestartWorkflow, **input_uppasd)

print(f'UppASDTemperatureRestartWorkflow submitted, PK: {process.pk}')
with open('UppASDTemperatureRestartWorkflow_jobPK.csv', 'w') as f:
    f.write(str(process.pk))
