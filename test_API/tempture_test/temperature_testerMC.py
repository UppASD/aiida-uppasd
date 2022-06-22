# -*- coding: utf-8 -*-
"""Base workchain"""
import os
from aiida import orm, load_profile
from aiida.engine import submit
from aiida_uppasd.workflows.temperature import UppASDTemperatureWorkflow

load_profile()

input_uppasd = {
    'inpsd_temp':
    orm.Dict(
        dict={
            'simid':
            orm.Str('bccFe100'),
            'ncell':
            orm.Str('12 12 12'),
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
    orm.Int(16),
    'max_wallclock_seconds':
    orm.Int(2000),
    'code':
    orm.Code.get_from_string('uppasd_dev@uppasd_local'),
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

builder = UppASDTemperatureWorkflow.get_builder()
job_node = submit(builder, **input_uppasd)

print(f'UppASDTemperatureWorkflow submitted, PK: {job_node.pk}')
with open('UppASDTemperatureWorkflow_jobPK.csv', 'w') as f:
    f.write(str(job_node.pk))
