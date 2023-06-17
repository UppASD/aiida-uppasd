# -*- coding: utf-8 -*-
"""Base workchain"""
import os

import numpy as np

import aiida
from aiida import orm
from aiida.engine import submit
from aiida.orm import Code, Float, Int, List, Str

from aiida_uppasd.workflows.thermal_dynamic import ThermalDynamicWorkflow

aiida.load_profile()
current_path = os.getcwd()
e_B = []
for i in range(0, 100, 10):
    e_B.append(f'0.0 0.0 {float(i / 10)}')
    #e_B.append('0.0 0.0 {}'.format(float(i / 10)))

cz = []
for i in range(32, 33, 10):
    cz.append(f'{i} {i} {i}')
    #cz.append('{} {} {}'.format(i, i, i))
input_uppasd = {
    'inpsd_temp':
    orm.Dict(
        dict={
            'simid':
            Str('bccFe100'),
            'BC':
            Str('P         P         P '),
            'cell':
            Str(
                """1.00000 0.00000 0.00000
                0.00000 1.00000 0.00000
                0.00000 0.00000 1.00000"""
            ),
            'sym':
            Int(1),
            'maptype':
            Int(1),
            'initmag':
            Int(3),
            'alat':
            Float(2.87e-10),
        }
    ),
    'num_machines':
    orm.Int(1),
    'num_mpiprocs_per_machine':
    orm.Int(1),
    'max_wallclock_seconds':
    orm.Int(59 * 60),

    #'code' :Code.get_from_string('dev_pdc@pdc_dardel_dev'),
    #'code' :Code.get_from_string('uppasd_dev@uppasd_local'),
    'code':
    Code.get_from_string('uppasd_nsc_2021_test@nsc_uppasd_2021'),
    'input_filename':
    orm.Str('inpsd.dat'),
    'parser_name':
    orm.Str('UppASD_core_parsers'),
    'label':
    orm.Str('uppasd_base_workflow_demo'),
    'description':
    orm.Str('Test base workflow'),
    'prepared_file_folder':
    Str(os.path.join(os.getcwd(), 'task1_input')),
    'except_filenames':
    List(list=[]),
    'retrieve_list_name':
    List(list=[('*.out', '.', 0), ('*.json', '.', 0)]),
    'tasks':
    List(list=['mc', 'thermodynamics']),
    'external_fields':
    List(list=e_B),
    'cell_size':
    List(list=cz),
    'plot_dir':
    Str(current_path),
    'temperatures':
    List(list=list(np.linspace(400, 1400, 30))),
}

#'tasks':List(list=[ 'stiffness','lswt'])

process = submit(ThermalDynamicWorkflow, **input_uppasd)

print(f'UppASDTemperatureRestartWorkflow submitted, PK: {process.pk}')
with open('UppASDTemperatureRestartWorkflow_jobPK.csv', 'w', encoding='utf-8') as f:
    f.write(str(process.pk))
