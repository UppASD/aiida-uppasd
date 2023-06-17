# -*- coding: utf-8 -*-
"""Base workchain"""
import os

import numpy as np

import aiida
from aiida import orm
from aiida.engine import submit

from aiida_uppasd.workflows.skyrmions_pd_and_graph import UppASDSkyrmionsWorkflow

aiida.load_profile()
current_path = os.getcwd()

e_B = []
for i in np.linspace(0, 8, 1):
    e_B.append(f'0.0 0.0 {i}')

input_uppasd = {
    'inpsd_skyr':
    orm.Dict(
        dict={
            'simid':
            orm.Str('skyr_demo'),
            'ncell':
            orm.Str('96    96       1'),
            'BC':
            orm.Str('P         P         0 '),
            'cell':
            orm.Str(
                """1.00000 0.00000 0.00000
                0.00000 1.00000 0.00000
                0.00000 0.00000 1.00000"""
            ),
            'do_prnstruct':
            orm.Int(2),
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
            'damping':
            orm.Float(0.5),
            'Nstep':
            orm.Int(1000),
            'timestep':
            orm.Str('1.000e-15'),
            'qpoints':
            orm.Str('F'),
            'plotenergy':
            orm.Int(1),
            'do_avrg':
            orm.Str('Y'),
            'skyno':
            orm.Str('T'),
            'do_tottraj':
            orm.Str('Y'),
            'tottraj_step':
            orm.Int(100),
            'do_cumu':
            orm.Str('Y'),
            'cumu_buff':
            orm.Int(10),
            'cumu_step':
            orm.Int(50),
        }
    ),
    'num_machines':
    orm.Int(1),
    'num_mpiprocs_per_machine':
    orm.Int(1),
    'max_wallclock_seconds':
    orm.Int(59 * 60),
    # "code": Code.get_from_string("uppasd_dev@uppasd_local"),
    'code':
    Code.get_from_string('dev_pdc@pdc_dardel_dev'),
    #"code": Code.get_from_string("uppasd_nsc_2021_test@nsc_uppasd_2021"),
    'input_filename':
    orm.Str('inpsd.dat'),
    'parser_name':
    orm.Str('UppASD_core_parsers'),
    'label':
    orm.Str('uppasd_base_workflow_demo'),
    'description':
    orm.Str('Test base workflow'),
    'prepared_file_folder':
    orm.Str(os.path.join(os.getcwd(), 'skyr_input')),
    'except_filenames':
    orm.List(list=[]),
    'retrieve_list_name':
    orm.List(list=[('*', '.', 0), ('*.json', '.', 0)]),
    # "temperatures": List(list=[0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,200,250,300]),
    # "external_fields": List(list=['0.0 0.0    0.0',  '0.0 0.0 -20.0', '0.0 0.0 -40.0', '0.0 0.0 -60.0', '0.0 0.0 -80.0', '0.0 0.0 -100.0',  '0.0 0.0 -120.0', '0.0 0.0 -140.0', '0.0 0.0 -160.0', '0.0 0.0 -180.0','0.0 0.0 -200.0','0.0 0.0 -250.0']),
    # "temperatures": List(list=[0.01,10,50,60]),
    'external_fields':
    orn.List(list=e_B),
    'temperatures':
    orm.List(list=list(np.linspace(0.001, 300, 60))),
    'plot_dir':
    orm.Str(current_path),
    'sk_plot':
    orm.Int(1),
    'sk_number_plot':
    orm.Int(1),
    'average_magnetic_moment_plot':
    orm.Int(1),
    'average_specific_heat_plot':
    orm.Int(1),
    'plot_individual':
    orm.Int(0),
    'plot_individual':
    orm.Int(0),
}

#'tasks':List(list=[ 'stiffness','lswt'

job_node = submit(UppASDSkyrmionsWorkflow, **input_uppasd)

print(f'UppASDSkyrmionsWorkflow submitted, PK: {job_node.pk}')
with open('UppASDSkyrmionsWorkflow_jobPK.csv', 'w') as f:
    f.write(str(job_node.pk))
