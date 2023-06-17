# -*- coding: utf-8 -*-
"""Base workchain"""
import os

import numpy as np

import aiida
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import (
    BaseRestartWorkChain,
    ExitCode,
    ProcessHandlerReport,
    ToContext,
    if_,
    process_handler,
    run,
    submit,
    while_,
)
from aiida.orm import ArrayData, Bool, Code, Dict, Float, FolderData, Int, List, RemoteData, SinglefileData, Str, XyData
from aiida.plugins import CalculationFactory, GroupFactory

from aiida_uppasd.workflows.skyrmions_pd_and_graph import UppASDSkyrmionsWorkflow

aiida.load_profile()
current_path = os.getcwd()

e_B = []
for i in np.linspace(0, 6, 10):
    e_B.append(f'0.0 0.0 {i}')

input_uppasd = {
    'inpsd_skyr':
    orm.Dict(
        dict={
            'simid':
            Str('skyr_demo'),
            'ncell':
            Str('128       128         1'),
            'BC':
            Str('P         P         0 '),
            'cell':
            Str(
                """0.1830127019      0.6830127019      0.000000
                0.6830127019      0.1830127019      0.000000
                0.0000000000      0.0000000000      1.000000"""
            ),
            'Sym':
            Int(0),
            'do_prnstruct':
            Int(2),
            'maptype':
            Int(1),
            'SDEalgh':
            Int(1),
            'Initmag':
            Int(1),
            'ip_mode':
            Str('H'),
            'mode':
            Str('S'),
            'damping':
            Float(0.023),
            'Nstep':
            Int(1000),
            'timestep':
            Str('1.000e-16'),
            'qpoints':
            Str('F'),
            'plotenergy':
            Int(1),
            'do_avrg':
            Str('Y'),
            'skyno':
            Str('T'),
            'do_tottraj':
            Str('Y'),
            'tottraj_step':
            Int(100),
            'do_cumu':
            Str('Y'),
            'cumu_step':
            Int(20),
            'cumu_buff':
            Int(10),
            'alat':
            Str('3.84e-10'),
        }
    ),
    'num_machines':
    orm.Int(1),
    'num_mpiprocs_per_machine':
    orm.Int(16),
    'max_wallclock_seconds':
    orm.Int(120 * 60),
    #"code": Code.get_from_string("uppasd_dev@uppasd_local"),
    'code':
    Code.get_from_string('dev_pdc@pdc_dardel_dev'),
    #"code": Code.get_from_string("uppasd_nsc_2021_test@nsc_uppasd_2021"),
    #"code": Code.get_from_string("uppasd_nsc_2021@nsc_uppasd_2021"),
    'input_filename':
    orm.Str('inpsd.dat'),
    'parser_name':
    orm.Str('UppASD_core_parsers'),
    'label':
    orm.Str('uppasd_base_workflow_demo'),
    'description':
    orm.Str('Test base workflow'),
    'prepared_file_folder':
    Str(os.path.join(os.getcwd(), 'skyr_input')),
    'except_filenames':
    List(list=['dmdata*']),
    'retrieve_list_name':
    List(list=[('*', '.', 0), ('*.json', '.', 0)]),
    'temperatures':
    List(list=list(np.linspace(0.001, 150, 10))),
    'external_fields':
    List(list=e_B),
    # "temperatures": List(list=[0.001,20,40,60,80]),
    #"external_fields": List(list=['0.0 0.0 10.0']),
    'plot_dir':
    Str(current_path),
    'sk_plot':
    Int(1),
    'sk_number_plot':
    Int(1),
    'average_magnetic_moment_plot':
    Int(1),
    'average_specific_heat_plot':
    Int(1),
    'plot_individual':
    Int(0),
    'plot_individual':
    Int(0),
}

job_node = submit(UppASDSkyrmionsWorkflow, **input_uppasd)

print(f'UppASDSkyrmionsWorkflow submitted, PK: {job_node.pk}')
with open('UppASDSkyrmionsWorkflow_jobPK.csv', 'w') as f:
    f.write(str(job_node.pk))
