# -*- coding: utf-8 -*-
"""Base workchain"""
from aiida import orm
import aiida
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import (
    run,
    submit,
    ToContext,
    if_,
    while_,
    BaseRestartWorkChain,
    process_handler,
    ProcessHandlerReport,
    ExitCode,
)
from aiida.plugins import CalculationFactory, GroupFactory
from aiida.orm import (
    Code,
    SinglefileData,
    Int,
    Float,
    Str,
    Bool,
    List,
    Dict,
    ArrayData,
    XyData,
    SinglefileData,
    FolderData,
    RemoteData,
)
from aiida_uppasd.workflows.skyrmions_pd_and_graph import UppASDSkyrmionsWorkflow
import os
import numpy as np

aiida.load_profile()
current_path = os.getcwd()

e_B = []
for i in np.linspace(0,8,1):
    e_B.append('0.0 0.0 {}'.format(i))

input_uppasd = {
    "inpsd_skyr": orm.Dict(
        dict={
            "simid": Str("skyr_demo"),
            "ncell": Str("96    96       1"),
            "BC": Str("P         P         0 "),
            "cell": Str(
                """1.00000 0.00000 0.00000
                0.00000 1.00000 0.00000
                0.00000 0.00000 1.00000"""
            ),
            'do_prnstruct': Int(2),
            'maptype': Int(2),
            'SDEalgh': Int(1),
            'Initmag': Int(3),
            'ip_mode': Str('Y'),
            'qm_svec': Str('1   -1   0 '),
            'qm_nvec': Str('0  0  1'),
            'mode': Str('S'),
            'damping': Float(0.5),
            'Nstep': Int(1000),
            'timestep': Str('1.000e-15'),
            'qpoints': Str('F'),
            'plotenergy': Int(1),
            'do_avrg': Str('Y'),
            'skyno': Str('T'),
            'do_tottraj': Str('Y'),
            'tottraj_step': Int(100),
            'do_cumu':Str('Y'),
            'cumu_buff' : Int(10),
            'cumu_step'  :  Int(50),
        }
    ),
    "num_machines": orm.Int(1),
    "num_mpiprocs_per_machine": orm.Int(1),
    "max_wallclock_seconds": orm.Int(59*60),
    # "code": Code.get_from_string("uppasd_dev@uppasd_local"),
    "code": Code.get_from_string("dev_pdc@pdc_dardel_dev"),
    #"code": Code.get_from_string("uppasd_nsc_2021_test@nsc_uppasd_2021"),
    "input_filename": orm.Str("inpsd.dat"),
    "parser_name": orm.Str("UppASD_core_parsers"),
    "label": orm.Str("uppasd_base_workflow_demo"),
    "description": orm.Str("Test base workflow"),
    "prepared_file_folder": Str(os.path.join(os.getcwd(), "skyr_input")),
    "except_filenames": List(list=[]),
    "retrieve_list_name": List(list=[("*", ".", 0), ("*.json", ".", 0)]),
    # "temperatures": List(list=[0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,200,250,300]),
    # "external_fields": List(list=['0.0 0.0    0.0',  '0.0 0.0 -20.0', '0.0 0.0 -40.0', '0.0 0.0 -60.0', '0.0 0.0 -80.0', '0.0 0.0 -100.0',  '0.0 0.0 -120.0', '0.0 0.0 -140.0', '0.0 0.0 -160.0', '0.0 0.0 -180.0','0.0 0.0 -200.0','0.0 0.0 -250.0']),
    # "temperatures": List(list=[0.01,10,50,60]),
    "external_fields": List(list=e_B),
    "temperatures": List(list=list(np.linspace(0.001,300,60))),
    "plot_dir": Str(current_path),  
    'sk_plot': Int(1),
    'sk_number_plot':Int(1),
    'average_magnetic_moment_plot':Int(1),
    'average_specific_heat_plot':Int(1),
    'plot_individual':Int(0),
    'plot_individual':Int(0),
}

#'tasks':List(list=[ 'stiffness','lswt'

job_node = submit(UppASDSkyrmionsWorkflow, **input_uppasd)

print("UppASDSkyrmionsWorkflow submitted, PK: {}".format(job_node.pk))
with open("UppASDSkyrmionsWorkflow_jobPK.csv", "w") as f:
    f.write(str(job_node.pk))
