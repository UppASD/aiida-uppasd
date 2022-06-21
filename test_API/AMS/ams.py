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
from aiida_uppasd.workflows.magnon_spectra import  UppASDMagnonSpectraRestartWorkflow
import os

aiida.load_profile()
current_path = os.getcwd()

input_uppasd = {
    "inpsd_ams": orm.Dict(
        dict={
            "simid": Str("bccFe100"),
            "ncell": Str("20        20        20"),
            "BC": Str("P         P         P "),
            "cell": Str(
                """1.00000 0.00000 0.00000
                0.00000 1.00000 0.00000
                0.00000 0.00000 1.00000"""
            ),
            'Mensemble': Int(1),
            'maptype': Int(2),
            'SDEalgh': Int(1),
            'Initmag': Int(3),
            'ip_mode': Str('M'),
            'ip_temp':Int(100),
            'ip_mcNstep':Int(5000),
            'qm_svec': Str('1   -1   0 '),
            'qm_nvec': Str('0  0  1'),
            'mode': Str('S'),
            'temp': Float(100),
            'damping': Float(0.01),
            'Nstep': Int(10000),
            'timestep': Str('1.000e-16'),
            'qpoints': Str('D'),
            'plotenergy': Int(1),
            'do_avrg': Str('Y'),
            'do_sc': Str('Q'),
            'do_ams': Str('Y'),
            'do_magdos': Str('Y'),
            'do_sc_proj': Str('Q'),
            'magdos_freq': Int(5000),
            'sc_step': Int(20),
            'sc_nstep': Int(5000),
            'magdos_freq': Int(200),
            'magdos_sigma': Int(30),
        }
    ),
    "num_machines": orm.Int(1),
    "num_mpiprocs_per_machine": orm.Int(16),
    "max_wallclock_seconds": orm.Int(2000),
    "code": Code.get_from_string("uppasd_dev@uppasd_local"),
    "input_filename": orm.Str("inpsd.dat"),
    "parser_name": orm.Str("UppASD_core_parsers"),
    "label": orm.Str("uppasd_base_workflow_demo"),
    "description": orm.Str("Test base workflow"),
    "prepared_file_folder": Str(os.path.join(os.getcwd(), "AMS_input")),
    "except_filenames": List(list=[]),
    "retrieve_list_name": List(list=[("*", ".", 0), ("*.json", ".", 0)]),
    "J_model": Int(-1),
    "plot_dir":Str(current_path),
    'AMSplot':Bool('True')
}

#'tasks':List(list=[ 'stiffness','lswt'

job_node = submit(UppASDMagnonSpectraRestartWorkflow, **input_uppasd)

print("UppASDAMSPlotWorkflow submitted, PK: {}".format(job_node.pk))
with open("UppASDAMSPlotWorkflow_jobPK.csv", "w") as f:
    f.write(str(job_node.pk))
