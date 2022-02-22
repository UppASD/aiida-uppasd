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
from aiida_uppasd.workflows.temperature import UppASDTemperatureWorkflow
import os

aiida.load_profile()


input_uppasd = {
    "inpsd_temp": orm.Dict(
        dict={
            "simid": Str("bccFe100"),
            "ncell": Str("12 12 12"),
            "BC": Str("P         P         P "),
            "cell": Str(
                """1.00000 0.00000 0.00000
                0.00000 1.00000 0.00000
                0.00000 0.00000 1.00000"""
            ),
            "sym": Int(1),
            "maptype": Int(1),
            "Initmag": Int(3),
            "alat": Float(2.87e-10),
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
    "prepared_file_folder": Str(os.path.join(os.getcwd(), "task1_input")),
    "except_filenames": List(list=[]),
    "retrieve_list_name": List(list=[("*.out", ".", 0), ("*.json", ".", 0)]),
    "tasks": List(list=["mc", "thermodynamics"]),
    "temperatures": List(
        list=[
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
        ]
    ),
}

#'tasks':List(list=[ 'stiffness','lswt'])

builder = UppASDTemperatureWorkflow.get_builder()
job_node = submit(builder, **input_uppasd)

print("UppASDTemperatureWorkflow submitted, PK: {}".format(job_node.pk))
with open("UppASDTemperatureWorkflow_jobPK.csv", "w") as f:
    f.write(str(job_node.pk))
