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
from aiida_uppasd.workflows.Tempture_CellSize import MCVariableCellWorkchain

aiida.load_profile()
import os

# get current path, it could change to whatever suitable path
current_path = os.getcwd()
# controllable input list
MCVariableCellWorkchain_input = {
    'N_list':
    List(list=[8, 12, 16, 20, 24, 28, 32]),
    'temp_list':
    List(list=[
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
    'plot_dir':
    Str(current_path),
}

builder = MCVariableCellWorkchain.get_builder()

process = submit(builder, **MCVariableCellWorkchain_input)

print(f'MCVariableCellWorkchain submitted, PK: {process.pk}')
