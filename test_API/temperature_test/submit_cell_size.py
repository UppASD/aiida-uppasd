# -*- coding: utf-8 -*-
"""Base workchain"""
import os
from aiida import orm, load_profile
from aiida.engine import submit
from aiida_uppasd.workflows.temperature_cell_size import MCVariableCellWorkchain

load_profile()

# get current path, it could change to whatever suitable path
current_path = os.getcwd()
# controllable input list
MCVariableCellWorkchain_input = {
    'N_list':
    orm.List(list=[8, 12, 16, 20, 24, 28, 32]),
    'temp_list':
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
    'plot_dir':
    orm.Str(current_path),
}

builder = MCVariableCellWorkchain.get_builder()

process = submit(builder, **MCVariableCellWorkchain_input)

print(f'MCVariableCellWorkchain submitted, PK: {process.pk}')
