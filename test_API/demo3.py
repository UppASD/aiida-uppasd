#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:15:05 2021

@author: qichen
"""
from aiida.plugins import DataFactory, CalculationFactory
from aiida.engine import run
from aiida.orm import Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
import numpy as np
import aiida
import os
from aiida.engine import submit

aiida.load_profile()
code = Code.get_from_string('uppasd_dev@uppasd_local')
#code = Code.get_from_string('uppasd_nsc_2021@nsc_uppasd_2021')
aiida_uppasd = CalculationFactory('UppASD_core_calculations')
builder = aiida_uppasd.get_builder()

#pre-prepared files folder:
prepared_file_folder = Str(os.path.join(os.getcwd(), 'demo3_input'))
except_filenames = List(list=[])

# inpsd.dat file selection
inpsd_dict = {
    'simid':
    Str('sky'),
    'ncell':
    Str('30 30 1'),
    'BC':
    Str('P         P         0 '),
    'cell':
    Str('1.00000 0.00000 0.00000\n0.00000 1.00000 0.00000\n0.00000 0.00000 1.00000'
        ),
    'do_prnstruct':
    Int(1),
    'maptype':
    Int(2),
    'SDEalgh':
    Int(1),
    'Initmag':
    Int(3),
    'ip_mode':
    Str('Y'),
    'qm_svec':
    Str('1   -1   0 '),
    'qm_nvec':
    Str('0  0  1'),
    'mode':
    Str('S'),
    'temp':
    Float(300.000),
    'damping':
    Float(0.500),
    'Nstep':
    Int(1000),
    'timestep':
    Str('1.000d-16'),
    'hfield':
    Str('0.0 0.0 -150.0 '),
    'skyno':
    Str('Y'),
    'qpoints':
    Str('F'),
    'plotenergy':
    Int(1),
    'do_avrg':
    Str('Y'),
    #new added flags
    'do_tottraj':
    Str('Y'),
    'tottraj_step':
    Int(1),
}

exchange = {
    '1': Str('1 1  1.0       0.0       0.0      1.00000'),
    '2': Str('1 1 -1.0       0.0       0.0      1.00000'),
    '3': Str('1 1  0.0       1.0       0.0      1.00000'),
    '4': Str('1 1  0.0      -1.0       0.0      1.00000'),
}

r_l = List(list=[('*.out', '.', 0)])
#we could use this to retrived all .out file to aiida

# set up calculation
inpsd_dict = Dict(dict=inpsd_dict)
exchange_dict = Dict(dict=exchange)
builder.code = code
builder.prepared_file_folder = prepared_file_folder
builder.except_filenames = except_filenames
builder.inpsd_dict = inpsd_dict
builder.exchange = exchange_dict
builder.retrieve_list_name = r_l
builder.metadata.options.resources = {
    'num_machines': 1,
    'num_mpiprocs_per_machine': 16
}
builder.metadata.options.max_wallclock_seconds = 60 * 30
builder.metadata.options.input_filename = 'inpsd.dat'
builder.metadata.options.parser_name = 'UppASD_core_parsers'
builder.metadata.label = 'Demo3'
builder.metadata.description = 'Test demo3 for UppASD-AiiDA'
job_node = submit(builder)
print(f'Job submitted, PK: {job_node.pk}')
