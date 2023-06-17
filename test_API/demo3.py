#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:15:05 2021

@author: qichen
"""
import os

from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import CalculationFactory

load_profile()
code = orm.Code.get_from_string('uppasd_dev@uppasd_local')
#code = Code.get_from_string('uppasd_nsc_2021@nsc_uppasd_2021')
aiida_uppasd = CalculationFactory('uppasd.uppasd_calculation')
builder = aiida_uppasd.get_builder()

#pre-prepared files folder:
prepared_file_folder = orm.Str(os.path.join(os.getcwd(), 'demo3_input'))
except_filenames = orm.List(list=[])

# inpsd.dat file selection
inpsd_dict = {
    'simid': orm.Str('sky'),
    'ncell': orm.Str('30 30 1'),
    'BC': orm.Str('P         P         0 '),
    'cell': orm.Str('1.00000 0.00000 0.00000\n0.00000 1.00000 0.00000\n0.00000 0.00000 1.00000'),
    'do_prnstruct': orm.Int(1),
    'maptype': orm.Int(2),
    'SDEalgh': orm.Int(1),
    'Initmag': orm.Int(3),
    'ip_mode': orm.Str('Y'),
    'qm_svec': orm.Str('1   -1   0 '),
    'qm_nvec': orm.Str('0  0  1'),
    'mode': orm.Str('S'),
    'temp': orm.Float(300.000),
    'damping': orm.Float(0.500),
    'Nstep': orm.Int(1000),
    'timestep': orm.Str('1.000d-16'),
    'hfield': orm.Str('0.0 0.0 -150.0 '),
    'skyno': orm.Str('Y'),
    'qpoints': orm.Str('F'),
    'plotenergy': orm.Int(1),
    'do_avrg': orm.Str('Y'),
    #new added flags
    'do_tottraj': orm.Str('Y'),
    'tottraj_step': orm.Int(1),
}

exchange = {
    '1': orm.Str('1 1  1.0       0.0       0.0      1.00000'),
    '2': orm.Str('1 1 -1.0       0.0       0.0      1.00000'),
    '3': orm.Str('1 1  0.0       1.0       0.0      1.00000'),
    '4': orm.Str('1 1  0.0      -1.0       0.0      1.00000'),
}

r_l = orm.List(list=[('*.out', '.', 0)])
#we could use this to retrieved all .out file to aiida

# set up calculation
inpsd_dict = orm.Dict(dict=inpsd_dict)
exchange_dict = orm.Dict(dict=exchange)
builder.code = code
builder.prepared_file_folder = prepared_file_folder
builder.except_filenames = except_filenames
builder.inpsd_dict = inpsd_dict
builder.exchange = exchange_dict
builder.retrieve_list_name = r_l
builder.metadata.options.resources = {'num_machines': 1, 'num_mpiprocs_per_machine': 16}
builder.metadata.options.max_wallclock_seconds = 60 * 30
builder.metadata.options.input_filename = 'inpsd.dat'
builder.metadata.options.parser_name = 'uppasd.uppasd_parser'
builder.metadata.label = 'Demo3'
builder.metadata.description = 'Test demo3 for UppASD-AiiDA'
job_node = submit(builder)
print(f'Job submitted, PK: {job_node.pk}')
