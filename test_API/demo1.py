#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:07:58 2021

@author: qichen
"""
#this demo offers a possibility that we could directly use traditional UppASD working folder to take the advantages from aiida

from aiida.plugins import DataFactory, CalculationFactory
from aiida.engine import run
from aiida.orm import Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
import numpy as np
import aiida
import os
aiida.load_profile()
code = Code.get_from_string('uppasd_dev@uppasd_local')
simid = Str('SCsurf_T')
r_l = List(list=[f'coord.{simid.value}.out',
                 f'qm_minima.{simid.value}.out',
                 f'qm_sweep.{simid.value}.out',
                 f'qpoints.out',
                 f'totenergy.{simid.value}.out',
                 f'averages.{simid.value}.out',
                 'fort.2000',
                 'inp.SCsurf_T.yaml',
                 'qm_restart.SCsurf_T.out',
                 'restart.SCsurf_T.out'])
dmdata = SinglefileData(
    file=os.path.join(os.getcwd(), "input_files", 'dmdata'))
jij = SinglefileData(
    file=os.path.join(os.getcwd(), "input_files", 'jij'))
momfile = SinglefileData(
    file=os.path.join(os.getcwd(), "input_files", 'momfile'))
posfile = SinglefileData(
    file=os.path.join(os.getcwd(), "input_files", 'posfile'))
qfile = SinglefileData(
    file=os.path.join(os.getcwd(), "input_files", 'qfile'))
inpsd_dat = SinglefileData(
    file=os.path.join(os.getcwd(), "input_files", 'inpsd.dat'))

inputs = {
    'code': code,
    'dmdata': dmdata,
    'jij': jij,
    'momfile': momfile,
    'posfile': posfile,
    'qfile': qfile,
    'inpsd_dat_exist': Int(1),
    'inpsd_dat': inpsd_dat,
    'retrieve_list_name': r_l,
    'metadata': {
        'options': {
            'max_wallclock_seconds': 60,
            'resources': {'num_machines': 1},
            'input_filename': 'inpsd.dat',
            'parser_name': 'UppASD_core_parsers',
        },
    },
}

result = run(CalculationFactory('UppASD_core_calculations'), **inputs)
