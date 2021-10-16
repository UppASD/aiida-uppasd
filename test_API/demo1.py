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
prepared_file_folder = Str(os.path.join(os.getcwd(),'demo1_input'))
except_filenames = List(list = [])

r_l = List(list=[('*.out','.', 0)])  
#we could use this to retrived all .out file to aiida


# set up calculation
builder.code = code
builder.prepared_file_folder = prepared_file_folder
builder.except_filenames = except_filenames
builder.retrieve_list_name = r_l
builder.metadata.options.resources = {'num_machines': 1,'num_mpiprocs_per_machine':16}
builder.metadata.options.max_wallclock_seconds = 600
builder.metadata.options.parser_name = 'UppASD_core_parsers'
builder.metadata.label = 'Demo5'
builder.metadata.description = 'Test demo5 for UppASD-AiiDA'
job_node = submit(builder)
print('Job submitted, PK: {}'.format(job_node.pk))
