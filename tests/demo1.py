#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:15:05 2021

@author: qichen

qichenx@kth.se


"""

import os
import sys
import time

from code_computer_setting import get_code, get_computer, get_path_to_executable

from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import CalculationFactory

load_profile()

# Try to find the UppASD executable file
while True:
    try:
        file_path = get_path_to_executable('sd')
        print(f'! Find UppASD executable file sd at: \n {file_path}')
        break
    except BaseException:  # pylint: disable=broad-except
        _msg = """
It seem we could not find UppASD executable file(../../../sd) automatically.
Could you please choose flag to continue:
(1: Run again,2: Type UppASD path,3: Exit demo)
Your answer:
        """
        user_input_path = input(_msg)
        if user_input_path == 1:
            pass
        elif user_input_path == 2:
            file_path = input('\n Enter the path: \n')
            print(f'Your UppASD path is: {file_path}')
            break
        elif user_input_path == 3:
            sys.exit()
        else:
            print('\nSelect from {1,2,3} :-)\n')

while True:
    _msg = """
Enter your local computer name:
(a.Create new computer b.Use default name:uppasd_local)
    """
    user_input_flag1 = input(_msg)
    if user_input_flag1 == 'a':
        computer_name = input('\n Enter the name: \n')
        new_computer = get_computer(computer_name)
        workdir = new_computer.get_workdir()
        _msg = f"""
        Your demo workdir is: {workdir}
        you could delete it after testing
        """
        print(_msg)
        break
    if user_input_flag1 == 'b':
        computer_name = 'uppasd_local'
        workdir = get_computer(computer_name).get_workdir()
        _msg = f"""
Your demo workdir is: {workdir}
you could delete it after testing
        """
        print(_msg)
        break
    print('\nSelect from {a,b} :-)\n')

while True:

    _msg = """
Enter your local code name:
(a.Create new code b.Use default name:default:uppasd_dev)
    """
    user_input_flag2 = input(_msg)
    if user_input_flag2 == 'a':
        code_name = input('\n Enter the name: \n')
        code = get_code(code_name, file_path, new_computer)
        break
    if user_input_flag2 == 'b':
        code_name = 'uppasd_dev'
        break
    print('\nSelect from {a,b} :-)\n')

print(f'your calculation will run on {code_name}@{computer_name}')

#choose your code here :
code = orm.Code.get_from_string(f'{code_name}@{computer_name}')
#choose calculation method from UppASD_AiiDA interface
aiida_uppasd = CalculationFactory('uppasd.uppasd_calculation')

#prepare all files like what you do for traditional UppASD calculation in the folder:
#Note that you should named UppASD input file with "inpsd" instead of "inpsd.dat" here.
prepared_file_folder = orm.Str(os.path.join(os.getcwd(), 'demo1_input'))
except_filenames = orm.List(list=[])
r_l = orm.List(list=[('*.out', '.', 0)])
#we could use this to retrieved all .out file to aiida

# set up calculation
builder = aiida_uppasd.get_builder()
builder.code = code
builder.prepared_file_folder = prepared_file_folder
builder.except_filenames = except_filenames
builder.retrieve_list_name = r_l
builder.metadata.options.resources = {
    'num_machines': 1,
    'num_mpiprocs_per_machine': 2,
}
builder.metadata.options.max_wallclock_seconds = 600
builder.metadata.options.parser_name = 'uppasd.uppasd_parser'
builder.metadata.label = 'Demo5'
builder.metadata.description = 'Test demo5 for UppASD-AiiDA'
job_node = submit(builder)
print(f'Job submitted, PK: {job_node.pk}')

while True:
    time.sleep(2)
    print('==========demo is running========')
    job = orm.load_node(job_node.pk)
    if job.exit_status is not None:
        print('\n==========demo is done========\n')
        break

os.system(f'verdi process show {job_node.pk}')
print('\nNow you could play with the result. :-)\n')
