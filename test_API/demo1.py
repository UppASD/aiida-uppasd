#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:15:05 2021

@author: qichen

qichenx@kth.se


"""
    

from aiida.plugins import DataFactory, CalculationFactory
from aiida.engine import run
from aiida.orm import load_node, Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
import numpy as np
import aiida
import os,sys,time
from aiida.engine import submit
from code_computer_setting import *
aiida.load_profile()

#Try to find the UppASD executable file
while True:
    try:
        file_path  = get_path_to_executable('sd')
        print("! Find UppASD executable file sd at: \n {}".format(file_path))
        break
    except:
        user_input_path =  input("It seem we could not find UppASD executable file(../../../sd) automatically. \nCould you please choose flag to continue: \n(1: Run again,2: Type UppASD path,3: Exit demo) \nYour answer: ")
        if user_input_path == 1:
            pass
        elif user_input_path == 2:
            file_path = input("\n Enter the path: \n")
            print("Your UppASD path is: {}".format(file_path))
            break
        elif user_input_path == 3:
            sys.exit()
        else:
            print("\nSelect from {1,2,3} :-)\n")
            
while True:
    user_input_flag1 = input("\n Enter your local computer name:\n (a.Creat new computer b.Use default name:uppasd_local)\n")
    if user_input_flag1 == 'a':
        computer_name = input('\n Enter the name: \n')
        new_computer = get_computer(computer_name)
        workdir = new_computer.get_workdir()
        print("\nYour demo workdir is: {}\n, you could delete it after testing\n".format(workdir))
        break
    elif user_input_flag1 == 'b':
        computer_name = "uppasd_local"
        workdir = get_computer(computer_name).get_workdir()
        print("\nYour demo workdir is: {}\n, you could delete it after testing\n".format(workdir))
        break
    else:
        print("\nSelect from {a,b} :-)\n")
    
    
while True:
    
    user_input_flag2 = input("\n Enter your local code name:\n (a.Creat new code b.Use default name:default:uppasd_dev)\n")
    if user_input_flag2 == "a":
        code_name = input('\n Enter the name: \n')
        code = get_code(code_name,file_path,new_computer)
        break
    elif user_input_flag2 == "b":
        code_name = "uppasd_dev"
        break
    else:
        print("\nSelect from {a,b} :-)\n")
    
print("your calculation will run on {}@{}".format(code_name,computer_name))



#choose your code here :
code = Code.get_from_string('{}@{}'.format(code_name,computer_name))
#choose calculation method from UppASD_AiiDA inferface
aiida_uppasd = CalculationFactory('UppASD_core_calculations')

#prepare all files like what you do for traditional UppASD calculation in the folder:
#Note that you should named UppASD input file with "inpsd" instead of "inpsd.dat" here.
prepared_file_folder = Str(os.path.join(os.getcwd(),'demo1_input'))
except_filenames = List(list = [])
r_l = List(list=[('*.out','.', 0)])  
#we could use this to retrived all .out file to aiida


# set up calculation
builder = aiida_uppasd.get_builder()
builder.code = code
builder.prepared_file_folder = prepared_file_folder
builder.except_filenames = except_filenames
builder.retrieve_list_name = r_l
builder.metadata.options.resources = {'num_machines':1,'num_mpiprocs_per_machine':2}
builder.metadata.options.max_wallclock_seconds = 600
builder.metadata.options.parser_name = 'UppASD_core_parsers'
builder.metadata.label = 'Demo5'
builder.metadata.description = 'Test demo5 for UppASD-AiiDA'
job_node = submit(builder)
print('Job submitted, PK: {}'.format(job_node.pk))

while True:
    time.sleep(2)
    print('==========demo is running========')
    job = load_node(job_node.pk)
    if job.exit_status != None:
        print('\n==========demo is done========\n')
        break
    
os.system('verdi process show {}'.format(job_node.pk))
print('\nNow you could play with the result. :-)\n')
