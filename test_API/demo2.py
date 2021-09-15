#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:15:05 2021

@author: qichen
"""
from aiida.plugins import DataFactory, CalculationFactory
from aiida.orm import Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
import aiida
import os
from aiida.engine import submit

aiida.load_profile()
code = Code.get_from_string('uppasd_dev@uppasd_local')
aiida_uppasd = CalculationFactory('UppASD_core_calculations')
builder = aiida_uppasd.get_builder()


#pre-prepared files
dmdata = SinglefileData(
    file=os.path.join(os.getcwd(), "input_files2", 'dmdata'))
jij = SinglefileData(
    file=os.path.join(os.getcwd(), "input_files2", 'jij'))
momfile = SinglefileData(
    file=os.path.join(os.getcwd(), "input_files2", 'momfile'))
posfile = SinglefileData(
    file=os.path.join(os.getcwd(), "input_files2", 'posfile'))
qfile = SinglefileData(
    file=os.path.join(os.getcwd(), "input_files2", 'qfile'))


# inpsd.dat file selection
simid = Str('SCsurf_T')

ncell = Str('128 128 1')

BC = Str('P         P         0 ')

cell = Str('''1.00000 0.00000 0.00000
0.00000 1.00000 0.00000
0.00000 0.00000 1.00000''')


do_prnstruct = Int(2)
maptype = Int(2)
SDEalgh = Int(1)
Initmag = Int(3)
ip_mode = Str('Q')
qm_svec = Str('1   -1   0 ')

qm_nvec = Str('0  0  1')

mode = Str('S')
temp = Float(0.000)
damping = Float(0.500)
Nstep = Int(5000)
timestep = Str('1.000d-15')
qpoints = Str('F')
plotenergy = Int(1)
do_avrg = Str('Y')


r_l = List(list=[f'coord.{simid.value}.out',
                 f'qm_minima.{simid.value}.out',
                 f'qm_sweep.{simid.value}.out',
                 'qpoints.out',
                 f'totenergy.{simid.value}.out',
                 f'averages.{simid.value}.out',
                 'fort.2000',
                 'inp.{simid.value}.yaml',
                 'qm_restart.{simid.value}.out',
                 'restart.{simid.value}.out'])


# set up calculation
inpsd = Dict()
inpsd.set_attribute('simid', simid)
inpsd.set_attribute('ncell', ncell)
inpsd.set_attribute('BC', BC)
inpsd.set_attribute('cell', cell)
inpsd.set_attribute('do_prnstruct', do_prnstruct)
inpsd.set_attribute('maptype', maptype)
inpsd.set_attribute('SDEalgh', SDEalgh)
inpsd.set_attribute('Initmag', Initmag)
inpsd.set_attribute('ip_mode', ip_mode)
inpsd.set_attribute('qm_svec', qm_svec)
inpsd.set_attribute('qm_nvec', qm_nvec)
inpsd.set_attribute('mode', mode)
inpsd.set_attribute('temp', temp)
inpsd.set_attribute('damping', damping)
inpsd.set_attribute('Nstep', Nstep)
inpsd.set_attribute('timestep', timestep)
inpsd.set_attribute('qpoints', qpoints)
inpsd.set_attribute('plotenergy', plotenergy)
inpsd.set_attribute('do_avrg', do_avrg)
#inpsd.set_attribute()


builder.code = code
builder.dmdata = dmdata
builder.jij = jij
builder.momfile = momfile
builder.posfile = posfile
builder.qfile = qfile
builder.inpsd = inpsd
builder.retrieve_list_name = r_l
builder.inpsd_dat_exist = Int(0)
builder.metadata.options.resources = {'num_machines': 1}
builder.metadata.options.max_wallclock_seconds = 120
builder.metadata.options.input_filename = 'inpsd.dat'
builder.metadata.options.parser_name = 'UppASD_core_parsers'
builder.metadata.label = 'Demo2'
builder.metadata.description = 'Test demo2 for UppASD-AiiDA'
job_node = submit(builder)
print('Job submitted, PK: {}'.format(job_node.pk))
