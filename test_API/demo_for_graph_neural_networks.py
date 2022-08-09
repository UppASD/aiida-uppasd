#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:15:05 2021

@author: qichen
"""
import os
import itertools
from aiida import orm, load_profile
from aiida.plugins import CalculationFactory
from aiida.engine import submit, workfunction

load_profile()


def calculate_single_trajectory(temp: orm.Float, hfield: orm.Str):
    """
    Calculate the trajectory for a given temperature and magnetic field.

    :param temp: value of the temperature
    :type temp: orm.Float
    :param hfield: value of the magnetic field
    :type hfield: orm.Str
    """
    code = orm.Code.get_from_string('uppasd_nsc_2021_test@nsc_uppasd_2021')
    #code = Code.get_from_string('uppasd_nsc_2021@nsc_uppasd_2021')
    aiida_uppasd = CalculationFactory('uppasd.uppasd_calculation')
    builder = aiida_uppasd.get_builder()
    #pre-prepared files folder:
    prepared_file_folder = orm.Str(os.path.join(os.getcwd(), 'demo7_input'))
    except_filenames = orm.List(list=[])
    # inpsd.dat file selection
    inpsd_dict = {
        'simid':
        orm.Str('graphnetwork'),
        'ncell':
        orm.Str('30 30 1'),
        'BC':
        orm.Str('P         P         0 '),
        'cell':
        orm.Str(
            '''1.00000 0.00000 0.00000
                0.00000 1.00000 0.00000
                0.00000 0.00000 1.00000'''
        ),
        'do_prnstruct':
        orm.Int(1),
        'maptype':
        orm.Int(2),
        'SDEalgh':
        orm.Int(1),
        'Initmag':
        orm.Int(3),
        'ip_mode':
        orm.Str('Y'),
        'qm_svec':
        orm.Str('1   -1   0 '),
        'qm_nvec':
        orm.Str('0  0  1'),
        'mode':
        orm.Str('S'),
        'temp':
        temp,
        'damping':
        orm.Float(0.500),
        'Nstep':
        orm.Int(10000),
        'timestep':
        orm.Str('1.000d-17'),
        'hfield':
        hfield,
        'skyno':
        orm.Str('Y'),
        'qpoints':
        orm.Str('F'),
        'plotenergy':
        orm.Int(1),
        'do_avrg':
        orm.Str('Y'),
        #new added flags
        'do_tottraj':
        orm.Str('Y'),
        'tottraj_step':
        orm.Int(10),
    }

    r_l = orm.List(list=[('*.out', '.', 0)])
    #we could use this to retrieved all .out file to aiida

    # set up calculation
    inpsd = orm.Dict(dict=inpsd_dict)
    builder.code = code
    builder.prepared_file_folder = prepared_file_folder
    builder.except_filenames = except_filenames
    builder.inpsd = inpsd
    builder.retrieve_list_name = r_l
    builder.metadata.options.resources = {
        'num_machines': 1,
        'num_mpiprocs_per_machine': 8,
    }
    builder.metadata.options.max_wallclock_seconds = 60 * 55
    builder.metadata.options.input_filename = 'inpsd.dat'
    builder.metadata.options.parser_name = 'uppasd.uppasd_parser'
    builder.metadata.label = 'Demo7'
    builder.metadata.description = 'Test demo7 for UppASD-AiiDA'
    job_node = submit(builder)
    print(f'Job submitted, PK: {job_node.pk}')


@workfunction
def prepare_for_graph_build(t_list: orm.List, h_list: orm.List):
    """
    Loop over the temperature and field

    :param t_list: list with the temperature data
    :type t_list: orm.List
    :param h_list: list with the field data
    :type h_list: orm.List
    """
    for temperature, hfield in itertools.product(
        t_list.attributes['list'],
        h_list.attributes['list'],
    ):
        calculate_single_trajectory(orm.Float(temperature), orm.Str(hfield))


temp_list = orm.List(list=list(range(0, 300, 5)))
hfield_l = []
for i in range(-150, 150, 10):
    hfield_l.append(f'0.0 0.0 {i}')
hfield_list = orm.List(list=hfield_l)

prepare_for_graph_build(temp_list, hfield_list)
