# -*- coding: utf-8 -*-
"""Utilities to convert between inpsd text format, Python dict, and JSON output"""

# Import numpy for array handling
from sys import stdout

import numpy as np

from aiida.tools import get_kpoints_path, spglib_tuple_to_structure


def double_fortran_to_float(data: str) -> float:
    """
    Convert Fortran double precision exponents to Python single

    :param data: string containing exponent
    :type data: str
    :return: converted string to float
    :rtype: float
    """
    return float('e-'.join(data.split('d-')))


def auto_convert(data):
    """ Convert values to optimal(...) type.

    :param data: data to be converted to proper type

    Credit: Stack Exchange
    """

    for _data_type in (int, float, double_fortran_to_float, str):
        try:
            return _data_type(data)
        except ValueError:
            pass
    return data


def quotes_for_json(string):
    """ Add double quotes around strings for JSON

    :param string: string to be decorated with double quotes
    """
    return f'"{string}"'


def json_conv(input_data):
    """ Convert data to string according to JSON format

    :param input_data: input value/list/array
    """
    if isinstance(input_data, (float, int)):
        data = input_data
    elif isinstance(input_data, str):
        data = quotes_for_json(input_data)
    elif isinstance(input_data, np.ndarray):
        data = input_data.tolist()
    elif isinstance(input_data, list):
        # Special check for lists of characters
        if isinstance(input_data[0], str) and len(input_data) == 3:
            data = '[ "' + input_data[0] + '" , "' + input_data[1] + '" , "' + input_data[2] + '" ]'
        else:
            data = input_data
    return str(' : ' + str(data) + ',')


# Define keys that need more than one value
vector_arg_list = ['hfield', 'ip_hfield', 'qm_svec', 'qm_nvec', 'jvec', 'bc', 'ncell']

# Define keys that have a special list for input
special_list = ['ip_nphase', 'ip_mcanneal', 'ntraj', 'positions', 'moments']


def inpsd_to_dict(handler):
    """Read UppASD input to Python dict

    The function defaults to single value for every key but exceptions are
    handled by the pre-defined lists "vector_arg_list" and "special_list".
    Currently all keywords except spatial temperature gradients are supported.

    :param handler: file handle to read from
    """
    inpsd_data = {}
    for line in handler:
        if len(line) > 1:
            key = line.split()[0].lower()
            if key == 'cell':
                alat = line.split()[1:4]
                blat = handler.readline().split()[0:3]
                clat = handler.readline().split()[0:3]
                data = np.asarray([
                    np.asarray(alat, dtype=float),
                    np.asarray(blat, dtype=float),
                    np.asarray(clat, dtype=float)
                ])
            elif key in special_list:
                number_lines = auto_convert(line.split()[1])
                data = []
                data.append(number_lines)
                for _ in range(number_lines):
                    data_line = handler.readline().split()
                    data.append([auto_convert(entry) for entry in data_line])
            elif key in vector_arg_list:
                data = [auto_convert(entry) for entry in line.split()[1:4]]
            else:
                data = auto_convert(line.split()[1])
            inpsd_data[key] = data
    return inpsd_data


def dict_to_inpsd(input_dict, handler=stdout):
    """ Print dict as UppASD input format

    The function prints either to the screen or to a file if defined

    :param input_dict: The dict that will be printed
    :param handler: file handle for writing to file (optional parameter)
    """

    for key in input_dict.keys():
        if key == 'cell':
            print(key, np.array2string(input_dict[key][0])[1:-1], file=handler)
            print('    ', np.array2string(input_dict[key][1])[1:-1], file=handler)
            print('    ', np.array2string(input_dict[key][2])[1:-1], file=handler)
        elif key in special_list:
            number_lines = input_dict[key][0]
            print(key, number_lines, file=handler)
            for line in range(number_lines):
                print(' '.join([str(entry) for entry in input_dict[key][line + 1]]), file=handler)
        elif key in vector_arg_list:
            print(key, ' '.join([str(entry) for entry in input_dict[key]]), file=handler)
        else:
            print(key, input_dict[key], file=handler)


def dict_to_json(input_dict, handler=stdout):
    """ Print dict on JSON format

    Hand-written routine is preferred since the standard implementation has issues
    with numpy arrays and insists on line-breaking lists ad nauseum

    The function prints either to the screen or to a file if defined

    :param input_dict: The dict that will be printed
    :param handler: file handle for writing to file (optional parameter)
    """
    print('{', file=handler)
    for key in input_dict.keys():
        print('  ', quotes_for_json(key), json_conv(input_dict[key]), file=handler)
    print('  "comment" : "Auto-converted UppASD dict"', file=handler)
    print('}', file=handler)


def read_posfile(posfile):
    """ Read the position file

    :param: file name for 'posfile'
    """
    try:
        with open(posfile, 'r', encoding='utf8') as handler:
            lines = handler.readlines()
            cell_positions = np.empty([0, 3])
            cell_numbers = []
            for line in lines:
                line_data = line.rstrip('\n').split()
                if len(line_data) > 0:
                    cell_positions = np.vstack((cell_positions, np.asarray(
                        line_data[2:5],
                        dtype=float,
                    )))
                    cell_numbers = np.append(cell_numbers, np.asarray(
                        line_data[1],
                        dtype=float,
                    ))
        return cell_positions, cell_numbers
    except IOError:
        print('Position file ', posfile, 'not accessible.')
    return None


if __name__ == '__main__':

    # Read inpsd file to dict
    with open('inpsd.dat', 'r', encoding='utf8') as file_handler:
        inpsd_dict = inpsd_to_dict(file_handler)

    # Write dict to JSON format
    with open('inpsd.json', 'w', encoding='utf8') as file_handler:
        dict_to_json(inpsd_dict, file_handler)

    # Write dict to inpsd format to screen

    # Setup structure for later use (spglib or aiida DataStructure)
    pos_exist = False
    positions = None
    numbers = None
    if 'posfile' in inpsd_dict.keys():
        positions, numbers = read_posfile(inpsd_dict['posfile'])
        pos_exist = True
    elif 'positions' in inpsd_dict.keys():
        pos_arr = np.array(inpsd_dict['positions'][1:])
        positions = pos_arr[:, 2:5]
        numbers = pos_arr[:, 1]
        pos_exist = True

    if pos_exist:
        cell = inpsd_dict['cell']
        pbc = []
        for entry in inpsd_dict['bc']:
            pbc.append(entry == 'P')
        print('pbc:', pbc)
        spgcell = (cell, positions, numbers)

        structure = spglib_tuple_to_structure(spgcell)
        structure.set_pbc(pbc)
        print('----------')
        print(structure.pbc)
        print(structure.cell)
        print(structure.sites)
        print('----------')
        kp = get_kpoints_path(structure, method='seekpath')
        print(kp.keys())
        print(kp['parameters']['point_coords'])
        print(kp['parameters']['path'])
