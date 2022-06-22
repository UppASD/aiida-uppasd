"""
Set of helper functions for parsing the UppASD generated files.
"""
import typing
import numpy as np
import pandas as pd

def parser_array_file(
    handler,
    skiprows: int = 0,
    usecols: typing.Union[list, None] = None,
) -> np.ndarray:
    """
    Generic file parser to handler csv files containing arrays

    :param handler: handler of the file that is going to be parsed
    :type handler: filelike object
    :param skiprows: number of rows to be skipped, defaults to 0
    :type skiprows: int, optional
    :param usecols: columns that will be used when parsing, defaults to None
    :type usecols: typing.Union[list, None], optional
    :return: array with the parsed data
    :rtype: np.ndarray
    """

    result = pd.read_csv(
        handler,
        sep=r'\s+',
        header=None,
        skiprows=skiprows,
        usecols=usecols,
    )
    return result.values

def parse_inpsd(handler) -> dict:
    """
    Parse the `inpsd.dat` input file for UppASD.

    :param handler: handler for the `inpsd.dat` file
    :type handler: filelike object
    :return: dictionary with the parsed `inpsd.dat` data
    :rtype: dict
    """    
    inpsd_data = {}

    lines = handler.readlines()
    for idx, line in enumerate(lines):
        line_data = line.rstrip('\n').split()
        if len(line_data) > 0:
            # Find the simulation id
            if line_data[0] == 'simid':
                inpsd_data['simid'] = line_data[1]

            # Find the cell data
            if line_data[0] == 'cell':
                cell = []
                lattice = np.empty([0, 3])
                line_data = lines[idx + 0].split()
                cell = np.append(cell, np.asarray(line_data[1:4]))
                lattice = np.vstack((lattice, np.asarray(line_data[1:4])))
                line_data = lines[idx + 1].split()
                cell = np.append(cell, np.asarray(line_data[0:3]))
                lattice = np.vstack((lattice, np.asarray(line_data[0:3])))
                line_data = lines[idx + 2].split()
                cell = np.append(cell, np.asarray(line_data[0:3]))
                lattice = np.vstack((lattice, np.asarray(line_data[0:3])))

                inpsd_data['cell'] = cell
                inpsd_data['lattice'] = lattice


            # Find the size of the simulated cell
            if line_data[0] == 'ncell':
                ncell_x = int(line_data[1])
                ncell_y = int(line_data[1])
                ncell_z = int(line_data[1])
                mesh = [ncell_x, ncell_y, ncell_z]

                inpsd_data['mesh'] = mesh

            if line_data[0] == 'timestep':
                inpsd_data['timestep'] = line_data[1]

            if line_data[0] == 'sc_nstep':
                inpsd_data['sc_nstep'] = line_data[1]

            if line_data[0] == 'sc_step':
                inpsd_data['sc_step'] = line_data[1]

    return inpsd_data

def parse_posfile(handler) -> typing.Union[np.ndarray, np.ndarray]:
    """
    Parse the position file for UppASD.

    :param handler: handler for the `posfile`
    :type handler: filelike object
    :return: the positions and atom numbers of the unit cell
    :rtype: typing.Union[np.ndarray, np.ndarray]
    """    
    # Read the name of the position file
    lines = handler.readlines()
    positions = np.empty([0, 3])
    numbers = []
    for line in lines:
        line_data = line.rstrip('\n').split()
        if len(line_data) > 0:
            positions = np.vstack((positions, np.asarray(line_data[2:5])))
            numbers = np.append(numbers, np.asarray(line_data[1]))
    return positions, numbers
