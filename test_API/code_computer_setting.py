# -*- coding: utf-8 -*-
'''
Taken from AiiDA-diff repo
https://github.com/aiidateam/aiida-diff/blob/master/aiida_diff/helpers.py
and modified by Qichen
'''

import shutil
import tempfile

from aiida.common.exceptions import NotExistent
from aiida.orm import Code, Computer

LOCALHOST_NAME = 'localhost-test'


def get_path_to_executable(executable):
    """ Get path to local executable.
    :param executable: Name of executable in the $PATH variable
    :type executable: str
    :return: path to executable
    :rtype: str
    """
    path = shutil.which(executable)
    if path is None:
        raise ValueError(f"'{executable}' executable not found in PATH.")
    return path


def get_computer(name=LOCALHOST_NAME, workdir=None):
    """Get AiiDA computer.
    Loads computer 'name' from the database, if exists.
    Sets up local computer 'name', if it isn't found in the DB.
    :param name: Name of computer to load or set up.
    :param workdir: path to work directory
        Used only when creating a new computer.
    :return: The computer node
    :rtype: :py:class:`aiida.orm.Computer`
    """

    try:
        computer = Computer.objects.get(label=name)
    except NotExistent:
        if workdir is None:
            workdir = tempfile.mkdtemp()

        computer = Computer(
            label=name,
            description='localhost computer set up by AiiDA_UppASD demo',
            hostname=name,
            workdir=workdir,
            transport_type='local',
            scheduler_type='direct'
        )
        computer.store()
        computer.set_minimum_job_poll_interval(0.)
        computer.configure()
        print(f"New computer '{name}' is created")
    return computer


def get_code(label, executable_path=None, computer=None):
    """Get local code.
    Sets up code for given entry point on given computer.
    :param entry_point: Entry point of calculation plugin
    :param computer: (local) AiiDA computer
    :return: The code node
    :rtype: :py:class:`aiida.orm.Code`
    """

    # try:
    #     executable = executables[entry_point]
    # except KeyError as exc:
    #     raise KeyError(
    #         "Entry point '{}' not recognized. Allowed values: {}".format(
    #             entry_point, list(executables.keys()))) from exc

    codes = Code.objects.find(filters={'label': label})  # pylint: disable=no-member
    if codes:
        return codes[0]
    path = get_path_to_executable(executable_path)
    code = Code(
        input_plugin_name='uppasd.uppasd_calculation',
        remote_computer_exec=[computer, path],
    )
    code.label = label
    print(f"New code '{label}' is created")
    return code.store()
