# -*- coding: utf-8 -*-
'''
Parser for UppASD
'''
import json
import numpy as np
import seekpath as spth
from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory
from aiida.common.exceptions import NotExistent
from aiida_uppasd.parsers.raw_parsers import parse_inpsd, parse_posfile, parser_array_file

ASDCalculation = CalculationFactory('UppASD_core_calculations')

# from UppASD repo postQ.py @Anders


class UppASDBaseParser(Parser):
    """
    Note that, all the parser here are just demos,or in other word,
    basic version(np.array) You can change it as you want and parser
    it to fit your need.

    :param Parser father class aiida.parsers.parser
    :type Parser aiida.parsers.parser module
    :return: parsed numpy array
    """

    def parse(self, **kwargs):  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        """
        IMPORTANT, All parser should be connect to the output port in core_calcs,
        since all files have been retrieved into the retrieved folder,
        we have no need to care about how many output file will UppASD generate,
        only thing that need to care is when parsing them into the data type you need,
        all parser should be connect to the output port in core_calcs
        """

        #results = ArrayData()
        try:
            out_folder = self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        list_of_files = out_folder.list_object_names()

        #parser _scheduler-stdout.txt to detect if Simulation is finished or not

        with out_folder.open('_scheduler-stdout.txt', 'rb') as handler:
            log = str(handler.read())
            if 'Simulation finished' in log:
                pass
            else:
                return ASDCalculation.exit_codes.WallTimeError

        for filename in list_of_files:
            print(f'File name in output list: {filename}')
            self.logger.info(f'File name in output list: {filename}')

            if 'coord' in filename:
                # parse coord.xx.out
                self.logger.info(f"Parsing '{filename}'")
                with out_folder.open(filename, 'rb') as handler:
                    data = parser_array_file(handler=handler)
                output = orm.ArrayData()
                output.set_array('coord', data)
                self.out('coord', output)

            if 'qpoints' in filename and 'qpointsdir' not in filename:
                self.logger.info(f"Parsing '{filename}'")
                with out_folder.open(filename, 'rb') as handler:
                    data = parser_array_file(handler=handler)
                output = orm.ArrayData()
                output.set_array('qpoints', data)
                self.out('qpoints', output)

            if 'averages' in filename:
                self.logger.info(f"Parsing '{filename}'")
                with out_folder.open(filename, 'rb') as handler:
                    data = parser_array_file(
                        handler=handler,
                        skiprows=1,
                    )

                output = orm.ArrayData()
                entries = [
                    'iterations',
                    'magnetization_x',
                    'magnetization_y',
                    'magnetization_z',
                    'magnetization_mod',
                    'magnetization_stdev',
                ]
                for index, entry in enumerate(entries):
                    output.set_array(entry, data[:, index])
                self.out('averages', output)

            if 'qm_sweep' in filename:
                self.logger.info(f"Parsing '{filename}'")
                with out_folder.open(filename, 'rb') as handler:
                    data = parser_array_file(handler=handler, skiprows=1)
                output = orm.ArrayData()
                output.set_array('q_vector', data[:, 1:4])
                output.set_array('energy_mry', data[:, 4])
                self.out('qm_sweep', output)

            if 'qm_minima' in filename:
                self.logger.info(f"Parsing '{filename}'")
                with out_folder.open(filename, 'rb') as handler:
                    data = parser_array_file(handler=handler, skiprows=1)
                output = orm.ArrayData()
                output.set_array('q_vector', data[:, 1:4])
                output.set_array('energy_mry', data[:, 4])
                self.out('qm_minima', output)

            if 'totenergy' in filename:
                self.logger.info(f"Parsing '{filename}'")
                with out_folder.open(filename, 'rb') as handler:
                    data = parser_array_file(handler=handler, skiprows=1)

                entries = [
                    'iterations',
                    'energy_total',
                    'energy_exchange',
                    'energy_anisotropy',
                    'energy_dzyaloshinskii_moriya',
                    'energy_pseudo_dipole',
                    'energy_biquadratic_dm',
                    'energy_biquadratic',
                    'energy_dipolar',
                    'energy_zeeman',
                    'energy_lsf',
                    'energy_chiral',
                ]

                output = orm.ArrayData()
                for index, entry in enumerate(entries):
                    output.set_array(entry, data[:, index])

                # it is not good to hold a particular output datatype
                self.out('totenergy', output)

            if 'moment' in filename:
                self.logger.info(f"Parsing '{filename}'")
                with out_folder.open(filename, 'rb') as handler:
                    data = parser_array_file(
                        handler=handler,
                        skiprows=7,
                        usecols=[0, 1, 4, 5, 6],
                    )

                entries = [
                    'iteration',
                    'ensemble',
                    'moments_x',
                    'moments_y',
                    'moments_z',
                ]

                output = orm.ArrayData()
                for index, entry in enumerate(entries):
                    output.set_array(entry, data[:, index])
                self.out('trajectories_moments', output)

            if 'dmdata' in filename:
                self.logger.info(f"Parsing '{filename}'")
                with out_folder.open(filename, 'rb') as handler:

                    data = parser_array_file(
                        handler=handler,
                        skiprows=5,
                        usecols=[0, 1, 4, 5, 6, 7, 8, 9],
                    )

                entries = [
                    'atom_i',
                    'atom_j',
                    'rij_x',
                    'rij_y',
                    'rij_z',
                    'dm_ij_x',
                    'dm_ij_y',
                    'dm_ij_z',
                ]

                output = orm.ArrayData()
                for index, entry in enumerate(entries):
                    output.set_array(entry, data[:, index])

                self.out('dmdata_out', output)

            if 'struct' in filename:
                self.logger.info(f"Parsing '{filename}'")
                with out_folder.open(filename, 'rb') as handler:

                    data = parser_array_file(
                        handler=handler,
                        skiprows=5,
                        usecols=[0, 1, 4, 5, 6, 7],
                    )

                entries = [
                    'atom_i',
                    'atom_j',
                    'rij_x',
                    'rij_y',
                    'rij_z',
                    'jij',
                ]

                output = orm.ArrayData()
                for index, entry in enumerate(entries):
                    output.set_array(entry, data[:, index])

                self.out('struct_out', output)

            if 'cumulant' in filename and not 'out' in filename:
                self.logger.info(f"Parsing '{filename}'")
                with out_folder.open(filename, 'rb') as handler:
                    cumulants = json.load(handler)
                self.out('cumulants', orm.Dict(dict=cumulants))

            if 'ams' in filename[0:4] and 'qpoints.out' in list_of_files:
                self.logger.info(f"Parsing '{filename}'")
                # Read qpoints to create BandsData object.
                output = orm.BandsData()
                with out_folder.open('qpoints.out', 'rb') as handler:
                    data = parser_array_file(handler=handler)
                output.set_kpoints(data[:, 1:4])

                with out_folder.open(filename, 'rb') as handler:
                    data = parser_array_file(handler=handler)

                output.set_bands(data[:, 1:-1], units='meV')
                self.out('ams', output)

        #section AMS plot
        #based on codes from UppASD repo postQ.py
        #AMSplot is a Bool  like Bool('True')

        ams_plot_settings = False
        if 'AMSplot' in self.node.inputs:
            ams_plot_settings = self.node.inputs.AMSplot.value

        if ams_plot_settings:
            with out_folder.open('inpsd.dat', 'r') as handler:
                inpsd_data = parse_inpsd(handler=handler)
            with out_folder.open('posfile', 'r') as handler:
                positions, numbers = parse_posfile(handler=handler)
            cell = (inpsd_data['lattice'], positions, numbers)
            kpath_obj = spth.get_path(cell)
            for filename in list_of_files:
                if 'ams' in filename[0:4]:
                    with out_folder.open(filename, 'rb') as handler:
                        ams = parser_array_file(handler=handler)
                        ams_dist_col = ams.shape[1] - 1
                if 'sqw' in filename[0:4]:
                    with out_folder.open(filename, 'rb') as handler:
                        sqw = parser_array_file(
                            handler=handler,
                            usecols=[0, 4, 5, 6, 7, 8],
                        )
                        number_qpoints = int(sqw[-1, 0])
                        number_frequencies = int(sqw[-1, 1])
                        sqw_x = np.reshape(sqw[:, 2], (number_qpoints, number_frequencies))
                        sqw_y = np.reshape(sqw[:, 3], (number_qpoints, number_frequencies))

            with out_folder.open('qfile', 'rb') as handler:
                qpoints = parser_array_file(
                    handler=handler,
                    skiprows=1,
                    usecols=[0, 1, 2],
                )
            axlab = []
            axidx = []
            axidx_abs = []
            for idx, row in enumerate(qpoints):
                for key, value in kpath_obj['point_coords'].items():
                    if (value == row).all():
                        axlab.append(key[0])
                        axidx.append(ams[idx, ams_dist_col])
                        axidx_abs.append(ams[idx, 0])
            ams_plot_data = orm.Dict(
                dict={
                    'timestep': inpsd_data['timestep'],
                    'sc_step': inpsd_data['sc_step'],
                    'sqw_x': sqw_x.tolist(),
                    'sqw_y': sqw_y.tolist(),
                    'ams': ams.tolist(),
                    'axidx_abs': axidx_abs,
                    'ams_dist_col': ams_dist_col,
                    'axlab': axlab
                }
            )
            self.out('AMS_plot_var', ams_plot_data)

        return ExitCode(0)
