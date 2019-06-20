from __future__ import absolute_import

from aiida.engine import ExitCode
from aiida.orm import Dict, XyData, ArrayData
from aiida.common import NotExistent
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

ASDCalculation = CalculationFactory('uppasd')

class ASDParser(Parser):

    def __init__(self, node):
        """Initialize ASDParser instance 
        """
        from aiida.common import exceptions
        super(ASDParser, self).__init__(node)
        if not issubclass(node.process_class, ASDCalculation):
            raise exceptions.ParsingError("Can only parse ASDCalculation")

    def parse_inpsd(self, **kwargs):
        import re
        import numpy as np

        input_file = open('inpsd.dat').read()

        prepend_str=['simid', 'do_prnstruct', 'Sym', 'Mensemble', 'Initmag',
                     'ip_mpde', 'ip_nphase', 'ip_mcanneal', 'mode', 'Nstep',
                     'do_avrg', 'avrg_step', 'avrg_buff', 'do_tottraj',
                     'tottraj_step', 'tottraj_buff', 'do_ams', 'plotenergy',
                     'do_sc', 'sc_nstep', 'sc_step', 'qpoints']

        parameters = dict()

        for keyword in prepend_str:
            _regex = r"(?>=)"+keyword+r"\s*\w+"

            _comp = re.compile(_regex, re.MULTILINE)
            _val = _comp.findall(input_file)
            parameters[keyword] = _val.replace(" ", "")

        return

    def parse_averages(self, **kwargs):
        """Parser for the average magnetization file
        """
        import pandas as pd
        # Check if the file is present
        try:
            self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        averages_filename = self.node.get_option('averages_filename')

        files_retrieved = self.retrieved.list_object_names()
        files_expected = [averages_filename]

        # Check if the averages file is in the list of retrieved files
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error("Found files '{}', expected to find '{}'".format(
                files_retrieved, files_expected))
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # add the averages file to the node
        self.logger.info("Parsing '{}'".format(averages_filename))
        try:
            m_ave = pd.read_csv(averages_filename, delim_whitespace=True)
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE

        _t_units = "iterations"

        if m_ave.columns.values[0] == "Time[s]":
            _t_units = "seconds"

        _m_units = r"\mu_B"*(len(m_ave.columns) - 1)

        ave_data = XyData()
        ave_data.set_x(m_ave.iloc[:,0], m_ave.columns.values[0], _t_units)
        ave_data.set_y(m_ave.iloc[:,1:], m_ave.columns.values[1:], _m_units)

        m_ave_dict = dict()

        for col_name in m_ave.columns.values[1:]:
            m_ave_dict[col_name] = m_ave[col_name].values[-1]

        self.out("final_averages", Dict(dict=m_ave_dict))
        self.out("averages", ave_data)
        return ExitCode(0)

    def parse_averages_proj(self, **kwargs):
        """Parse the projected type magnetic averages file
        """
        import pandas as pd
        # Check if the file is present
        try:
            self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        ave_proj_type_filename = self.node.get_option('ave_proj_type_filename')

        files_retrieved = self.retrieved.list_object_names()

        files_expected = [ave_proj_type_filename]

        if not set(files_expected) <= set(files_retrieved):
            self.logger.error("Found files '{}', expected to find '{}'".format(
                files_retrieved, files_expected))
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        self.logger.info("Parsing '{}'".format(ave_proj_type_filename))

        try:
            ave_proj_type = pd.read_csv(ave_proj_type_filename,
                                        delim_whitespace=True)
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE

        ave_proj_type_dict = dict()

        num_types = ave_proj_type['Proj'].max()

        for col_name in ave_proj_type.columns.values[1:]:
            ave_proj_type_dict[col_name] = dict()
            for ii in range(1, num_types + 1):
                ave_proj_type_dict[col_name][ii] =\
                    ave_proj_type[col_name].values[-ii]

        _t_units = "iterations"

        if ave_proj_type.columns.values[0] == "Time[s]":
            _t_units = "seconds"

        _mom_units = r"\mu_B"*(len(ave_proj_type.columns) - 2)

        ave_proj_type_data = [None]*num_types

        for ii in range(1, num_types + 1):
            ave_proj_type_data[ii - 1] = XyData()
            ave_proj_type_data[ii - 1].\
                set_x(ave_proj_type.iloc[:,0], ave_proj_type.columns.values[0],
                      _t_units)
            ave_proj_type_data[ii - 1].\
                set_y(ave_proj_type.iloc[:,2:][ave_proj_type.iloc[1] == ii],
                      ave_proj_type.columns.values[2:], _mom_units)

            self.out("projected averages " + str(ii),
                     ave_proj_type_data[ii - 1])

        self.out("final projected averages", Dict(dict=ave_proj_type_dict))
        return ExitCode(0)

    def parse_averages_chem(self, **kwargs):
        """Parse the projected chemical magnetic averages file
        """
        import pandas as pd
        # Check if the file is present
        try:
            self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        ave_proj_chem_filename = self.node.get_option('ave_proj_chem_filename')

        files_retrieved = self.retrieved.list_object_names()

        files_expected = [ave_proj_chem_filename]

        if not set(files_expected) <= set(files_retrieved):
            self.logger.error("Found files '{}', expected to find '{}'".format(
                files_retrieved, files_expected))
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        self.logger.info("Parsing '{}'".format(ave_proj_chem_filename))

        try:
            ave_proj_chem = pd.read_csv(ave_proj_chem_filename,
                                        delim_whitespace=True)
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE

        ave_proj_chem_dict = dict()

        num_chem = ave_proj_chem['Proj'].max()

        for col_name in ave_proj_chem.columns.values[1:]:
            ave_proj_chem_dict[col_name] = dict()
            for ii in range(1, num_chem + 1):
                ave_proj_chem_dict[col_name][ii] =\
                    ave_proj_chem[col_name].values[-ii]

        _t_units = "iterations"

        if ave_proj_chem.columns.values[0] == "Time[s]":
            _t_units = "seconds"

        _mom_units = r"\mu_B"*(len(ave_proj_chem.columns) - 2)

        ave_proj_chem_data = [None]*num_chem

        for ii in range(1, num_chem + 1):
            ave_proj_chem_data[ii - 1] = XyData()
            ave_proj_chem_data[ii - 1].\
                set_x(ave_proj_chem.iloc[:,0], ave_proj_chem.columns.values[0],
                      _t_units)
            ave_proj_chem_data[ii - 1].\
                set_y(ave_proj_chem.iloc[:,2:][ave_proj_chem.iloc[1] == ii],
                      ave_proj_chem.columns.values[2:], _mom_units)

            self.out("projected chemical averages " + str(ii),
                     ave_proj_chem_data[ii - 1])

        self.out("final projected chemicall averages",
                 Dict(dict=ave_proj_chem_dict))
        return ExitCode(0)

    def parse_energies(self, **kwargs):
        """Parser for the total energy file
        """
        import pandas as pd
        # Check if the file is present
        try:
            self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        energies_filename = self.node.get_option('totenergy_filename')

        files_retrieved = self.retrieved.list_object_names()

        files_expected = [energies_filename]

        if not set(files_expected) <= set(files_retrieved):
            self.logger.error("Found files '{}', expected to find '{}'".format(
                files_retrieved, files_expected))
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        self.logger.info("Parsing '{}'".format(energies_filename))

        try:
            ene = pd.read_csv(energies_filename, delim_whitespace=True)
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE
        
        _t_units = "iterations"

        if ene.columns.values[0] == "Time":
            _t_units = "seconds"

        _ene_units = "mRy"*(len(ene.columns) - 1)

        ene_data = XyData()
        ene_data.set_x(ene.iloc[:,0], ene.columns.values[0], _t_units)
        ene_data.set_y(ene.iloc[:,1:], ene.columns.values[1:], _ene_units)

        ene_dict = dict()

        for col_name in ene.columns.values[1:]:
            ene_dict[col_name] = ene[col_name].values[-1]

        self.out("final_energies", Dict(dict=ene_dict))
        self.out("energies", ene_data)
        return ExitCode(0)

    def parse_cummulants(self, **kwargs):
        """Parser for the cummulants file
        """
        import pandas as pd
        # Check if the file is present
        try:
            self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        cummu_filename = self.node.get_option('cummu_filename')

        files_retrieved = self.retrieved.list_object_names()

        files_expected = [cummu_filename]

        if not set(files_expected) <= set(files_retrieved):
            self.logger.error("Found files '{}', expected to find '{}'".format(
                files_retrieved, files_expected))
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        self.logger.info("Parsing '{}'".format(cummu_filename))

        try:
            cummu = pd.read_csv(cummu_filename, delim_whitespace=True)
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE

        _t_units = "iterations"

        if cummu.columns.values[0] == "Time":
            _t_units = "seconds"

        _cummu_units = [r"\mu_B", r"\mu_B^2", r"\mu_B^4", " ", " ", " ", "mRy",
                        "mRy", "mRy"]

        cummu_data = XyData()
        cummu_data.set_x(cummu.iloc[:,0], cummu.columns.values[0], _t_units)
        cummu_data.set_y(cummu.iloc[:,1:], cummu.columns.values[1:],
                         _cummu_units)

        cummu_dict = dict()

        for col_name in cummu.columns.values[1:]:
            cummu_dict[col_name] = cummu[col_name].values[-1]

        self.out("cummulants_final", Dict(dict=cummu_dict))
        self.out("cummulants", cummu_data)
        return ExitCode(0)

    def parse_sqw(self, **kwargs):
        """Parse the spin-spin correlation function file
        """
        import numpy as np
        import pandas as pd
        from scipy import signal
        # Check if the file is present
        try:
            self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        sqw_filename = self.node.get_option('sqw_filename')

        files_retrieved = self.retrieved.list_object_names()

        files_expected = [sqw_filename]

        if not set(files_expected) <= set(files_retrieved):
            self.logger.error("Found files '{}', expected to find '{}'".format(
                files_retrieved, files_expected))
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        self.logger.info("Parsing '{}'".format(sqw_filename))

        try:
            sqwa = pd.read_csv(sqw_filename, delim_whitespace=True,
                               header=None).values
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE
        
        qd = int(sqwa[sqwa.shape[0] - 1, 0])
        ed = int(sqwa[sqwa.shape[0] - 1, 4])
        sqw_data = ArrayData()
        sigma = 1.50
        gauss = signal.gaussian(ed, std=sigma)
        sqw_labels =\
            [r'$S_x(q,\omega)$ [meV]', r'$S_y(q,\omega)$ [meV]',
             r'$S_z(q,\omega)$ [meV]', r'$S^2(q,\omega)$ [meV]']
        # Perform a convolution with a windowing function for each q-point
        for iq in range(0, qd):
            indx = np.where(sqwa[:, 0] == (iq + 1))[0]
            for ii in range(0, 4):
                sqwa[indx, ii + 5] =\
                    signal.convolve(sqwa[indx, ii + 5], gauss, mode='same')
        # Find the peaks and normalize the data
        for ii in range(5, len(sqwa[0])):
            sqw = np.transpose((np.reshape(sqwa[:, ii], (qd, ed))
                                [:, 0: int(ed/2)]))
            normMat = np.diag(1.0/np.amax(sqw, axis=0))
            sqw = np.matmul(sqw, normMat)
            sqw_data.set_array(sqw, sqw_labels[ii])

        self.out("sqw",sqw_data)
        return ExitCode(0)