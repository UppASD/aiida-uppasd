from __future__ import absolute_import

from aiida.engine import ExitCode
from aiida.orm import Dict, XyData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

ASDCalculation = CalculationFactory('uppasd')

class ASDParser(Parser):

    def __init__(self, node):
        """Initialize ASDParser instance 
        
        Arguments:
            node {[type]} -- [description]
        """
        from aiida.common import exceptions
        super(ASDParser, self).__init__(node)
        if not issubclass(node.process_class, ASDCalculation):
            raise exceptions.ParsingError("Can only parse ASDCalculation")

    def parse_averages(self, **kwargs):
        import pandas as pd
        from aiida.orm import SinglefileData

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
        m_ave = pd.read_csv(averages_filename, delim_whitespace=True)

        _t_units = "iterations"

        if m_ave.columns.values[0] == "Time[s]":
            _t_units = "seconds"

        _m_units = r"\mu_B"*(len(m_ave.columns) - 1)

        ave_data = XyData()
        ave_data.set_x(m_ave.iloc[:,0], m_ave.columns.values[0], _t_units)
        ave_data.set_y(m_ave.iloc[:,1:], m_ave.columns.values[1:], _m_units)

        m_ave_dict = dict()

        for col_name in m_ave.columns.values[1, :]:
            m_ave_dict[col_name] = m_ave[col_name].values[-1]

        self.out("final_averages", Dict(dict=m_ave_dict))
        self.out("averages", ave_data)

        return ExitCode(0)

    def parse_energies(self, **kwargs):
        import pandas as pd
        from aiida.orm import SinglefileData

        energies_filename = self.node.get_option('totenergy_filename')

        files_retrieved = self.retrieved.list_object_names()

        files_expected = [energies_filename]

        if not set(files_expected) <= set(files_retrieved):
            self.logger.error("Found files '{}', expected to find '{}'".format(
                files_retrieved, files_expected))
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        self.logger.info("Parsing '{}'".format(energies_filename))
        ene = pd.read_csv(energies_filename, delim_whitespace=True)
        
        _t_units = "iterations"

        if ene.columns.values[0] == "Time":
            _t_units = "seconds"

        _ene_units = "mRy"*(len(ene.columns) - 1)

        ene_data = XyData()
        ene_data.set_x(ene.iloc[:,0], ene.columns.values[0], _t_units)
        ene_data.set_y(ene.iloc[:,1:], ene.columns.values[1:], _ene_units)

        ene_dict = dict()

        for col_name in ene.columns.values[1, :]:
            ene_dict[col_name] = ene[col_name].values[-1]

        self.out("final_energies", Dict(dict=ene_dict))
        self.out("energies", ene_data)
        return ExitCode(0)

    def parse_cummulants(self, **kwargs):
        import pandas as pd
        from aiida.orm import SinglefileData

        cummu_filename = self.node.get_option('cummu_filename')

        files_retrieved = self.retrieved.list_object_names()

        files_expected = [cummu_filename]

        if not set(files_expected) <= set(files_retrieved):
            self.logger.error("Found files '{}', expected to find '{}'".format(
                files_retrieved, files_expected))
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        self.logger.info("Parsing '{}'".format(cummu_filename))
        cummu = pd.read_csv(cummu_filename, delim_whitespace=True)

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

        for col_name in cummu.columns.values[1, :]:
            cummu_dict[col_name] = cummu[col_name].values[-1]

        self.out("cummulants_final", Dict(dict=cummu_dict))
        self.out("cummulants", cummu_data)
        return ExitCode(0)