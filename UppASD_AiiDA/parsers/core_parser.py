'''
Parser for UppASD
'''
from aiida.engine import ExitCode
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory
from aiida.orm import SinglefileData, ArrayData, Dict
import numpy as np
import pandas as pd


def total_energy_file_paser(file_name_of_total_energy):
    # here the inputfile name should be totenergy.SCsurf_T.out
    result = pd.read_csv(file_name_of_total_energy,
                         sep='\s+', header=None).drop([0])
    Iter_num_totenergy = np.array(list(result[0]))
    Tot = np.array(list(result[1]))
    Exc = np.array(list(result[2]))
    Ani = np.array(list(result[3]))
    DM = np.array(list(result[4]))
    PD = np.array(list(result[5]))
    BiqDM = np.array(list(result[6]))
    BQ = np.array(list(result[7]))
    Dip = np.array(list(result[8]))
    Zeeman = np.array(list(result[9]))
    LSF = np.array(list(result[10]))
    Chir = np.array(list(result[11]))
    return Iter_num_totenergy, Tot, Exc, Ani, DM, PD, BiqDM, BQ, Dip, Zeeman, LSF, Chir


def coord_file_paser(file_name_of_coord):
    # this matrix includes series number: the first C
    result = pd.read_csv(file_name_of_coord, sep='\s+', header=None)
    coord = np.array(result)
    return coord


def qpoints_file_paser(file_name_of_qpoints):
    result = pd.read_csv(file_name_of_qpoints, sep='\s+', header=None)
    qpoints = np.array(result)
    return qpoints


def qm_sweep_file_paser(file_name_of_qm_sweep):
    # the header is not suitable so we delete it
    result = pd.read_csv(file_name_of_qm_sweep, sep='\s+',
                         header=None, skiprows=1)
    Q_vector = np.array(result)[:, 1:4]
    Energy_mRy = np.array(result)[:, 4]
    return Q_vector, Energy_mRy


def qm_minima_file_paser(file_name_of_qm_minima):
    result = pd.read_csv(file_name_of_qm_minima,
                         sep='\s+', header=None, skiprows=1)
    Q_vector = np.array(result)[:, 1:4]
    Energy_mRy = np.array(result)[:, 4]
    return Q_vector, Energy_mRy


def averages_file_paser(file_name_of_averages):
    result = pd.read_csv(file_name_of_averages,
                         sep='\s+', header=None).drop([0])
    Iter_num_average = np.array(list(np.array(result)[:, 0]))
    M_x = np.array(list(np.array(result)[:, 1]))
    M_y = np.array(list(np.array(result)[:, 2]))
    M_z = np.array(list(np.array(result)[:, 3]))
    M = np.array(list(np.array(result)[:, 4]))
    M_stdv = np.array(list(np.array(result)[:, 5]))
    return Iter_num_average,M_x, M_y, M_z, M, M_stdv


def moment_file_paser(mom_out_file):
    mom_output = pd.read_csv(mom_out_file,sep='\s+', header=None,skiprows=7)
    mom_x = mom_output[4]
    mom_y = mom_output[5]
    mom_z = mom_output[6]
    mom_states_x =np.array(mom_x)
    mom_states_y =np.array(mom_y)
    mom_states_z =np.array(mom_z)
    return mom_states_x,mom_states_y,mom_states_z
 



'''
# add new paser if needed
def _file_paser(file_name_of_):
    result = pd.read_csv("",sep ='\s+',header=None)
     = np.array(result)
    return
'''


class SpinDynamic_core_parser(Parser):
    """
    Parser class for parsing output of calculation.
    """

    # def __init__(self, node):
    #     """
    #     """

    def parse(self, **kwargs):
        """
        In this version of API we parse :
        totenergy.SCsurf_T.out coord.SCsurf_T.out  qpoints.out  averages.SCsurf_T.out  qm_sweep.SCsurf_T.out  qm_minima.SCsurf_T.out

        """
       #results = ArrayData()
        output_folder = self.retrieved

        retrived_file_name_list = output_folder.list_object_names()

        for name in retrived_file_name_list:
            if 'coord' in name:
                coord_filename = name
            if 'qpoints' in name:
                qpoints_filename = name
            if 'averages' in name:
                averages_filename = name
            if 'qm_sweep' in name:
                qm_sweep_filename = name
            if 'qm_minima' in name:
                qm_minima_filename = name
            if 'totenergy' in name:
                totenergy_filename = name

            
            if 'moment' in name:
                moment_filename = name
            

        # parse totenergy.xx.out
        self.logger.info("Parsing '{}'".format(totenergy_filename))
        with output_folder.open(totenergy_filename, 'rb') as f:
            Iter_num_totenergy, Tot, Exc, Ani, DM, PD, BiqDM, BQ, Dip, Zeeman, LSF, Chir = total_energy_file_paser(
                f)
            output_totenergy = ArrayData()
            output_totenergy.set_array('Iter_num_totenergy', Iter_num_totenergy)
            output_totenergy.set_array('Tot', Tot)
            output_totenergy.set_array('Exc', Exc)
            output_totenergy.set_array('Ani', Ani)
            output_totenergy.set_array('DM', DM)
            output_totenergy.set_array('PD', PD)
            output_totenergy.set_array('BiqDM', BiqDM)
            output_totenergy.set_array('BQ', BQ)
            output_totenergy.set_array('Dip', Dip)
            output_totenergy.set_array('Zeeman', Zeeman)
            output_totenergy.set_array('LSF', LSF)
            output_totenergy.set_array('Chir', Chir)
        # it is not good to hold a particular output datatype
        self.out('totenergy', output_totenergy)


        # parse coord.xx.out
        self.logger.info("Parsing '{}'".format(coord_filename))
        with output_folder.open(coord_filename, 'rb') as f:
            coord = coord_file_paser(f)
            output_coord = ArrayData()
            output_coord.set_array('coord', coord)
        self.out('coord', output_coord)

        # parse qpoints.xx.out
        self.logger.info("Parsing '{}'".format(qpoints_filename))
        with output_folder.open(qpoints_filename, 'rb') as f:
            qpoints = qpoints_file_paser(f)
            output_qpoints = ArrayData()
            output_qpoints.set_array('qpoints', qpoints)
        self.out('qpoints', output_qpoints)

        # parse averages.xx.out
        self.logger.info("Parsing '{}'".format(averages_filename))
        with output_folder.open(averages_filename, 'rb') as f:
            Iter_num_average, M_x, M_y, M_z, M, M_stdv = averages_file_paser(f)
            output_averages = ArrayData()
            output_averages.set_array('Iter_num_average', Iter_num_average)
            output_averages.set_array('M_x', M_x)
            output_averages.set_array('M_y', M_y)
            output_averages.set_array('M_z', M_z)
            output_averages.set_array('M', M)
            output_averages.set_array('M_stdv', M_stdv)
        self.out('averages', output_averages)

        # parse qm_sweep.xx.out
        self.logger.info("Parsing '{}'".format(qm_sweep_filename))
        with output_folder.open(qm_sweep_filename, 'rb') as f:
            Q_vector, Energy_mRy = qm_sweep_file_paser(f)
            output_qm_sweep = ArrayData()
            output_qm_sweep.set_array('Q_vector', Q_vector)
            output_qm_sweep.set_array('Energy_mRy', Energy_mRy)
        self.out('qm_sweep', output_qm_sweep)

        # parse qm_minima.xx.out
        self.logger.info("Parsing '{}'".format(qm_minima_filename))
        with output_folder.open(qm_minima_filename, 'rb') as f:
            Q_vector, Energy_mRy = qm_minima_file_paser(f)
            output_qm_minima = ArrayData()
            output_qm_minima.set_array('Q_vector', Q_vector)
            output_qm_minima.set_array('Energy_mRy', Energy_mRy)
        self.out('qm_minima', output_qm_minima)


    
        # parse moment.xx.out
        self.logger.info("Parsing '{}'".format(moment_filename))
        with output_folder.open(moment_filename, 'rb') as f:
            mom_states_x,mom_states_y,mom_states_z = moment_file_paser(f)
            output_mom_states = ArrayData()
            output_mom_states.set_array('mom_states_x', mom_states_x)
            output_mom_states.set_array('mom_states_y', mom_states_y)
            output_mom_states.set_array('mom_states_z', mom_states_z)
        self.out('mom_states_traj', output_mom_states)
        

        return ExitCode(0)
        # here we united all output array into a Dict
