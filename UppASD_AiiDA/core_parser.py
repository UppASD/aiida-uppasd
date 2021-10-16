'''
Parser for UppASD
'''
from aiida.engine import ExitCode
from aiida.parsers.parser import Parser
from aiida.orm import ArrayData
import numpy as np
import pandas as pd

class SpinDynamic_core_parser(Parser):
    """    
    | Note that, all the parser here are just demos,or in other word, 
    | basic version(np.array) You can change it as you want and parser
    | it to fit your need.
    | 
    | you could you the example here to write you parser:
    |     
    | #def _file_paser(file_name_of_):
    | #    result = pd.read_csv("",sep ='\s+',header=None)
    | #     = np.array(result)
    | #return



    :param Parser: father class aiida.parsers.parser 
    :type Parser: aiida.parsers.parser module 
    :return: parsed numpy array
    :rtype: numpy array
    """    




    
    def total_energy_file_paser(self,file_name_of_total_energy):
        """        
        | file_name_of_total_energy : file name of total energy
        | 
        | this should get from retrieved file automatically.

        :param file_name_of_total_energy: 
        :type file_name_of_total_energy: opened file
        :return: np array
        :rtype: numpy array
        """        
       
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
    
    
    def coord_file_paser(self,file_name_of_coord):
        """
        | this should get from retrieved file automatically.
        | 
        :param file_name_of_coord: coord file 
        :type file_name_of_coord: opened output file
        :return: np array
        :rtype:  np.array
        """        

        # this matrix includes series number: the first C
        result = pd.read_csv(file_name_of_coord, sep='\s+', header=None)
        coord = np.array(result)
        return coord
    
    
    def qpoints_file_paser(self,file_name_of_qpoints):
        """
        | this should get from retrieved file automatically.
        | 
        :param file_name_of_qpoints: points file
        :type file_name_of_qpoints: opened output file
        :return: np array
        :rtype: np array
        """        
       
        result = pd.read_csv(file_name_of_qpoints, sep='\s+', header=None)
        qpoints = np.array(result)
        return qpoints
    
    
    def qm_sweep_file_paser(self,file_name_of_qm_sweep):
        """
        | this should get from retrieved file automatically.
        | 
        :param file_name_of_qm_sweep: qm sweep file
        :type file_name_of_qm_sweep: opened output file
        :return: np.array
        :rtype: np.array
        """        
    
        # the header is not suitable so we delete it
        result = pd.read_csv(file_name_of_qm_sweep, sep='\s+',
                             header=None, skiprows=1)
        Q_vector = np.array(result)[:, 1:4]
        Energy_mRy = np.array(result)[:, 4]
        return Q_vector, Energy_mRy
    
    
    def qm_minima_file_paser(self,file_name_of_qm_minima):
        """
        | this should get from retrieved file automatically.
        | 
        :param file_name_of_qm_minima: qm minima file
        :type file_name_of_qm_minima: opened output file
        :return: np.array
        :rtype: np.array
        """        
        result = pd.read_csv(file_name_of_qm_minima,
                             sep='\s+', header=None, skiprows=1)
        Q_vector = np.array(result)[:, 1:4]
        Energy_mRy = np.array(result)[:, 4]
        return Q_vector, Energy_mRy
    
    
    def averages_file_paser(self,file_name_of_averages):
        """
        | this should get from retrieved file automatically.
        | 
        :param file_name_of_averages: averages file
        :type file_name_of_averages:opened output file
        :return: np.array
        :rtype:np.array
        """        
        result = pd.read_csv(file_name_of_averages,
                             sep='\s+', header=None).drop([0])
        Iter_num_average = np.array(list(np.array(result)[:, 0]))
        M_x = np.array(list(np.array(result)[:, 1]))
        M_y = np.array(list(np.array(result)[:, 2]))
        M_z = np.array(list(np.array(result)[:, 3]))
        M = np.array(list(np.array(result)[:, 4]))
        M_stdv = np.array(list(np.array(result)[:, 5]))
        return Iter_num_average, M_x, M_y, M_z, M, M_stdv
    
    
    def moment_file_paser(self,mom_out_file):
        """
        | this should get from retrieved file automatically.
        | 

        :param mom_out_file: moment file
        :type mom_out_file: opened output file
        :return: np.array
        :rtype: np.array
        """        
       
        mom_output = pd.read_csv(mom_out_file, sep='\s+', header=None, skiprows=7)
        mom_x = mom_output[4]
        mom_y = mom_output[5]
        mom_z = mom_output[6]
        mom_states_x = np.array(mom_x)
        mom_states_y = np.array(mom_y)
        mom_states_z = np.array(mom_z)
        return mom_states_x, mom_states_y, mom_states_z
    
    
    def dm_out_parser(self,dmdata_out_file):
        """
        | this should get from retrieved file automatically.
        | 
        :param dmdata_out_file: dmdata file
        :type dmdata_out_file: opened output file
        :return: np.array
        :rtype: np.array
        """        
       
        data_full = pd.read_csv(dmdata_out_file, sep='\s+',
                                header=None, skiprows=5)
        atom_site_A = list(data_full[0])
        atom_site_B = list(data_full[1])
        #because dgl nodes is start from 0 so we need to use atom site minius 1 to fit DGL
        atom_site_A[:] = [i-1 for i in atom_site_A]
        atom_site_B[:] = [i-1 for i in atom_site_B]
        atom_site_A = np.array(atom_site_A)
        atom_site_B = np.array(atom_site_B)
        Jij_x = np.array(data_full[4])
        Jij_y = np.array(data_full[5])
        Jij_z = np.array(data_full[6])
        DM_x = np.array(data_full[7])
        DM_y = np.array(data_full[8])
        DM_z = np.array(data_full[9])
        return atom_site_A, atom_site_B, Jij_x, Jij_y, Jij_z, DM_x, DM_y, DM_z

    
    




    def parse(self, **kwargs):
        """
        | !!! IMPORTANT All parser should be connect to the output port in core_calcs.py

        | 
        | **kwargs : all files in the retrieved folder, we have no need to care about how many 
        | output file will UppASD generate, only care those we need and parser them into the 
        | data type you need
        |
        | one example for customizd parser
        | #    def _file_paser(file_name_of_):
        | #        result = pd.read_csv("",sep ='\s+',header=None)
        | #         = np.array(result)
        | #        return
        | #
        | #        Returns
        | -------
        | All parser should be connect to the output port in core_calcs.py
        """        

       #results = ArrayData()
        output_folder = self.retrieved

        retrived_file_name_list = output_folder.list_object_names()

        for name in retrived_file_name_list:
            if 'coord' in name:
                coord_filename = name
                # parse coord.xx.out
                self.logger.info("Parsing '{}'".format(coord_filename))
                with output_folder.open(coord_filename, 'rb') as f:
                    coord = self.coord_file_paser(f)
                    output_coord = ArrayData()
                    output_coord.set_array('coord', coord)
                self.out('coord', output_coord)
            if 'qpoints' in name:
                qpoints_filename = name
                # parse qpoints.xx.out
                self.logger.info("Parsing '{}'".format(qpoints_filename))
                with output_folder.open(qpoints_filename, 'rb') as f:
                    qpoints = self.qpoints_file_paser(f)
                    output_qpoints = ArrayData()
                    output_qpoints.set_array('qpoints', qpoints)
                self.out('qpoints', output_qpoints)
            if 'averages' in name:
                averages_filename = name
                # parse averages.xx.out
                self.logger.info("Parsing '{}'".format(averages_filename))
                with output_folder.open(averages_filename, 'rb') as f:
                    Iter_num_average, M_x, M_y, M_z, M, M_stdv = self.averages_file_paser(
                        f)
                    output_averages = ArrayData()
                    output_averages.set_array(
                        'Iter_num_average', Iter_num_average)
                    output_averages.set_array('M_x', M_x)
                    output_averages.set_array('M_y', M_y)
                    output_averages.set_array('M_z', M_z)
                    output_averages.set_array('M', M)
                    output_averages.set_array('M_stdv', M_stdv)
                self.out('averages', output_averages)
            if 'qm_sweep' in name:
                qm_sweep_filename = name
                # parse qm_sweep.xx.out
                self.logger.info("Parsing '{}'".format(qm_sweep_filename))
                with output_folder.open(qm_sweep_filename, 'rb') as f:
                    Q_vector, Energy_mRy = self.qm_sweep_file_paser(f)
                    output_qm_sweep = ArrayData()
                    output_qm_sweep.set_array('Q_vector', Q_vector)
                    output_qm_sweep.set_array('Energy_mRy', Energy_mRy)
                self.out('qm_sweep', output_qm_sweep)

            if 'qm_minima' in name:
                qm_minima_filename = name
                # parse qm_minima.xx.out
                self.logger.info("Parsing '{}'".format(qm_minima_filename))
                with output_folder.open(qm_minima_filename, 'rb') as f:
                    Q_vector, Energy_mRy = self.qm_minima_file_paser(f)
                    output_qm_minima = ArrayData()
                    output_qm_minima.set_array('Q_vector', Q_vector)
                    output_qm_minima.set_array('Energy_mRy', Energy_mRy)
                self.out('qm_minima', output_qm_minima)

            if 'totenergy' in name:
                totenergy_filename = name
                # parse totenergy.xx.out
                self.logger.info("Parsing '{}'".format(totenergy_filename))
                with output_folder.open(totenergy_filename, 'rb') as f:
                    Iter_num_totenergy, Tot, Exc, Ani, DM, PD, BiqDM, BQ, Dip, Zeeman, LSF, Chir = self.total_energy_file_paser(
                        f)
                    output_totenergy = ArrayData()
                    output_totenergy.set_array(
                        'Iter_num_totenergy', Iter_num_totenergy)
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

            if 'moment' in name:
                moment_filename = name

                # parse moment.xx.out
                self.logger.info("Parsing '{}'".format(moment_filename))
                with output_folder.open(moment_filename, 'rb') as f:
                    mom_states_x, mom_states_y, mom_states_z = self.moment_file_paser(
                        f)
                    output_mom_states = ArrayData()
                    output_mom_states.set_array('mom_states_x', mom_states_x)
                    output_mom_states.set_array('mom_states_y', mom_states_y)
                    output_mom_states.set_array('mom_states_z', mom_states_z)
                self.out('mom_states_traj', output_mom_states)

            if 'dmdata' in name:
                dmdata_out_file = name
                # parse dmdata.xx.out
                self.logger.info("Parsing '{}'".format(dmdata_out_file))
                with output_folder.open(dmdata_out_file, 'rb') as f:
                    atom_site_A, atom_site_B, Jij_x, Jij_y, Jij_z, DM_x, DM_y, DM_z = self.dm_out_parser(
                        f)
                    dmdata_out = ArrayData()
                    dmdata_out.set_array('atom_site_A', atom_site_A)
                    dmdata_out.set_array('atom_site_B', atom_site_B)
                    dmdata_out.set_array('Jij_x', Jij_x)
                    dmdata_out.set_array('Jij_y', Jij_y)
                    dmdata_out.set_array('Jij_z', Jij_z)
                    dmdata_out.set_array('DM_x', DM_x)
                    dmdata_out.set_array('DM_y', DM_y)
                    dmdata_out.set_array('DM_z', DM_z)
                self.out('dmdata_out', dmdata_out)

        return ExitCode(0)
