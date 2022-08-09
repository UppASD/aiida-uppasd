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


class SpinDynamic_core_parser(Parser):
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
        :param file_name_of_total_energy 
        :type file_name_of_total_energy opened file
        :return: np array
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
    
    
    def coord_file_parser(self,file_name_of_coord):
        """
        :param file_name_of_coord coord file 
        :type file_name_of_coord opened output file
        :return: np array
        """        

        # this matrix includes series number: the first C
        result = pd.read_csv(file_name_of_coord, sep='\s+', header=None)
        coord = np.array(result)
        return coord
    
    
    def qpoints_file_parser(self,file_name_of_qpoints):
        """
        :param file_name_of_qpoints points file
        :type file_name_of_qpoints opened output file
        :return: np array
        """        
       
        result = pd.read_csv(file_name_of_qpoints, sep='\s+', header=None)
        qpoints = np.array(result)
        return qpoints
    
    def ams_file_parser(self,file_name_of_ams):
        """
        :param file_name_of_ams points file
        :type file_name_of_ams opened output file
        :return: np array
        """

        result = pd.read_csv(file_name_of_ams, sep='\s+', header=None)
        ams = np.array(result)

        return ams
    
    def qm_sweep_file_parser(self,file_name_of_qm_sweep):
        """
        :param file_name_of_qm_sweep qm sweep file
        :type file_name_of_qm_sweep opened output file
        :return: np.array
        """        
    
        # the header is not suitable so we delete it
        result = pd.read_csv(file_name_of_qm_sweep, sep='\s+',
                             header=None, skiprows=1)
        Q_vector = np.array(result)[:, 1:4]
        Energy_mRy = np.array(result)[:, 4]
        return Q_vector, Energy_mRy
    
    
    def qm_minima_file_parser(self,file_name_of_qm_minima):
        """
        :param file_name_of_qm_minima qm minima file
        :type file_name_of_qm_minima opened output file
        :return: np.array
        """        
        result = pd.read_csv(file_name_of_qm_minima,
                             sep='\s+', header=None, skiprows=1)
        Q_vector = np.array(result)[:, 1:4]
        Energy_mRy = np.array(result)[:, 4]
        return Q_vector, Energy_mRy
    
    
    def averages_file_parser(self,file_name_of_averages):
        """
        :param file_name_of_averages averages file 
        :type file_name_of_averages opened output file
        :return: np.array
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
    
    
    def moment_file_parser(self,mom_out_file):
        """
        :param mom_out_file moment file
        :type mom_out_file opened output file
        :return: np.array
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
        :param dmdata_out_file dmdata file
        :type dmdata_out_file opened output file
        :return: np.array
        """        
       
        data_full = pd.read_csv(dmdata_out_file, sep='\s+',
                                header=None, skiprows=5)
        atom_site_A = list(data_full[0])
        atom_site_B = list(data_full[1])
        #because dgl nodes is start from 0 so we need to use atom site minius 1 to fit DGL
        #remove this in github repo
        atom_site_A[:] = [i-1 for i in atom_site_A]
        atom_site_B[:] = [i-1 for i in atom_site_B]

        atom_site_A = np.array(atom_site_A)
        atom_site_B = np.array(atom_site_B)
        rij_x = np.array(data_full[4])
        rij_y = np.array(data_full[5])
        rij_z = np.array(data_full[6])
        DM_x = np.array(data_full[7])
        DM_y = np.array(data_full[8])
        DM_z = np.array(data_full[9])
        return atom_site_A, atom_site_B, rij_x, rij_y, rij_z, DM_x, DM_y, DM_z

    def struct_out_parser(self,struct_out_file):
        """
        :param dmdata_out_file dmdata file
        :type dmdata_out_file opened output file
        :return: np.array
        """        
        #Qichen: here I am not sure if that is needed to keep the code running except some empty outputfile
        #so just one try here for test.
        try:
            data_full = pd.read_csv(struct_out_file, sep='\s+',
                                    header=None, skiprows=5)
            atom_site_A = list(data_full[0])
            atom_site_B = list(data_full[1])
            #because dgl nodes is start from 0 so we need to use atom site minius 1 to fit DGL
            #remove this in github repo
            atom_site_A[:] = [i-1 for i in atom_site_A]
            atom_site_B[:] = [i-1 for i in atom_site_B]

            atom_site_A = np.array(atom_site_A)
            atom_site_B = np.array(atom_site_B)
            rij_x = np.array(data_full[4])
            rij_y = np.array(data_full[5])
            rij_z = np.array(data_full[6])
            J_ij = np.array(data_full[7])
            rij = np.array(data_full[8])
        except:
            atom_site_A, atom_site_B, rij_x, rij_y, rij_z, J_ij, rij =  np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
        
        
        return atom_site_A, atom_site_B, rij_x, rij_y, rij_z, J_ij, rij 
    
    def sk_number_parser(self,sk_number_outfile):
        """
        :param sknumber.xxx.out
        :return: float 
        """       
        data_full = pd.read_csv(sk_number_outfile, sep='\s+',
                                header=None, skiprows=1)
        sk_num = float(data_full[-1:][1])
        sk_avg_num = float(data_full[-1:][2])
        return sk_num,sk_avg_num


    def parse(self, **kwargs):
        """IMPORTANT, All parser should be connect to the output port in core_calcs, 
         since all files have been retrieved into the retrieved folder, 
         we have no need to care about how many output file will UppASD generate, 
         only thing that need to care is when parsering them into the data type you need, 
         all parser should be connect to the output port in core_calcs
        """        

       #results = ArrayData()

        output_folder = self.retrieved

        retrived_file_name_list = output_folder.list_object_names()

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
                self.logger.info("Parsing '{}'".format(coord_filename))
                with output_folder.open(coord_filename, 'rb') as f:
                    coord = self.coord_file_parser(f)
                    output_coord = ArrayData()
                    output_coord.set_array('coord', coord)
                self.out('coord', output_coord)
            if 'qpoints' in name and 'qpointsdir' not in name:
                qpoints_filename = name
                # parse qpoints.xx.out
                self.logger.info("Parsing '{}'".format(qpoints_filename))
                with output_folder.open(qpoints_filename, 'rb') as f:
                    qpoints = self.qpoints_file_parser(f)
                    output_qpoints = ArrayData()
                    output_qpoints.set_array('qpoints', qpoints)
                self.out('qpoints', output_qpoints)
            if 'averages' in name:
                averages_filename = name
                # parse averages.xx.out
                self.logger.info("Parsing '{}'".format(averages_filename))
                with output_folder.open(averages_filename, 'rb') as f:
                    Iter_num_average, M_x, M_y, M_z, M, M_stdv = self.averages_file_parser(
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
                    Q_vector, Energy_mRy = self.qm_sweep_file_parser(f)
                    output_qm_sweep = ArrayData()
                    output_qm_sweep.set_array('Q_vector', Q_vector)
                    output_qm_sweep.set_array('Energy_mRy', Energy_mRy)
                self.out('qm_sweep', output_qm_sweep)

            if 'qm_minima' in name:
                qm_minima_filename = name
                # parse qm_minima.xx.out
                self.logger.info("Parsing '{}'".format(qm_minima_filename))
                with output_folder.open(qm_minima_filename, 'rb') as f:
                    Q_vector, Energy_mRy = self.qm_minima_file_parser(f)
                    output_qm_minima = ArrayData()
                    output_qm_minima.set_array('Q_vector', Q_vector)
                    output_qm_minima.set_array('Energy_mRy', Energy_mRy)
                self.out('qm_minima', output_qm_minima)

                # it is not good to hold a particular output datatype
                self.out('totenergy', output_totenergy)

            if 'moment' in name:
                moment_filename = name

                # parse moment.xx.out
                self.logger.info("Parsing '{}'".format(moment_filename))
                with output_folder.open(moment_filename, 'rb') as f:
                    mom_states_x, mom_states_y, mom_states_z = self.moment_file_parser(
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
                    atom_site_A, atom_site_B, rij_x, rij_y, rij_z, DM_x, DM_y, DM_z = self.dm_out_parser(
                        f)
                    dmdata_out = ArrayData()
                    dmdata_out.set_array('atom_site_A', atom_site_A)
                    dmdata_out.set_array('atom_site_B', atom_site_B)
                    dmdata_out.set_array('rij_x', rij_x)
                    dmdata_out.set_array('rij_y', rij_y)
                    dmdata_out.set_array('rij_z', rij_z)
                    dmdata_out.set_array('DM_x', DM_x)
                    dmdata_out.set_array('DM_y', DM_y)
                    dmdata_out.set_array('DM_z', DM_z)
                self.out('dmdata_out', dmdata_out)


            if 'struct' in name:
                struct_out_file = name 
                self.logger.info("Parsing '{}'".format(struct_out_file))
                with output_folder.open(struct_out_file, 'rb') as f:
                    atom_site_A, atom_site_B, rij_x, rij_y, rij_z, J_ij, rij  = self.struct_out_parser(
                        f)
                    struct_out = ArrayData()
                    struct_out.set_array('atom_site_A', atom_site_A)
                    struct_out.set_array('atom_site_B', atom_site_B)
                    struct_out.set_array('rij_x', rij_x)
                    struct_out.set_array('rij_y', rij_y)
                    struct_out.set_array('rij_z', rij_z)
                    struct_out.set_array('J_ij', J_ij)
                    struct_out.set_array('rij', rij)
                self.out('struct_out', struct_out)

            if 'cumulant' in name and not 'out' in name:
                cumulant_file = name 
                self.logger.info("Parsing '{}'".format(cumulant_file))
                with output_folder.open(cumulant_file,'rb') as f:
                    cumulants=json.load(f)
                self.out('cumulants',Dict(dict=cumulants))


            if 'sknumber' in name:
                sknumber_out_file = name
                # parse dmdata.xx.out
                self.logger.info("Parsing '{}'".format(sknumber_out_file))
                with output_folder.open(sknumber_out_file, 'rb') as f:
                    sk_num,sk_avg_num = self.sk_number_parser(
                        f)
                    sk_number_out = ArrayData()
                    sk_number_out.set_array('sk_number_data', np.array([sk_num,sk_avg_num]))
                self.out('sk_num_out', sk_number_out)

            if 'ams' in name[0:4]:
                ams_filename = name
                # parse qpoints.xx.out
                self.logger.info("Parsing '{}'".format(ams_filename))
                # Read qpoints to create BandsData object. TODO: exception handling
                try:
                    with output_folder.open('qpoints.out','rb') as qf:
                        qpoints=self.qpoints_file_parser(qf)

                    with output_folder.open(ams_filename, 'rb') as f:
                        ams = self.ams_file_parser(f)

                    output_ams = BandsData()
                    output_ams.set_kpoints(qpoints[:,1:4])
                    output_ams.set_bands(ams[:,1:-1],units='meV')
                    ### Qfile is not available yet
                    ### with output_folder.open('qfile','rb') as qf:
                    ###     qp=pd.read_table(qf, sep='\s+', header=None,skiprows=1)

                    ### qp.fillna(0, inplace=True)
                    ### qa=np.array(qp)
                    ### label_names=qa[qa[:,3]!=0,3]
                    ### label_idx=np.where(qa[:,3]!=0)[0]
                    ### labels=[]
                    ### for l,i in zip(label_names.tolist(),label_idx.tolist()):
                    ###     labels.append((i,l))
                    ###output_ams.labels=labels

                    self.out('ams', output_ams)
                except Exception:
                    print('No qpoint file found')
                    pass
            
        #section AMS plot
        #based on codes from UppASD repo postQ.py
        #AMSplot is a Bool  like Bool('True')
        try:
            AMSplot_settings = self.node.inputs.AMSplot.value
        except:
            AMSplot_settings = False
        if AMSplot_settings:
            with output_folder.open('inpsd.dat', 'r') as infile:
                lines=infile.readlines()
                for idx,line in enumerate(lines):
                    line_data=line.rstrip('\n').split()
                    if len(line_data)>0:
                        # Find the simulation id
                        if(line_data[0]=='simid'):
                            simid=line_data[1]
                            #print('simid: ',simid)

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

                        # Read the name of the position file
                        if(line_data[0].strip()=='posfile'):
                            with output_folder.open('posfile', 'r') as pfile:
                                lines=pfile.readlines()
                                positions=np.empty([0,3])
                                numbers=[]
                                for idx,line in enumerate(lines):
                                    line_data=line.rstrip('\n').split()
                                    if len(line_data)>0:
                                        positions=np.vstack((positions,np.asarray(line_data[2:5])))
                                        numbers=np.append(numbers,np.asarray(line_data[1]))
            hbar=4.135667662e-15
            cell=(lattice,positions,numbers)
            kpath_obj=spth.get_path(cell)
            for name in retrived_file_name_list:
                if 'ams' in name[0:4]:
                    ams_filename = name
                    with output_folder.open(ams_filename, 'rb') as f:
                        ams_tempp=np.loadtxt(f)
                        ams_dist_col=ams_tempp.shape[1]-1
                if  'sqw' in name[0:4]:
                    sqw_filename = name
                    with output_folder.open(sqw_filename, 'rb') as f:
                        sqw=np.genfromtxt(f,usecols=(0,4,5,6,7,8))
                        nq=int(sqw[-1,0])
                        nw=int(sqw[-1,1])
                        sqw_x=np.reshape(sqw[:,2],(nq,nw))
                        sqw_y=np.reshape(sqw[:,3],(nq,nw))
                        sqw_z=np.reshape(sqw[:,4],(nq,nw))
                        sqw_t=sqw_x**2+sqw_y**2

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
