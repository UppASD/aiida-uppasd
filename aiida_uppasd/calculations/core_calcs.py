# -*- coding: utf-8 -*-
"""
AiiDA calculation plugin wrapping the SD executable (from UppASD packages).
"""
import os
from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob


class UppASD(CalcJob):
    """
    | AiiDA calculation plugin wrapping the SD executable (from UppASD packages).

    :param CalcJob: father class
    :type CalcJob: aiida.engine.CalcJob class
    """
    @classmethod
    def define(cls, spec):
        """
        | Define inputs and outputs of the calculation.
        | Note that you can add whatever flags here as wanted.
        | (Recommend add "required=False,")

        | Remember that only if you put the output port here, your parser in
        | core_parser.py will work. otherwise you will go an error report or
        | could not see your parsed output array in database.

        | You can also use our example down there to make your output port.

        | #example:
        | #spec.output('xx_out', valid_type=ArrayData, required=False,
        | #            help='')

        | ToDo : we need to write all the possible output port here. so our user could
        | be easier.
        """

        super(UppASD, cls).define(spec)
        # input file sections :
        spec.input(
            'prepared_file_folder',
            valid_type=orm.Str,
            required=False,
            help='path to prepared_file_folder',
        )
        spec.input(
            'except_filenames',
            valid_type=orm.List,
            required=False,
            help='list of excepted filenames',
        )
        spec.input(
            'inpsd_dict',
            valid_type=orm.Dict,
            help='the dict of inpsd.dat',
            required=False,
        )
        spec.input(
            'exchange',
            valid_type=orm.Dict,
            help='the dict of exchange.dat',
            required=False,
        )
        spec.input(
            'retrieve_list_name',
            valid_type=orm.List,
            help='list of output file name',
        )
        spec.input(
            'AMSplot',
            valid_type=orm.Bool,
            help='flag for plot ams or not',
            required=False,
        )

        # output sections:
        spec.output(
            'totenergy',
            valid_type=orm.ArrayData,
            required=False,
            help='all data that stored in totenergy.out',
        )
        spec.output(
            'coord',
            valid_type=orm.ArrayData,
            required=False,
            help='all data that stored in coord.out',
        )
        spec.output(
            'ams',
            valid_type=orm.BandsData,
            required=False,
            help='Adiabatic magnon spectrum',
        )
        spec.output(
            'qpoints',
            valid_type=orm.ArrayData,
            required=False,
            help='all data that stored in qpoints.out',
        )
        spec.output(
            'averages',
            valid_type=orm.ArrayData,
            required=False,
            help='all data that stored in averages.out',
        )
        spec.output(
            'qm_sweep',
            valid_type=orm.ArrayData,
            required=False,
            help='all data that stored in qm_sweep.out',
        )
        spec.output(
            'qm_minima',
            valid_type=orm.ArrayData,
            required=False,
            help='all data that stored in qm_minima.out',
        )
        spec.output(
            'mom_states_traj',
            valid_type=orm.ArrayData,
            required=False,
            help='all data that stored in moment.out',
        )
        spec.output(
            'dmdata_out',
            valid_type=orm.ArrayData,
            required=False,
            help='all data that stored in dmdata_xx.out',
        )
        spec.output(
            'struct_out',
            valid_type=orm.ArrayData,
            required=False,
            help='all data that stored in dmdata_xx.out',
        )
        spec.output(
            'cumulants',
            valid_type=orm.Dict,
            required=False,
            help='Thermodynamic data stored in dict',
        )
        spec.output(
            'AMS_plot_var',
            valid_type=orm.Dict,
            required=False,
            help='AMS_plot data stored in dict',
        )
        spec.output(
            'cal_finish_tag',
            valid_type=orm.Str,
            required=False,
            help='Tags to detect if calculation is finished or not',
        )

        #spec.exit_code(100, 'ERROR_MISSING_OUTPUT_FILES',
        #message='Calculation did not produce all expected output files.')
        spec.exit_code(451, 'WallTimeError', message='Hit the max wall time')

    def find_out_files(self, filepath, except_files=None):  # pylint: disable=no-self-use
        """

        | find out file names in the folder of input path and exclude files in except_files(a list)

        | Note that we have set the default except file : ".DS_Store"

        :param filepath: path of input folder.
        :type filepath: str
        :param except_files: list of file name that need to be excluded.
        Make sure only need file with correct name(same name with flags)
        could be placed in the input folder., defaults to None
        :type except_files: list, optional
        :return:  a list of file names for input and generate tags.
        """

        if '.DS_Store' not in except_files:
            except_files.append('.DS_Store')
        filenames = [
            f for f in os.listdir(filepath)
            if (os.path.isfile(os.path.join(filepath, f))
                and f not in except_files)
        ]
        return filenames

    def prepare_for_submission(self, folder):
        """
        | IMPORTANT!  we assumed that you have responsibility to name all
        | file in the input folder with same name of input flag that need
        | to generate in inspd.dat file.
        |
        | And of course you can just put everything into the folder and send
        | it to aiida like what show in demo1  but remember to name 'inpsd.dat'
        | as 'inpsd' without the file extension name'.out'

        :param folder:  an `aiida.common.folders.Folder` where the plugin
        should temporarily place all files needed by the calculation.
        :type folder: aiida.common.folders.Folder
        :return: calcinfo
        """

        calcinfo = datastructures.CalcInfo()
        local_list = []
        user_define_dict_name_list = []
        auto_name = globals()
        input_filenames = self.find_out_files(
            self.inputs.prepared_file_folder.value,
            self.inputs.except_filenames.get_list())

        for name in input_filenames:
            auto_name[name] = orm.SinglefileData(file=os.path.join(
                self.inputs.prepared_file_folder.value, name)).store()

        #J_ij exchange parameters
        if 'exchange' not in input_filenames:
            with folder.open('exchange', 'a+') as handler:
                for flag in self.inputs.exchange.attributes_keys():
                    handler.write(f'{self.inputs.exchange[flag]}\n')
            user_define_dict_name_list.append(
                'exchange')  #for activation of  'exchange' flag

        if 'inpsd' not in input_filenames:  # nothing in inpsd dict
            # note that we take the inpsd.dat first, that means if we have both
            # inpsd dict and inpsd.dat file we only use inpsd.dat file
            # Create input file: inpsd.dat
            with folder.open(
                    self.options.input_filename, 'a+'
            ) as handler:  #here input_filename is an input option not a list
                for flag in self.inputs.inpsd_dict.attributes_keys():
                    handler.write(f'{flag}' +
                                  f'    {self.inputs.inpsd_dict[flag]}\n')
                for name in input_filenames:
                    handler.write(f'{name}    ./{name}\n')
                for name_ud in user_define_dict_name_list:
                    handler.write(f'{name_ud}    ./{name_ud}\n')

        for name in input_filenames:
            if str(name) != 'inpsd':
                #I believe all our user are kind people, they will not do evil
                # things with eval() function :-)
                # ToDo: replace eval() here to make sure the satisfy
                local_list.append((eval(name).uuid, eval(name).filename,
                                   eval(name).filename))
            else:
                local_list.append(
                    (eval(name).uuid, eval(name).filename, 'inpsd.dat'))
        calcinfo.local_copy_list = local_list

        input_retrieve_list_name = self.inputs.retrieve_list_name
        codeinfo = datastructures.CodeInfo()
        # note that nothing need here for UppASD,(at least now :-) )
        codeinfo.cmdline_params = []
        codeinfo.code_uuid = self.inputs.code.uuid
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = input_retrieve_list_name.get_list()
        return calcinfo
