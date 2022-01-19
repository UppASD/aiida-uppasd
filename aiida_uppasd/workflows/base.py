# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import ToContext, if_, while_, WorkChain,BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode
from aiida.plugins import CalculationFactory, GroupFactory
from aiida.orm import Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData

ASDCalculation = CalculationFactory('UppASD_core_calculations')
class ASDBaseWorkChain(BaseRestartWorkChain):
    """Base workchain first
    #Workchain to run an UppASD simulation with automated error handling and restarts.#"""
    _process_class = ASDCalculation

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(cls._process_class,exclude=['metadata'])
        spec.input(
            'code',
            valid_type=orm.Code,
            help='the code to run UppASD (preset code on local computer or remote cluster)',
            required=True,
        )
        # spec.input(
        #     'inpsd_dict',
        #     valid_type=orm.Dict,
        #     help='Dict for inputs',
        #     required=False,
        # )
        # spec.input(
        #     'exchange_dict',
        #     valid_type=orm.Dict,
        #     help='Dict for exchange',
        #     required=False,
        # )
        # spec.input(
        #     'prepared_file_folder',
        #     valid_type=orm.Str,
        #     help='folder name for prepared input files',
        #     required=False,
        # )
        # spec.input(
        #     'except_filenames',
        #     valid_type=orm.List,
        #     help='list of file name for exception',
        #     required=False,
        # )
        # spec.input(
        #     'retrieve_list_name',
        #     valid_type=orm.List,
        #     help='list of file name for retrieve',
        #     required=False,
        # )
        spec.input(
            'input_filename',
            valid_type=orm.Str,
            help='input filename for UppASD',
            required=False,
            default=lambda: orm.Str('inpsd.dat')
        )
        spec.input(
            'parser_name',
            valid_type=orm.Str,
            help='parser_name for aiida-uppasd',
            required=False,
            default=lambda: orm.Str('UppASD_core_parsers')
        )
        spec.input(
            'label',
            valid_type=orm.Str,
            help='label for all calculation in this workchain',
            required=False,
            default=lambda: orm.Str('uppasd_cals')
        )
        spec.input(
            'description',
            valid_type=orm.Str,
            help='description in each cals in uppasd_aiida base workchain',
            required=False,
            default=lambda: orm.Str('cals in uppasd_aiida base workchain')
        )
        spec.input(
                    'num_mpiprocs_per_machine',
                    valid_type=orm.Int,
                    help='The resource in cluster to use',
                    required=False,
                    default=lambda:orm.Int(16)
                )
        spec.input(
                    'num_machines',
                    valid_type=orm.Int,
                    help='The resource in cluster to use',
                    required=False,
                    default=lambda:orm.Int(1)
                )            
            
        spec.input(
            'max_wallclock_seconds',
            valid_type=orm.Int,
            help='wall-time limits(s)',
            required=False,
            default=lambda: orm.Int(30 * 60)
        )


        spec.outline(
            cls.uppasd,
            cls.results,
        )

        spec.expose_outputs(ASDCalculation)
        '''
        spec.output('totenergy', valid_type=ArrayData,
                    help='all data that stored in totenergy.out')
        spec.output('coord', valid_type=ArrayData, required=False,
                    help='all data that stored in coord.out')
        spec.output('qpoints', valid_type=ArrayData, required=False,
                    help='all data that stored in qpoints.out')
        spec.output('averages', valid_type=ArrayData, required=False,
                    help='all data that stored in averages.out')
        spec.output('qm_sweep', valid_type=ArrayData, required=False,
                    help='all data that stored in qm_sweep.out')
        spec.output('qm_minima', valid_type=ArrayData, required=False,
                    help='all data that stored in qm_minima.out')
        spec.output('mom_states_traj', valid_type=ArrayData, required=False,
                    help='all data that stored in moment.out')
        spec.output('dmdata_out', valid_type=ArrayData, required=False,
                    help='all data that stored in dmdata_xx.out')
        spec.output('struct_out', valid_type=ArrayData, required=False,
                    help='all data that stored in dmdata_xx.out')
        '''
        spec.exit_code(400, 'WallTimeError', message='Hit the max wall time')


    def uppasd(self):
        """Add two numbers using the `ArithmeticAddCalculation` calculation job plugin."""
        builder = ASDCalculation.get_builder()
        builder.code = self.inputs.code
        builder.prepared_file_folder = self.inputs.prepared_file_folder
        builder.except_filenames = self.inputs.except_filenames
        builder.inpsd_dict = self.inputs.inpsd_dict
        builder.exchange = self.inputs.exchange
        builder.retrieve_list_name = self.inputs.retrieve_list_name
        builder.metadata.options.resources ={'num_machines': self.inputs.num_machines.value,'num_mpiprocs_per_machine':self.inputs.num_mpiprocs_per_machine.value}
        builder.metadata.options.max_wallclock_seconds =self.inputs.max_wallclock_seconds.value
        builder.metadata.options.input_filename =self.inputs.input_filename.value
        builder.metadata.options.parser_name =self.inputs.parser_name.value
        builder.metadata.label = self.inputs.label.value
        builder.metadata.description = self.inputs.description.value
        job_node = self.submit(builder)

        return ToContext(uppasd_result=job_node)

    def results(self):
        #for test we output total energy array in the workchain result
        self.out_many(self.exposed_outputs(self.ctx.uppasd_result,ASDCalculation))