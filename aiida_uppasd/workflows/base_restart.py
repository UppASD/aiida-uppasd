# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
from aiida import orm
from aiida.engine import (
    while_,
    BaseRestartWorkChain,
    process_handler,
    ProcessHandlerReport,
)
from aiida.plugins import CalculationFactory

ASDCalculation = CalculationFactory('uppasd.uppasd_calculation')


class ASDBaseRestartWorkChain(BaseRestartWorkChain):
    """
    @author Qichen Xu
    Base restart workchain: Workchain to run an UppASD simulation with
    automated error handling and restarts.
    develop version: 0.2.0 (Feb5 2022)
    """

    _process_class = ASDCalculation

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(cls._process_class, exclude=['metadata'])

        spec.input(
            'code',
            valid_type=orm.Code,
            help='the code to run UppASD',
            required=True,
        )
        spec.input(
            'input_filename',
            valid_type=orm.Str,
            help='input filename for UppASD',
            required=False,
            default=lambda: orm.Str('inpsd.dat'),
        )
        spec.input(
            'parser_name',
            valid_type=orm.Str,
            help='parser_name for aiida-uppasd',
            required=False,
            default=lambda: orm.Str('UppASDBaseParser'),
        )
        spec.input(
            'label',
            valid_type=orm.Str,
            help='label for all calculation in this workchain',
            required=False,
            default=lambda: orm.Str('uppasd_cals'),
        )
        spec.input(
            'description',
            valid_type=orm.Str,
            help='description in each cals in uppasd_aiida base workchain',
            required=False,
            default=lambda: orm.Str('cals in uppasd_aiida base workchain'),
        )
        spec.input(
            'num_mpiprocs_per_machine',
            valid_type=orm.Int,
            help='The resource in cluster to use',
            required=False,
            # default=lambda:orm.Int(16)
        )
        spec.input(
            'num_machines',
            valid_type=orm.Int,
            help='The resource in cluster to use',
            required=False,
            # default=lambda:orm.Int(1)
        )

        spec.input(
            'max_wallclock_seconds',
            valid_type=orm.Int,
            help='wall-time limits(s)',
            required=False,
            # default=lambda: orm.Int(30 * 60)
        )
        # you could set the maximum iteration time
        # (although it is defined within aiida's base restart workchain,
        # I think it is still good to show here)
        spec.input(
            'max_iterations',
            valid_type=orm.Int,
            help='maximum iteration time for restart',
            required=False,
            default=lambda: orm.Int(5),
        )
        #
        # #here I post two input methods from aiida base restart workchain,
        # use it if needed
        # spec.input(
        #   'max_iterations', valid_type=orm.Int, default=lambda: orm.Int(5),
        # help='Maximum number of iterations the work chain will restart
        # the process to finish successfully.')

        # spec.input(
        #   'clean_workdir',
        #   valid_type=orm.Bool, default=lambda: orm.Bool(False),
        # help='If `True`, work directories of all called calculation jobs will
        # be cleaned at the end of execution.')
        #
        spec.outline(
            cls.setup,
            cls.inputs_process,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            # cls.result,
            cls.results,
        )

        spec.expose_outputs(cls._process_class)
        spec.exit_code(451, 'WallTimeError', message='Hit the max wall time')

    def results(self):
        """Here we rewrite result method and named it with results.

        Attach the outputs specified in the output specification from the last
        completed process.
        """
        node = self.ctx.children[self.ctx.iteration - 1]

        max_iterations = self.inputs.max_iterations.value
        if not self.ctx.is_finished and self.ctx.iteration >= max_iterations:
            self.report(
                f'reached the maximum number of iterations {max_iterations}: '
                f'last ran {self.ctx.process_name}<{node.pk}>'
            )
            return self.exit_codes.ERROR_MAXIMUM_ITERATIONS_EXCEEDED  # pylint: disable=no-member

        self.report(f'work chain completed after {self.ctx.iteration} iterations')

        exposed_outputs = self.exposed_outputs(node, self.process_class)

        self.out_many(exposed_outputs)

        return None

    def inputs_process(self):
        """
        Process all input data and store it into self.inputs node for auto
        restart operation.
        """
        self.report('Processing input data for the ASDBaseRestartWorkChain')
        code = self.inputs.code
        prepared_file_folder = self.inputs.prepared_file_folder
        except_filenames = self.inputs.except_filenames
        inpsd_dict = self.inputs.inpsd_dict
        retrieve_list_name = self.inputs.retrieve_list_name
        self.ctx.inputs = {
            'code': code,
            'prepared_file_folder': prepared_file_folder,
            'except_filenames': except_filenames,
            'inpsd_dict': inpsd_dict,
            'retrieve_list_name': retrieve_list_name,
            'metadata': {
                'options': {
                    'resources': {
                        'num_machines': self.inputs.num_machines.value,
                        'num_mpiprocs_per_machine': self.inputs.num_mpiprocs_per_machine.value,
                    },
                    'max_wallclock_seconds': self.inputs.max_wallclock_seconds.value,
                    'input_filename': self.inputs.input_filename.value,
                    'parser_name': self.inputs.parser_name.value,
                    'withmpi': True,
                },
                'label': self.inputs.label.value,
                'description': self.inputs.description.value,
            },
        }

        # some optional values
        if 'exchange' in self.inputs:
            self.ctx.inputs['exchange'] = self.inputs.exchange

        if 'AMSplots' in self.inputs:
            self.ctx.inputs['AMSplot'] = self.inputs.AMSplot

    # here we may say our priority in workflow will start from 500
    # and wall time is the first one
    @process_handler(
        priority=500,
        exit_codes=[
            ASDCalculation.exit_codes.WallTimeError,
        ],
    )
    def handle_out_of_walltime(self, node):  # pylint: disable=unused-argument
        """
        Handle `WallTimeError` exit code.

        1. simply restart this calculation for whole workflow

        ToDo:
        2. try to use restart mode from UppASD
        """
        self.report('WallTimeError happened')
        # 1. simply restart this calculation for whole workflow
        self.ctx.inputs['metadata']['options']['max_wallclock_seconds'] += int(600)
        self.report('600s has been added to maximum walltime and the calculation will restart')

        return ProcessHandlerReport(do_break=False)
