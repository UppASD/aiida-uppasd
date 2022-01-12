# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import ToContext, if_, while_, BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode
from aiida.plugins import CalculationFactory, GroupFactory

ASDCalculation = CalculationFactory('UppASD_core_calcs')


class ASDBaseWorkChain(BaseRestartWorkChain):
    """Workchain to run an UppASD simulation with automated error handling and restarts."""

    _process_class = ASDCalculation

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(ASDCalculation)

        spec.outline(
                cls.setup, 
                while_(cls.should_run_process)( 
                    cls.run_process, cls.inspect_process,
                )
        )

        spec.expose_outputs(ASDCalculation)

