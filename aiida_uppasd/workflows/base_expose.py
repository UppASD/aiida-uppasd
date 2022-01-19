# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import ToContext, if_, while_, WorkChain,BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode
from aiida.plugins import CalculationFactory, GroupFactory
from aiida.orm import Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
from aiida_uppasd.workflows.base import ASDBaseWorkChain
ASDCalculation = CalculationFactory('UppASD_core_calculations')
class UppASDTaskWorkflow(WorkChain):
    """Base workchain first
    #Workchain to run an UppASD simulation with automated error handling and restarts.#"""
    _process_class = ASDCalculation

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseWorkChain)
        spec.expose_outputs(ASDBaseWorkChain,include=['totenergy'])

        spec.outline(
            cls.run_ASDBaseWorkChain,
            cls.results,
        )

    def run_ASDBaseWorkChain(self):
        ASDBaseWorkChain_result = self.submit(ASDBaseWorkChain, **self.exposed_inputs(ASDBaseWorkChain))
        return ToContext(ASDBaseWorkChain_result=ASDBaseWorkChain_result)

    def results(self):
        #for test we output total energy array in the workchain result
        self.out_many(
            self.exposed_outputs(self.ctx.ASDBaseWorkChain_result, ASDBaseWorkChain)
        )
