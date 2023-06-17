# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain
from aiida.plugins import CalculationFactory

from aiida_uppasd.workflows.base import ASDBaseWorkChain

ASDCalculation = CalculationFactory('uppasd.uppasd_calculation')


class UppASDTaskWorkflow(WorkChain):
    """Base workchain first
    #Workchain to run an UppASD simulation with automated error handling and restarts.#"""
    _process_class = ASDCalculation

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseWorkChain)
        spec.expose_outputs(ASDBaseWorkChain, include=['totenergy', 'cumulants'])

        spec.input(
            'inpsd_temp',
            valid_type=orm.Dict,
            help='temp dict of inpsd.dat',
            required=False,
        )

        spec.outline(
            cls.load_defaults,
            cls.run_base_workchain,
            cls.results,
        )

    def load_defaults(self):
        """
        Load the default values for the workchain.
        """
        ### Preparing to include defaults from .json files
        self.inputs.inpsd_dict = self.inputs.inpsd_temp.get_dict()

    def run_base_workchain(self):
        """
        Perform a base UppASD calculation
        """
        inputs = AttributeDict()
        inputs.code = self.inputs.code
        inputs.prepared_file_folder = self.inputs.prepared_file_folder
        inputs.except_filenames = self.inputs.except_filenames
        inputs.inpsd_dict = orm.Dict(dict=self.inputs.inpsd_dict)
        inputs.exchange = self.inputs.exchange
        inputs.retrieve_list_name = self.inputs.retrieve_list_name
        asd_base_result = self.submit(ASDBaseWorkChain, **inputs)

        return ToContext(asb_base_result=asd_base_result)

    def results(self):
        """
        Expose the results of the workchain to the parent.
        """
        #for test we output total energy array in the workchain result
        self.out_many(self.exposed_outputs(self.ctx.asd_base_result, ASDBaseWorkChain))
