# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
import json
from pathlib import Path

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain
from aiida.plugins import CalculationFactory

from aiida_uppasd.workflows.base import ASDBaseWorkChain

ASDCalculation = CalculationFactory('uppasd.uppasd_calculation')


class UppASDFastFerroWorkflow(WorkChain):
    """Base workchain first
    #Workchain to run an UppASD simulation with automated error handling and restarts.#"""
    _process_class = ASDCalculation

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseWorkChain)

        spec.input(
            'inpsd_temp',
            valid_type=orm.Dict,
            help='temp dict of inpsd.dat',
            required=False,
        )  # default=lambda: Dict(dict={})A

        spec.expose_outputs(ASDBaseWorkChain)

        spec.outline(
            cls.load_fmtasks,
            cls.run_base_workchain,
            cls.results,
        )
        #cls.results,

    def load_fmtasks(self):
        """
        Load the default values for the workflow.
        """

        fpath = str(Path(__file__).resolve().parent.parent) + '/defaults/tasks/'
        task_dict = {}

        _fm_tasks = ['tc_mfa', 'stiffness', 'lswt']
        for task in _fm_tasks:
            self.report(task)
            fname = fpath + str(task) + '.json'
            with open(fname, 'r') as handler:
                self.report(fname)
                tmp_dict = json.load(handler)
                task_dict.update(tmp_dict)

        task_dict.update(self.inputs.inpsd_temp.get_dict())
        self.inputs.inpsd_dict = task_dict

    def run_base_workchain(self):
        """
        Perform a calculation using the base workchain.
        """
        inputs = AttributeDict()
        inputs.code = self.inputs.code
        inputs.prepared_file_folder = self.inputs.prepared_file_folder
        inputs.except_filenames = self.inputs.except_filenames
        inputs.inpsd_dict = orm.Dict(dict=self.inputs.inpsd_dict)
        if 'exchange' in self.inputs:
            inputs.exchange = self.inputs.exchange
        inputs.retrieve_list_name = self.inputs.retrieve_list_name

        self.report('Running FM analysis')
        asd_base_result = self.submit(ASDBaseWorkChain, **inputs)

        return ToContext(asd_base_result=asd_base_result)

    def results(self):
        """
        Expose the outputs from the base workchain to the parent.
        """
        #for test we output total energy array in the workchain result
        self.out_many(self.exposed_outputs(self.ctx.asd_base_result, ASDBaseWorkChain))
