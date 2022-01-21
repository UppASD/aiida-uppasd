# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import ToContext, if_, while_, WorkChain,BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode
from aiida.plugins import CalculationFactory, GroupFactory
from aiida.orm import Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
from aiida_uppasd.workflows.base import ASDBaseWorkChain

import json

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

        spec.input('inpsd_temp', valid_type=Dict,
                   help='temp dict of inpsd.dat', required=False)  # default=lambda: Dict(dict={})

        spec.outline(
            cls.load_defaults,
            cls.run_ASDBaseWorkChain,
            cls.results,
        )

    def load_defaults(self):
        from pathlib import Path
        ### Preparing to include defaults from .json files
        fpath=str(Path(__file__).resolve().parent.parent)+'/defaults/tasks/'
        ### fname='/Users/andersb/Jobb/People/Qichen/aiida-uppasd/aiida_uppasd/workflows/defaults/spinwaves.json'
        ### with open(fname,'r') as f:
        ###     tmp_dict=json.load(f)
        ###
        ### tmp_dict.update(self.inputs.inpsd_temp.get_dict())
        ### self.inputs.inpsd_dict = tmp_dict
        self.inputs.inpsd_dict = self.inputs.inpsd_temp.get_dict()
        return

    def run_ASDBaseWorkChain(self):

        inputs = AttributeDict()
        inputs.code = self.inputs.code
        inputs.prepared_file_folder = self.inputs.prepared_file_folder
        inputs.except_filenames = self.inputs.except_filenames
        inputs.inpsd_dict = Dict(dict=self.inputs.inpsd_dict)
        inputs.exchange = self.inputs.exchange
        inputs.retrieve_list_name = self.inputs.retrieve_list_name
        ASDBaseWorkChain_result = self.submit(ASDBaseWorkChain, **inputs)

        ### self.inputs.pop('inpsd_temp')
        ### ASDBaseWorkChain_result = self.submit(ASDBaseWorkChain, **self.inputs)

        ### self.exposed_inputs(ASDBaseWorkChain)['inpsd_dict']=Dict(dict=self.inputs.inpsd_dict)
        ### ASDBaseWorkChain_result = self.submit(ASDBaseWorkChain, **self.exposed_inputs(ASDBaseWorkChain))
        return ToContext(ASDBaseWorkChain_result=ASDBaseWorkChain_result)

    def results(self):
        #for test we output total energy array in the workchain result
        self.out_many(
            self.exposed_outputs(self.ctx.ASDBaseWorkChain_result, ASDBaseWorkChain)
        )