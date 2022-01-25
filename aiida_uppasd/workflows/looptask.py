# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import ToContext, if_, while_, WorkChain,BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode, calcfunction
from aiida.plugins import CalculationFactory, GroupFactory
from aiida.orm import Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
from aiida_uppasd.workflows.base import ASDBaseWorkChain

import json

ASDCalculation = CalculationFactory('UppASD_core_calculations')


@calcfunction
def get_loop_data(**kwargs):
    """Store loop data in Dict node."""

    _labels = ['temperature','magnetization','binder_cumulant','susceptibility','specific_heat','energy']

    outputs=AttributeDict()

    for label in _labels:
        outputs[label]=[]
    for ldum, result in kwargs.items():
        for label in _labels:
            outputs[label].append(result[label])

    return Dict(dict=outputs)


class UppASDLoopTaskWorkflow(WorkChain):
    """Base workchain first
    #Workchain to run an UppASD simulation with automated error handling and restarts.#"""
    _process_class = ASDCalculation

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseWorkChain)
        spec.expose_outputs(ASDBaseWorkChain,include=['totenergy','cumulants'])

        spec.input('inpsd_temp', valid_type=Dict,
                   help='temp dict of inpsd.dat', required=False)  # default=lambda: Dict(dict={})

        spec.input('tasks', valid_type=List,
                   help='task dict for inpsd.dat', required=False)  # default=lambda: Dict(dict={})

        spec.input('loop_key', valid_type=Str,
                   help='task dict for inpsd.dat', required=False)  # default=lambda: Dict(dict={})

        spec.input('loop_values', valid_type=List,
                   help='task dict for inpsd.dat', required=False)  # default=lambda: Dict(dict={})

        spec.output('loop_output', valid_type=Dict, help='Result Dict for loops')  


        spec.outline(
                cls.load_tasks,
                cls.loop_tasks,
                cls.results,
                )


    def load_tasks(self):
        from pathlib import Path
        fpath=str(Path(__file__).resolve().parent.parent)+'/defaults/tasks/'
        task_dict={}
        for task in self.inputs.tasks:
            self.report(task)
            fname=fpath+str(task)+'.json'
            with open(fname,'r') as f:
                self.report(fname)
                tmp_dict=json.load(f)
                task_dict.update(tmp_dict)
        
        task_dict.update(self.inputs.inpsd_temp.get_dict())
        self.inputs.inpsd_dict = task_dict
        return 

    def generate_inputs(self):
        inputs = AttributeDict()
        inputs.code = self.inputs.code
        inputs.prepared_file_folder = self.inputs.prepared_file_folder
        inputs.except_filenames = self.inputs.except_filenames
        inputs.inpsd_dict = Dict(dict=self.inputs.inpsd_dict)
        try:
            inputs.exchange = self.inputs.exchange
        except:
            pass
        inputs.retrieve_list_name = self.inputs.retrieve_list_name

        return inputs

    def loop_tasks(self):

        calculations = {}

        for idx,value in enumerate(self.inputs.loop_values):
            self.report('Running loop for variable {} with value {}'.format(self.inputs.loop_key.value,value))
            self.inputs.inpsd_dict[self.inputs.loop_key.value] = value
            self.inputs.inpsd_dict['ip_'+self.inputs.loop_key.value] = value
            inputs=self.generate_inputs()

            future = self.submit(ASDBaseWorkChain, **inputs)
            calculations['T'+str(idx)] = future

        return ToContext(**calculations)


    def results(self):
        """Process results."""
        inputs = { 'T'+str(idx): self.ctx['T'+str(idx)].get_outgoing().get_node_by_label('cumulants') for idx,label in enumerate(self.inputs.loop_values) }
        loop_output = get_loop_data(**inputs)

        self.out('loop_output',loop_output)