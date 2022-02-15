# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import ToContext, if_, while_, WorkChain,BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode, calcfunction
from aiida.plugins import CalculationFactory, GroupFactory
from aiida.orm import Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
from aiida_uppasd.workflows.base_restart import ASDBaseRestartWorkChain

from numpy import gradient,asarray,insert
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
import json

ASDCalculation = CalculationFactory('UppASD_core_calculations')


@calcfunction
def get_temperature_data(**kwargs):
    """Store loop data in Dict node."""

    kb=1.38064852e-23/2.179872325e-21

    _labels = ['temperature','magnetization','binder_cumulant','susceptibility','specific_heat','energy']

    outputs=AttributeDict()

    for label in _labels:
        outputs[label]=[]
    for ldum, result in kwargs.items():
        for label in _labels:
            outputs[label].append(result[label])

    # Also calculate specific heat (in k_B) from temperature gradient
    T=asarray(outputs.temperature)+1.0e-12
    U=asarray(outputs.energy)
    C=gradient(U)/gradient(T)

    # Calculate the entropy
    dS=C/T
    S=integrate.cumtrapz(y=dS,x=T)
    # Use spline interpolation for improved low temperature behaviour
    Sspline = InterpolatedUnivariateSpline(T[1:], S, k=3)
    Si = Sspline(T)
    S0 = Sspline(T[0])
    S = Si-S0
    F=U-T*S

    # Store the gradient specific heat as 'dudt' as well as entropy and free energy
    C=C/kb
    outputs.dudt=C.tolist()
    outputs.entropy=S.tolist()
    outputs.free_e=F.tolist()

    return Dict(dict=outputs)


class UppASDTemperatureRestartWorkflow(WorkChain):
    """Base workchain first
    #Workchain to run an UppASD simulation with automated error handling and restarts.#"""
    _process_class = ASDCalculation

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseRestartWorkChain)
        spec.expose_outputs(ASDBaseRestartWorkChain,include=['cumulants'])

        spec.input('inpsd_temp', valid_type=Dict,
                   help='temp dict of inpsd.dat', required=False)  # default=lambda: Dict(dict={})

        spec.input('tasks', valid_type=List,
                   help='task dict for inpsd.dat', required=False)  # default=lambda: Dict(dict={})

        spec.input('temperatures', valid_type=List,
                   help='task dict for inpsd.dat', required=False)  # default=lambda: Dict(dict={})

        spec.output('temperature_output', valid_type=Dict, help='Result Dict for temperature')  


        spec.outline(
                cls.load_tasks,
                cls.loop_temperatures,
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
        
        # Override list of tasks to ensure that thermodynamic measurables are calculated
        task_dict['do_cumu']='Y'
        task_dict['plotenergy']=1

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

    def loop_temperatures(self):

        calculations = {}

        for idx,temperature in enumerate(self.inputs.temperatures):
            self.report('Running loop for temperature with value {}'.format(temperature))
            self.inputs.inpsd_dict['temp'] = temperature
            self.inputs.inpsd_dict['ip_temp'] = temperature
            inputs=self.generate_inputs()

            future = self.submit(ASDBaseRestartWorkChain, **inputs)
            calculations['T'+str(idx)] = future

        return ToContext(**calculations)


    def results(self):
        """Process results."""
        inputs = { 'T'+str(idx): self.ctx['T'+str(idx)].get_outgoing().get_node_by_label('cumulants') for idx,label in enumerate(self.inputs.temperatures) }
        temperature_output = get_temperature_data(**inputs)

        self.out('temperature_output',temperature_output)