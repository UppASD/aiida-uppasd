# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
import json
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, calcfunction
from aiida.plugins import CalculationFactory

from aiida_uppasd.workflows.base import ASDBaseWorkChain

ASDCalculation = CalculationFactory('uppasd.uppasd_calculation')


@calcfunction
def get_temperature_data(**kwargs):  # pylint: disable=too-many-locals
    """Store loop data in Dict node."""

    boltzmann_constant = 1.38064852e-23 / 2.179872325e-21

    _labels = [
        'temperature',
        'magnetization',
        'binder_cumulant',
        'susceptibility',
        'specific_heat',
        'energy',
    ]

    outputs = AttributeDict()

    for label in _labels:
        outputs[label] = []
    for _, result in kwargs.items():
        for label in _labels:
            outputs[label].append(result[label])

    # Also calculate specific heat (in k_B) from temperature gradient
    _temperature = np.asarray(outputs.temperature) + 1.0e-12
    _energy = np.asarray(outputs.energy)
    _specific_heat = np.gradient(_energy) / np.gradient(_temperature)

    # Calculate the entropy
    _entropy_derivative = _specific_heat / _temperature
    _entropy = integrate.cumtrapz(y=_entropy_derivative, x=_temperature)
    # Use spline interpolation for improved low temperature behavior
    _entropy_spline = InterpolatedUnivariateSpline(
        _temperature[1:],
        _entropy,
        k=3,
    )
    _entropy_full = _entropy_spline(_temperature)
    _entropy_zero = _entropy_spline(_temperature[0])
    _energy = _entropy_full - _entropy_zero
    _free_energy = _energy - _temperature * _entropy

    # Store the gradient specific heat as 'dudt' as well as entropy and free energy
    _specific_heat = _specific_heat / boltzmann_constant
    outputs.dudt = _specific_heat.tolist()
    outputs.entropy = _entropy.tolist()
    outputs.free_e = _free_energy.tolist()

    return orm.Dict(dict=outputs)


class UppASDTemperatureWorkflow(WorkChain):
    """Base workchain first
    #Workchain to run an UppASD simulation with automated error handling and restarts.#"""
    _process_class = ASDCalculation

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseWorkChain)
        spec.expose_outputs(ASDBaseWorkChain, include=['cumulants'])

        spec.input(
            'inpsd_temp',
            valid_type=orm.Dict,
            help='temp dict of inpsd.dat',
            required=False,
        )

        spec.input(
            'tasks',
            valid_type=orm.List,
            help='task dict for inpsd.dat',
            required=False,
        )

        spec.input(
            'temperatures',
            valid_type=orm.List,
            help='task dict for inpsd.dat',
            required=False,
        )

        spec.output(
            'temperature_output',
            valid_type=orm.Dict,
            help='Result Dict for temperature',
        )

        spec.outline(
            cls.load_tasks,
            cls.loop_temperatures,
            cls.results,
        )

    def load_tasks(self):
        """
        Load the default values for the workflow.
        """
        fpath = str(Path(__file__).resolve().parent.parent) + '/defaults/tasks/'
        task_dict = {}
        for task in self.inputs.tasks:
            self.report(task)
            fname = fpath + str(task) + '.json'
            with open(fname, 'r') as handler:
                self.report(fname)
                tmp_dict = json.load(handler)
                task_dict.update(tmp_dict)

        # Override list of tasks to ensure that thermodynamic measurables are calculated
        task_dict['do_cumu'] = 'Y'
        task_dict['plotenergy'] = 1

        task_dict.update(self.inputs.inpsd_temp.get_dict())
        self.inputs.inpsd_dict = task_dict

    def generate_inputs(self):
        """
        Generate the inputs needed for the workchain
        """
        inputs = AttributeDict()
        inputs.code = self.inputs.code
        inputs.prepared_file_folder = self.inputs.prepared_file_folder
        inputs.except_filenames = self.inputs.except_filenames
        inputs.inpsd_dict = orm.Dict(dict=self.inputs.inpsd_dict)
        if 'exchange' in self.inputs:
            inputs.exchange = self.inputs.exchange
        inputs.retrieve_list_name = self.inputs.retrieve_list_name

        return inputs

    def loop_temperatures(self):
        """
        Submit a workchain for each of the temperatures to be studied
        """
        calculations = {}

        for idx, temperature in enumerate(self.inputs.temperatures):
            self.report(f'Running loop for temperature with value {temperature}')
            self.inputs.inpsd_dict['temp'] = temperature
            self.inputs.inpsd_dict['ip_temp'] = temperature
            inputs = self.generate_inputs()

            future = self.submit(ASDBaseWorkChain, **inputs)
            calculations['T' + str(idx)] = future

        return ToContext(**calculations)

    def results(self):
        """Process results."""
        inputs = {
            'T' + str(idx): self.ctx['T' + str(idx)].get_outgoing().get_node_by_label('cumulants')
            for idx, _ in enumerate(self.inputs.temperatures)
        }
        temperature_output = get_temperature_data(**inputs)

        self.out('temperature_output', temperature_output)
