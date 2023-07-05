# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
import json
from pathlib import Path

from numpy import asarray, gradient
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline

from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, calcfunction
from aiida.orm import Dict, Int, List, Str
from aiida.plugins import CalculationFactory

from aiida_uppasd.workflows.base_restart import ASDBaseRestartWorkChain

ASDCalculation = CalculationFactory('UppASD_core_calculations')


@calcfunction
def get_temperature_data(**kwargs):
    """Store loop data in Dict node."""

    boltzman_constant = 1.38064852e-23 / 2.179872325e-21

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
    _temperature = asarray(outputs.temperature) + 1.0e-12
    _energy = asarray(outputs.energy)
    _specific_heat = gradient(_energy) / gradient(_temperature)

    # Calculate the entropy
    d_entropy = _specific_heat / _temperature
    _entropy = integrate.cumtrapz(y=d_entropy, x=_temperature)
    # Use spline interpolation for improved low temperature behaviour

    _entropy_spline = InterpolatedUnivariateSpline(_temperature[1:], _entropy, k=3)
    _entropy_i = _entropy_spline(_temperature)
    _entropy_0 = _entropy_spline(_temperature[0])
    _entropy = _entropy_i - _entropy_0
    _free_energy = _energy - _temperature * _entropy

    # Store the gradient specific heat as 'dudt' as well as entropy and free energy
    _specific_heat = _specific_heat / boltzman_constant
    outputs.dudt = _specific_heat.tolist()
    outputs.entropy = _entropy.tolist()
    outputs.free_e = _free_energy.tolist()

    return Dict(dict=outputs)


class ThermalDynamicWorkflow(WorkChain):  # pylint: disable=too-many-public-methods
    """Base workchain first
    #Workchain to run an UppASD simulation with automated error handling and restarts.#"""
    _process_class = ASDCalculation

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseRestartWorkChain)
        spec.expose_outputs(ASDBaseRestartWorkChain, include=['cumulants'])

        spec.input(
            'inpsd_temp',
            valid_type=Dict,
            help='temp dict of inpsd.dat',
            required=False,
        )

        spec.input(
            'tasks',
            valid_type=List,
            help='task dict for inpsd.dat',
            required=False,
        )

        spec.input(
            'temperatures',
            valid_type=List,
            help='task dict for inpsd.dat',
            required=False,
        )
        spec.input(
            'external_fields',
            valid_type=List,
            help='task dict for inpsd.dat',
            required=False,
        )
        spec.input(
            'cell_size',
            valid_type=List,
            help='task dict for inpsd.dat',
            required=False,
        )
        spec.input(
            'plot_dir',
            valid_type=Str,
            help='plot dir ',
            required=False,
        )
        spec.input(
            'M_T_plot',
            valid_type=Int,
            help='flag for plotting skyrmions',
            required=False,
            default=lambda: Int(1),
        )
        spec.input(
            'M_phase_diagram',
            valid_type=Int,
            help='flag for ',
            required=False,
            default=lambda: Int(1),
        )
        spec.input(
            'susceptibility_T_plot',
            valid_type=Int,
            help='flag for plotting skyrmions',
            required=False,
            default=lambda: Int(1),
        )
        spec.input(
            'susceptibility_phase_diagram',
            valid_type=Int,
            help='flag for ',
            required=False,
            default=lambda: Int(1),
        )
        spec.input(
            'specific_heat_T_plot',
            valid_type=Int,
            help='flag for plotting skyrmions',
            required=False,
            default=lambda: Int(1),
        )
        spec.input(
            'specific_heat_phase_diagram',
            valid_type=Int,
            help='flag for ',
            required=False,
            default=lambda: Int(1),
        )

        spec.input(
            'energy_T_plot',
            valid_type=Int,
            help='flag for plotting skyrmions',
            required=False,
            default=lambda: Int(1),
        )
        spec.input(
            'energy_phase_diagram',
            valid_type=Int,
            help='flag for ',
            required=False,
            default=lambda: Int(1),
        )

        spec.input(
            'free_e_T_plot',
            valid_type=Int,
            help='flag for plotting skyrmions',
            required=False,
            default=lambda: Int(1),
        )
        spec.input(
            'free_e_phase_diagram',
            valid_type=Int,
            help='flag for ',
            required=False,
            default=lambda: Int(1),
        )

        spec.input(
            'entropy_T_plot',
            valid_type=Int,
            help='flag for plotting skyrmions',
            required=False,
            default=lambda: Int(1),
        )
        spec.input(
            'entropy_phase_diagram',
            valid_type=Int,
            help='flag for ',
            required=False,
            default=lambda: Int(1),
        )

        spec.input(
            'dudt_T_plot',
            valid_type=Int,
            help='flag for plotting skyrmions',
            required=False,
            default=lambda: Int(1),
        )
        spec.input(
            'dudt_phase_diagram',
            valid_type=Int,
            help='flag for ',
            required=False,
            default=lambda: Int(1),
        )

        spec.input(
            'binder_cumulant_T_plot',
            valid_type=Int,
            help='flag for plotting skyrmions',
            required=False,
            default=lambda: Int(1),
        )
        spec.input(
            'binder_cumulant_phase_diagram',
            valid_type=Int,
            help='flag for ',
            required=False,
            default=lambda: Int(1),
        )

        spec.output(
            'thermal_dynamic_output',
            valid_type=Dict,
            help='Result Dict for temperature',
        )
        spec.exit_code(
            701,
            'ThermalDynamic_T_error',
            message='IN TD CALC T LENGTH SHOULD LARGER THAN 1',
        )
        spec.outline(
            cls.load_tasks,
            cls.loop_temperatures,
            cls.results,
        )

    def error_report(self):
        """Report error"""
        return self.exit_codes.ThermalDynamic_T_error  # pylint: disable=no-member

    def check_mag_phase_diagram(self):
        """Check if the magnetization phase diagram should be produced"""
        if (
            self.inputs.M_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) > 1:
            return True

        return False

    def load_tasks(self):
        """Load predetermined parameters for the task"""
        fpath = str(Path(__file__).resolve().parent.parent) + '/defaults/tasks/'
        task_dict = {}
        for task in self.inputs.tasks:
            self.report(task)
            fname = fpath + str(task) + '.json'
            with open(fname, 'r', encoding='utf8') as handler:
                self.report(fname)
                tmp_dict = json.load(handler)
                task_dict.update(tmp_dict)

        # Override list of tasks to ensure that thermodynamic measurables are calculated
        task_dict['do_cumu'] = 'Y'
        task_dict['plotenergy'] = 1

        task_dict.update(self.inputs.inpsd_temp.get_dict())
        self.inputs.inpsd_dict = task_dict

    def generate_inputs(self):
        """Generate the inputs for the calculations"""
        inputs = self.exposed_inputs(
            ASDBaseRestartWorkChain
        )  #we need take input dict from the BaseRestartWorkchain, not a new one.
        inputs.code = self.inputs.code
        inputs.prepared_file_folder = self.inputs.prepared_file_folder
        inputs.except_filenames = self.inputs.except_filenames
        inputs.inpsd_dict = Dict(dict=self.inputs.inpsd_dict)
        try:
            inputs.exchange = self.inputs.exchange
        except BaseException:
            pass
        inputs.retrieve_list_name = self.inputs.retrieve_list_name

        return inputs

    def loop_temperatures(self):
        """Submit the calculations to the context"""
        calculations = {}
        for idx, cell_size in enumerate(self.inputs.cell_size):
            for idx_1, external_field in enumerate(self.inputs.external_fields):
                for idx_2, temperature in enumerate(self.inputs.temperatures):
                    self.inputs.inpsd_dict['temp'] = temperature
                    self.inputs.inpsd_dict['ip_temp'] = temperature
                    self.inputs.inpsd_dict['ip_hfield'] = external_field
                    self.inputs.inpsd_dict['ncell'] = cell_size
                    inputs = self.generate_inputs()
                    future = self.submit(ASDBaseRestartWorkChain, **inputs)
                    calculations['C' + str(idx) + 'B' + str(idx_1) + 'T' + str(idx_2)] = future
                    self.report(f"{'C' + str(idx) + 'B' + str(idx_1) + 'T' + str(idx_2)} is run")
        return ToContext(**calculations)

    def results(self):
        """Process results."""
        td_out = {}
        for idx, cell_size in enumerate(self.inputs.cell_size):
            td_out[f'{cell_size}'] = {}
            for idx_1, external_field in enumerate(self.inputs.external_fields):
                outputs = {}
                for idx_2, _ in enumerate(self.inputs.temperatures):
                    outputs['C' + str(idx) + 'B' + str(idx_1) + 'T' +
                            str(idx_2)] = self.ctx['C' + str(idx) + 'B' + str(idx_1) + 'T' +
                                                   str(idx_2)].get_outgoing().get_node_by_label('cumulants')
                temperature_output = get_temperature_data(**outputs)
                td_out[f'{cell_size}'][f'{external_field}'] = temperature_output.get_dict()
        outdict = Dict(dict=td_out).store()
        self.out('thermal_dynamic_output', outdict)
