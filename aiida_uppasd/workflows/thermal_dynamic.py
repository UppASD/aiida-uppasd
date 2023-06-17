# -*- coding: utf-8 -*-
"""Workchain to run an UppASD simulation with automated error handling and restarts."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray, gradient
import plotly.graph_objects as go
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline

from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, calcfunction, if_
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


def plot_pd(
    plot_dir,
    heat_map,
    x_label_list,
    y_label_list,
    plot_name,
    xlabel,
    ylabel,
):
    #pylint: disable=too-many-arguments
    """Plotting the phase diagram"""
    fig = go.Figure(data=go.Heatmap(
        z=heat_map,
        x=x_label_list,
        y=y_label_list,
        zsmooth='best',
    ))
    fig.update_xaxes(range=[int(float(x_label_list[0])), int(float(x_label_list[-1]))])
    fig.update_yaxes(range=[int(float(y_label_list[0])), int(float(y_label_list[-1]))])
    fig.update_traces(colorscale='Jet', selector=dict(type='heatmap'))
    fig.update_layout(yaxis={'title': f'{xlabel}', 'tickangle': -90}, xaxis={'title': f'{ylabel}'})
    fig.write_image(f"{plot_dir}/{plot_name.replace(' ', '_')}.png")


def plot_line(
    x_data,
    y_data,
    line_name_list,
    x_label,
    y_label,
    plot_path,
    plot_name,
    leg_name_list,
):
    #pylint: disable=too-many-arguments
    """Line plot"""
    #Since we need all lines in one plot here x and y should be a dict with the line name on that
    plt.figure()
    _, axes = plt.subplots()
    for index, _entry in enumerate(line_name_list):
        axes.plot(
            x_data,
            y_data[_entry],
            label=f'{leg_name_list[index]}',
        )
    axes.legend()
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    plt.savefig(f'{plot_path}/{plot_name}.png')
    plt.close()


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
            if_(cls.check_mag_phase_diagram)(cls.plot_mag_phase_diagram
                                             ).elif_(cls.plot_mag_temp)(cls.plot_mag_temp).else_(cls.error_report),
            if_(cls.check_sus_phase_diagram)(cls.plot_sus_phase_diagram).elif_(cls.check_susceptibility_temp
                                                                               )(cls.plot_susceptibility_temp
                                                                                 ).else_(cls.error_report),
            if_(cls.check_cev_phase_diagram)(cls.plot_cev_phase_diagram).elif_(cls.check_specific_heat_temp
                                                                               )(cls.plot_specific_heat_temp
                                                                                 ).else_(cls.error_report),
            if_(cls.check_free_e_phase_diagram)(cls.plot_free_e_phase_diagram).elif_(cls.check_free_e_temp
                                                                                     )(cls.plot_free_e_temp
                                                                                       ).else_(cls.error_report),
            if_(cls.check_entropy_phase_diagram)(cls.plot_entropy_phase_diagram).elif_(cls.check_entropy_temp
                                                                                       )(cls.plot_entropy_temp
                                                                                         ).else_(cls.error_report),
            if_(cls.check_energy_phase_diagram)(cls.plot_energy_phase_diagram).elif_(cls.check_energy_temp
                                                                                     )(cls.plot_energy_temp
                                                                                       ).else_(cls.error_report),
            if_(cls.check_dudt_phase_diagram)(cls.plot_dudt_phase_diagram
                                              ).elif_(cls.check_dudt_temp)(cls.plot_dudt_temp).else_(cls.error_report),
            if_(cls.check_binder_cumu_phase_diagram)(cls.plot_binder_cumu_phase_diagram
                                                     ).elif_(cls.check_binder_cumulant_temp
                                                             )(cls.plot_binder_cumulant_temp).else_(cls.error_report),
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

    def plot_mag_phase_diagram(self):
        """Plot the magnetization phase diagram"""
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for _, cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict[f'{cell_size}']
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['magnetization'])
            plot_pd(
                self.inputs.plot_dir.value, pd_for_plot, self.inputs.temperatures.get_list(), y_label_list,
                ('M_T' + str(cell_size)), 'B', 'T'
            )

    def check_mag_temp(self):
        """Check if the magnetization phase diagram should be plotted"""
        if (
            self.inputs.M_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) == 1:
            return True
        return False

    def plot_mag_temp(self):
        """Plot the magnetization as a function of temperature"""
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for _, cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict[f'{cell_size}']
            leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['magnetization']
                line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
                index = index + 1
        plot_line(
            self.inputs.temperatures.get_list(),
            line_for_plot,
            line_name_list,
            'T',
            'M',
            self.inputs.plot_dir.value,
            'M_T',
            leg_name_list,
        )

        #specific_heat
    def check_cev_phase_diagram(self):
        """Check if the specific heat should be produced"""
        if (
            self.inputs.specific_heat_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) > 1:
            return True

        return False

    def plot_cev_phase_diagram(self):
        """Check if the specific heat phase diagram should be produced"""
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for _, cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict[f'{cell_size}']
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['specific_heat'])
            plot_pd(
                self.inputs.plot_dir.value, pd_for_plot, self.inputs.temperatures.get_list(), y_label_list,
                ('specific_heat_T' + str(cell_size)), 'B', 'T'
            )

    def check_specific_heat_temp(self):
        """Check if the specific heat vs temperature should be produced"""
        if (
            self.inputs.specific_heat_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) == 1:
            return True
        return False

    def plot_specific_heat_temp(self):
        """Plot the specific head as as function of temperature"""
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for _, cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict[f'{cell_size}']
            leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['specific_heat']
                line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
                index = index + 1
        plot_line(
            self.inputs.temperatures.get_list(), line_for_plot, line_name_list, 'T', 'specific_heat',
            self.inputs.plot_dir.value, 'specific_heat_T', leg_name_list
        )

        #susceptibility
    def check_sus_phase_diagram(self):
        """Check if the magnetic susceptibility phase diagram should be produced"""
        if (
            self.inputs.susceptibility_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) > 1:
            return True
        return False

    def plot_sus_phase_diagram(self):
        """Plot the magnetic susceptibility phase diagram"""
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for _, cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict[f'{cell_size}']
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['susceptibility'])
            plot_pd(
                self.inputs.plot_dir.value, pd_for_plot, self.inputs.temperatures.get_list(), y_label_list,
                ('susceptibility_T' + str(cell_size)), 'B', 'T'
            )

    def check_susceptibility_temp(self):
        """Check if the susceptibility vs temp should be produced"""
        if (
            self.inputs.susceptibility_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) == 1:
            return True
        return False

    def plot_susceptibility_temp(self):
        """Plot the susceptibility as a function of temperature"""
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for _, cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict[f'{cell_size}']
            leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['susceptibility']
                line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
                index = index + 1
        plot_line(
            self.inputs.temperatures.get_list(), line_for_plot, line_name_list, 'T', 'Susceptibility',
            self.inputs.plot_dir.value, 'Susceptibility_T', leg_name_list
        )

        #free_e
    def check_free_e_phase_diagram(self):
        """Check if the free energy diagram should be produced"""
        if (
            self.inputs.free_e_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) > 1:
            return True
        return False

    def plot_free_e_phase_diagram(self):
        """Plot the free energy phase diagram"""
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for _, cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict[f'{cell_size}']
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['free_e'])
            plot_pd(
                self.inputs.plot_dir.value,
                pd_for_plot,
                self.inputs.temperatures.get_list(),
                y_label_list,
                ('free_e_T' + str(cell_size)),
                'B',
                'T',
            )

    def check_free_e_temp(self):
        """Check if the free energy vs temperature should be produced"""
        if (
            self.inputs.free_e_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) == 1:
            return True
        return False

    def plot_free_e_temp(self):
        """Plot the free energy as a function of temperature"""
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for _, cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict[f'{cell_size}']
            leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['free_e']
                line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
                index = index + 1
        plot_line(
            self.inputs.temperatures.get_list(), line_for_plot, line_name_list, 'T', 'free_e',
            self.inputs.plot_dir.value, 'free_e_T', leg_name_list
        )

        #entropy
    def check_entropy_phase_diagram(self):
        """Check if the entropy phase diagram should be produced"""
        if (
            self.inputs.entropy_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) > 1:
            return True
        return False

    def plot_entropy_phase_diagram(self):
        """Plot the entropy phase diagram"""
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for _, cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict[f'{cell_size}']
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['entropy'])
            plot_pd(
                self.inputs.plot_dir.value, pd_for_plot, self.inputs.temperatures.get_list(), y_label_list,
                ('entropy_T' + str(cell_size)), 'B', 'T'
            )

    def check_entropy_temp(self):
        """Check if the entropy vs temperature should be produced"""
        if (
            self.inputs.entropy_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) == 1:
            return True
        return False

    def plot_entropy_temp(self):
        """Plot the entropy vs temperature"""
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for _, cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict[f'{cell_size}']
            leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['entropy']
                line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
                index = index + 1
        plot_line(
            self.inputs.temperatures.get_list(), line_for_plot, line_name_list, 'T', 'Entropy',
            self.inputs.plot_dir.value, 'Entropy_T', leg_name_list
        )

        #energy
    def check_energy_phase_diagram(self):
        """Check if the energy phase diagram should be produced"""
        if (
            self.inputs.energy_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) > 1:
            return True

        return False

    def plot_energy_phase_diagram(self):
        """Plot the energy phase diagram"""
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for _, cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict[f'{cell_size}']
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['energy'])
            plot_pd(
                self.inputs.plot_dir.value, pd_for_plot, self.inputs.temperatures.get_list(), y_label_list,
                ('energy_T' + str(cell_size)), 'B', 'T'
            )

    def check_energy_temp(self):
        """Check if the energy vs temperature should be produced"""
        if (
            self.inputs.energy_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) == 1:
            return True
        return False

    def plot_energy_temp(self):
        """Plot the energy vs temperature"""
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for _, cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict[f'{cell_size}']
            leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['energy']
                line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
                index = index + 1
        plot_line(
            self.inputs.temperatures.get_list(), line_for_plot, line_name_list, 'T', 'energy',
            self.inputs.plot_dir.value, 'energy_T', leg_name_list
        )

        #dudt
    def check_dudt_phase_diagram(self):
        """Check if the variation of energy phase diagram should be produced"""
        if (
            self.inputs.dudt_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) > 1:
            return True
        return False

    def plot_dudt_phase_diagram(self):
        """Plot the variation of the energy phase diagram"""
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for _, cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict[f'{cell_size}']
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['dudt'])
            plot_pd(
                self.inputs.plot_dir.value, pd_for_plot, self.inputs.temperatures.get_list(), y_label_list,
                ('dudt_T' + str(cell_size)), 'B', 'T'
            )

    def check_dudt_temp(self):
        """Check if the variation of energy vs temperature should be produced"""
        if (
            self.inputs.dudt_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) == 1:
            return True
        return False

    def plot_dudt_temp(self):
        """Plot the variation of the energy vs temperature"""
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for _, cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict[f'{cell_size}']
            leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['dudt']
                line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
                index = index + 1
        plot_line(
            self.inputs.temperatures.get_list(),
            line_for_plot,
            line_name_list,
            'T',
            'dudt',
            self.inputs.plot_dir.value,
            'dudt_T',
            leg_name_list,
        )

        #binder_cumulant
    def check_binder_cumu_phase_diagram(self):
        """Check if the binder cumulant phase diagram should be produced"""
        if (
            self.inputs.binder_cumulant_phase_diagram.value > int(0) and
            len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())
        ) > 1:
            return True
        return False

    def plot_binder_cumu_phase_diagram(self):
        """Plot the binder cumulant phase diagram"""
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for _, cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict[f'{cell_size}']
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['binder_cumulant'])
            plot_pd(
                self.inputs.plot_dir.value, pd_for_plot, self.inputs.temperatures.get_list(), y_label_list,
                ('binder_cumulant_T' + str(cell_size)), 'B', 'T'
            )

    def check_binder_cumulant_temp(self):
        """Check if the binder cumulant as a function of temperature should be produced"""
        if (
            self.inputs.binder_cumulant_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and
            len(self.inputs.external_fields.get_list())
        ) == 1:
            return True
        return False

    def plot_binder_cumulant_temp(self):
        """Plot the binder cumulant as as function of temperature"""
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for _, cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict[f'{cell_size}']
            leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['binder_cumulant']
                line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
                index = index + 1
        plot_line(
            self.inputs.temperatures.get_list(),
            line_for_plot,
            line_name_list,
            'T',
            'binder_cumulant',
            self.inputs.plot_dir.value,
            'binder_cumulant_T',
            leg_name_list,
        )

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
        self.ctx.td_dict = td_out
        outdict = Dict(dict=td_out).store()
        self.out('thermal_dynamic_output', outdict)
