#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:03:40 2022

@author Qichen Xu
Workchain demo generating data for Machine Learning
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from aiida.engine import ToContext, WorkChain, calcfunction, if_
from aiida.orm import ArrayData, Dict, Int, List, Str

from aiida_uppasd.workflows.base_restart import ASDBaseRestartWorkChain
from aiida_uppasd.workflows.skyrmions_pd_and_graph import plot_realspace, read_atoms, read_vectors_data


@calcfunction
def store_sknum_data(sky_num_out, sky_avg_num_out):
    """Store the skyrmion number and average skyrmion number as arrays"""
    sk_num_for_plot = ArrayData()
    sk_num_for_plot.set_array('sky_num_out', np.array(sky_num_out.get_list()))
    sk_num_for_plot.set_array('sky_avg_num_out', np.array(sky_avg_num_out.get_list()))
    return sk_num_for_plot


class UppASDSkyrmionsWorkflow(WorkChain):
    """base workchain for skyrmions
    what we have in this workchain:
    1. generate skyrmions phase diagram and single plot via flag sk_plot, \
        generate single plot only if we have one B and one T, otherwise \
        provide phase diagram.
    # 2. generate graph database for machine learning training, here we use \
    DGL https://www.dgl.ai/. if flag 'ML_graph' is 1, we need additional \
        input to generate data. e.g.   !! this target is moved to another \
            workflow
    3. B list
    4. T list
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseRestartWorkChain)
        spec.expose_outputs(ASDBaseRestartWorkChain, include=['cumulants'])

        spec.input(
            'sk_plot',
            valid_type=Int,
            help='flag for plotting skyrmions',
            required=False,
        )
        spec.input(
            'sk_number_plot',
            valid_type=Int,
            help='flag for generating graph database ',
            required=False,
        )
        spec.input(
            'average_magnetic_moment_plot',
            valid_type=Int,
            help='flag for average_magnetic_moment plot',
            required=False
        )
        spec.input(
            'average_specific_heat_plot',
            valid_type=Int,
            help='flag for average specific heat plot',
            required=False,
        )

        spec.input(
            'inpsd_skyr',
            valid_type=Dict,
            help='dict of inpsd.dat',
            required=False,
        )

        spec.input(
            'temperatures',
            valid_type=List,
            help='T list for inpsd.dat',
            required=False,
        )
        spec.input(
            'external_fields',
            valid_type=List,
            help='B list for inpsd.dat',
            required=False,
        )
        spec.input(
            'plot_dir',
            valid_type=Str,
            help='plot dir ',
            required=False,
        )

        spec.output(
            'sk_num_for_plot',
            valid_type=ArrayData,
            help='skyrmions number ',
            required=False,
        )
        spec.outline(
            cls.submits,
            if_(cls.check_for_plot_sk)(
                cls.results_and_plot,
                cls.combined_for_pd,
            ),
            if_(cls.check_for_plot_sk_number)(
                cls.plot_skynumber,
                cls.plot_skynumber_avg,
                cls.plot_skynumber_number_heat_map,
                cls.plot_sknumber_ave_heatmap,
            ),
            if_(cls.check_plot_ave_magnetic_mom)(cls.plot_average_magnetic_moment),
            if_(cls.check_plot_ave_specific_heat)(cls.plot_average_specific_heat),
        )

    def check_for_plot_sk(self):
        """Check if the skyrmion should plotted"""
        try:
            if self.inputs.sk_plot.value > int(0):
                return True
            return False
        except BaseException:
            return False

    def check_for_plot_sk_number(self):
        """Check if the skyrmion number should be plotted"""
        try:
            if self.inputs.sk_number_plot.value > int(0):
                return True
            return False
        except BaseException:
            return False

    def check_plot_ave_magnetic_mom(self):
        """Check fi the average magnetization should be plotted"""
        try:
            if self.inputs.average_magnetic_moment_plot.value > int(0):
                return True
            return False
        except BaseException:
            return False

    def check_plot_ave_specific_heat(self):
        """Check if the average specific heat should be plotted"""
        try:
            if self.inputs.average_specific_heat_plot.value > int(0):
                return True
            return False
        except BaseException:
            return False

    def generate_inputs(self):
        """Generate the inputs for the calculation"""
        inputs = self.exposed_inputs(
            ASDBaseRestartWorkChain
        )  #we need take input dict from the BaseRestartWorkchain, not a new one.
        inputs.code = self.inputs.code
        inputs.inpsd_dict = Dict(dict=self.inputs.inpsd_dict)
        inputs.prepared_file_folder = self.inputs.prepared_file_folder
        inputs.except_filenames = self.inputs.except_filenames
        inputs.retrieve_list_name = self.inputs.retrieve_list_name
        return inputs

    def submits(self):
        """Submit the calculations"""
        calculations = {}
        self.inputs.inpsd_dict = {}
        self.inputs.inpsd_dict.update(self.inputs.inpsd_skyr.get_dict())

        for index_temperature, temperature in enumerate(self.inputs.temperatures):
            self.inputs.inpsd_dict['temp'] = temperature
            self.inputs.inpsd_dict['ip_temp'] = temperature
            for index_field, external_field in enumerate(self.inputs.external_fields):
                self.inputs.inpsd_dict['hfield'] = external_field
                inputs = self.generate_inputs()
                future = self.submit(ASDBaseRestartWorkChain, **inputs)
                self.report(f'Running loop for T with value {temperature} and B with value {external_field}')
                calculations[f'T{index_temperature}_B{index_field}'] = future

        return ToContext(**calculations)

    def results_and_plot(self):
        """Collect the results and plot the data"""
        sky_num_out = []
        sky_avg_num_out = []

        for index_temperature, _ in enumerate(self.inputs.temperatures):
            sm_list = []
            sam_list = []
            for index_field, _ in enumerate(self.inputs.external_fields):
                coord_array = self.ctx[f'T{index_temperature}_B{index_field}'].get_outgoing().get_node_by_label('coord')

                # Size of system for plot
                max_number_atoms = 1000000
                mom_states = self.ctx[f'T{index_temperature}_B{index_field}'].get_outgoing(
                ).get_node_by_label('mom_states_traj')
                points, number_atoms = read_atoms(coord_array, max_number_atoms)
                vectors, colors = read_vectors_data(mom_states, number_atoms)
                plot_dir = self.inputs.plot_dir.value
                plot_realspace(
                    points,
                    vectors,
                    colors,
                    index_temperature,
                    index_field,
                    plot_dir,
                )

                sk_data = self.ctx[f'T{index_temperature}_B{index_field}'].get_outgoing(
                ).get_node_by_label('sk_num_out')
                sm_list.append(sk_data.get_array('sk_number_data')[0])
                sam_list.append(sk_data.get_array('sk_number_data')[1])

            sky_num_out.append(sm_list)
            sky_avg_num_out.append(sam_list)

        sk_num_for_plot = store_sknum_data(List(list=sky_num_out), List(list=sky_avg_num_out))

        self.out('sk_num_for_plot', sk_num_for_plot)

    def combined_for_pd(self):
        """Combine the data for the phase diagram"""
        skyrmions_singleplot = []
        plot_dir = self.inputs.plot_dir.value
        for index_temperature, _ in enumerate(self.inputs.temperatures):
            for index_field, _ in enumerate(self.inputs.external_fields):
                skyrmions_singleplot.append(
                    Image.open(f'{plot_dir}/T{index_temperature}B{index_field}.png').crop((200, 0, 848, 640))
                )  #we need to crop the white edges
        phase_diagram = Image.new('RGB', (648 * len(self.inputs.temperatures), 640 * len(self.inputs.external_fields)))
        for index_temperature, _ in enumerate(self.inputs.temperatures):
            for index_field, _ in enumerate(self.inputs.external_fields):
                phase_diagram.paste(
                    skyrmions_singleplot[len(self.inputs.external_fields) * index_temperature + index_field],
                    (0 + 648 * index_temperature, 0 + 640 * index_field)
                )
        phase_diagram.save(f'{plot_dir}/PhaseDiagram.png', quality=100)

    def plot_skynumber(self):
        """Plot the skyrmion number"""
        plot_dir = self.inputs.plot_dir.value
        _, axes = plt.subplots()
        for index_field, external_field in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for index_temperature, _ in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx[f'T{index_temperature}_B{index_field}'].get_outgoing().get_node_by_label('sk_num_out').
                    get_array('sk_number_data')[0]
                )
            axes.plot(self.inputs.temperatures.get_list(), sk_number_list, label=f'B: {external_field[-6:]} T')
        axes.legend(fontsize='xx-small')
        axes.set_ylabel('Skyrmion number')
        axes.set_xlabel('Temperature')
        plt.savefig(f'{plot_dir}/sk_number.png')

    def plot_skynumber_avg(self):
        """Plot the average skyrmion number"""
        plot_dir = self.inputs.plot_dir.value
        _, axes = plt.subplots()
        for index_field, external_field in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for index_temperature, _ in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx[f'T{index_temperature}_B{index_field}'].get_outgoing().get_node_by_label('sk_num_out').
                    get_array('sk_number_data')[1]
                )
            axes.plot(
                self.inputs.temperatures.get_list(), sk_number_list, label=f'B: {external_field[-6:]} T'
            )  #this should be changed with different B
        axes.legend(fontsize='xx-small')
        axes.set_ylabel('Skyrmion number')
        axes.set_xlabel('Temperature')
        plt.savefig(f'{plot_dir}/sk_number_avg.png')

    def plot_skynumber_number_heat_map(self):
        """Plot the average skyrmion number heat map"""
        plot_dir = self.inputs.plot_dir.value
        heat_map = []
        for index_field, _ in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for index_temperature, _ in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx[f'T{index_temperature}_B{index_field}'].get_outgoing().get_node_by_label('sk_num_out').
                    get_array('sk_number_data')[0]
                )
            heat_map.append(sk_number_list)
        y_orginal = self.inputs.external_fields.get_list()
        y_for_plot = []
        for i in y_orginal:
            y_for_plot.append(float(i.split()[-1]))

        fig = go.Figure(
            data=go.Heatmap(
                z=heat_map,
                x=self.inputs.temperatures.get_list(),
                y=y_for_plot,
                zsmooth='best',
            )
        )

        fig.update_xaxes(range=[self.inputs.temperatures.get_list()[0], self.inputs.temperatures.get_list()[-1]])
        fig.update_yaxes(range=[y_for_plot[0], y_for_plot[-1]])
        fig.update_traces(colorscale='Jet', selector=dict(type='heatmap'))
        fig.write_image(f'{plot_dir}/sk_number_heatmap.png')

    def plot_sknumber_ave_heatmap(self):
        """Plot the skyrmion number average heat map"""
        plot_dir = self.inputs.plot_dir.value
        heat_map = []
        for index_field, _ in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for index_temperature, _ in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx[f'T{index_temperature}_B{index_field}'].get_outgoing().get_node_by_label('sk_num_out').
                    get_array('sk_number_data')[1]
                )
            heat_map.append(sk_number_list)
        y_orginal = self.inputs.external_fields.get_list()
        y_for_plot = []
        for i in y_orginal:
            y_for_plot.append(float(i.split()[-1]))

        fig = go.Figure(
            data=go.Heatmap(
                z=heat_map,
                x=self.inputs.temperatures.get_list(),
                y=y_for_plot,
                zsmooth='best',
            )
        )

        fig.update_xaxes(range=[self.inputs.temperatures.get_list()[0], self.inputs.temperatures.get_list()[-1]])
        fig.update_yaxes(range=[y_for_plot[0], y_for_plot[-1]])
        fig.update_traces(colorscale='Jet', selector=dict(type='heatmap'))
        fig.write_image(f'{plot_dir}/sk_number_heatmap_avg.png')

    def plot_average_magnetic_moment(self):
        """Plot the average magnetic moment"""
        plot_dir = self.inputs.plot_dir.value
        heat_map = []
        for index_field, _ in enumerate(self.inputs.external_fields):
            average_magnetic_moment = []
            for index_temperature, _ in enumerate(self.inputs.temperatures):
                average_magnetic_moment.append(
                    self.ctx[f'T{index_temperature}_B{index_field}'].get_outgoing().get_node_by_label('cumulants')
                    ['magnetization']
                )
            heat_map.append(average_magnetic_moment)
        y_orginal = self.inputs.external_fields.get_list()
        y_for_plot = []
        for i in y_orginal:
            y_for_plot.append(float(i.split()[-1]))

        fig = go.Figure(
            data=go.Heatmap(
                z=heat_map,
                x=self.inputs.temperatures.get_list(),
                y=y_for_plot,
                zsmooth='best',
            )
        )
        fig.update_xaxes(range=[self.inputs.temperatures.get_list()[0], self.inputs.temperatures.get_list()[-1]])
        fig.update_yaxes(range=[int(y_for_plot[0]), int(y_for_plot[-1])])
        fig.update_traces(colorscale='Jet', selector=dict(type='heatmap'))
        fig.write_image(f'{plot_dir}/average_magnetic_moment.png')

    def plot_average_specific_heat(self):
        """Plot the average specific heat"""
        plot_dir = self.inputs.plot_dir.value

        if len(self.inputs.external_fields.get_list()) == 1:
            #plot the line for one B
            for index_field, _ in enumerate(self.inputs.external_fields):
                energy_list = []
                temperature_list = self.inputs.temperatures.get_list()
                for index_temperature, _ in enumerate(self.inputs.temperatures):
                    energy_list.append(
                        self.ctx[f'T{index_temperature}_B{index_field}'].get_outgoing().get_node_by_label('cumulants')
                        ['energy']
                    )
                de_dt = (np.gradient(np.array(energy_list)) / np.gradient(np.array(temperature_list)))
                de_dt = (de_dt / de_dt[0]).tolist()
            fig, axes = plt.subplots()
            axes.plot(self.inputs.temperatures.get_list(), de_dt)
            axes.legend(fontsize='xx-small')
            axes.set_ylabel('C')
            axes.set_xlabel('Temperature')
            plt.savefig(f'{plot_dir}/CV_T.png')

        else:
            heat_map = []
            for index_field, _ in enumerate(self.inputs.external_fields):
                energy_list = []
                temperature_list = self.inputs.temperatures.get_list()

                for index_temperature, _ in enumerate(self.inputs.temperatures):
                    energy_list.append(
                        self.ctx[f'T{index_temperature}_B{index_field}'].get_outgoing().get_node_by_label('cumulants')
                        ['energy']
                    )
                de_dt = (np.gradient(np.array(energy_list)) / np.gradient(np.array(temperature_list))).tolist()
                heat_map.append(de_dt)
            y_orginal = self.inputs.external_fields.get_list()
            y_for_plot = []
            for i in y_orginal:
                y_for_plot.append(float(i.split()[-1]))

            fig = go.Figure(
                data=go.Heatmap(
                    z=heat_map,
                    x=self.inputs.temperatures.get_list(),
                    y=y_for_plot,
                    zsmooth='best',
                )
            )

            fig.update_xaxes(range=[self.inputs.temperatures.get_list()[0], self.inputs.temperatures.get_list()[-1]])
            fig.update_yaxes(range=[int(y_for_plot[0]), int(y_for_plot[-1])])
            fig.update_traces(colorscale='Jet', selector=dict(type='heatmap'))
            fig.write_image(f'{plot_dir}/average_specific_heat.png')
