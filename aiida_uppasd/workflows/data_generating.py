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


@calcfunction
def store_sknum_data(sky_num_out, sky_avg_num_out):
    sk_num_for_plot = ArrayData()
    sk_num_for_plot.set_array('sky_num_out', np.array(sky_num_out.get_list()))
    sk_num_for_plot.set_array('sky_avg_num_out', np.array(sky_avg_num_out.get_list()))
    return sk_num_for_plot


class UppASDSkyrmionsWorkflow(WorkChain):
    """base workchain for skyrmions
    what we have in this workchain:
    1. generate skyrmions phase diagram and single plot via flag sk_plot, generate single plot only if we have one B and one T, otherwise provide phase diagram.
    # 2. generate graph database for machine learning training, here we use DGL https://www.dgl.ai/. if flag 'ML_graph' is 1, we need additional input to generate data. e.g.   !! this target is moved to another workflow
    3. B list
    4. T list
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseRestartWorkChain)
        spec.expose_outputs(ASDBaseRestartWorkChain, include=['cumulants'])

        spec.input('sk_plot', valid_type=Int, help='flag for plotting skyrmions', required=False)
        spec.input('sk_number_plot', valid_type=Int, help='flag for generating graph database ', required=False)
        spec.input(
            'average_magnetic_moment_plot',
            valid_type=Int,
            help='flag for average_magnetic_moment plot',
            required=False
        )
        spec.input(
            'average_specific_heat_plot', valid_type=Int, help='flag for average specific heat plot', required=False
        )

        spec.input(
            'inpsd_skyr', valid_type=Dict, help='dict of inpsd.dat', required=False
        )  # default=lambda: Dict(dict={})

        spec.input(
            'temperatures', valid_type=List, help='T list for inpsd.dat', required=False
        )  # default=lambda: Dict(dict={})
        spec.input(
            'external_fields', valid_type=List, help='B list for inpsd.dat', required=False
        )  # default=lambda: Dict(dict={})
        spec.input('plot_dir', valid_type=Str, help='plot dir ', required=False)

        spec.output('sk_num_for_plot', valid_type=ArrayData, help='skyrmions number ', required=False)
        spec.outline(
            cls.submits,
            if_(cls.check_for_plot_sk)(
                cls.results_and_plot,
                cls.combined_for_PD,
            ),
            if_(cls.check_for_plot_sk_number)(
                cls.plot_skynumber,
                cls.plot_skynumber_avg,
                cls.plot_skynumber_number_heat_map,
                cls.plot_skynumber_number_avg_heat_map,
            ),
            if_(cls.check_for_plot_average_magnetic_moment)(cls.plot_average_magnetic_moment),
            if_(cls.check_for_plot_average_specific_heat)(cls.plot_average_specific_heat),
        )

    def check_for_plot_sk(self):
        try:
            if self.inputs.sk_plot.value > int(0):
                return True
            else:
                return False
        except:
            return False

    def check_for_plot_sk_number(self):
        try:
            if self.inputs.sk_number_plot.value > int(0):
                return True
            else:
                return False
        except:
            return False

    def check_for_plot_average_magnetic_moment(self):
        try:
            if self.inputs.average_magnetic_moment_plot.value > int(0):
                return True
            else:
                return False
        except:
            return False

    def check_for_plot_average_specific_heat(self):
        try:
            if self.inputs.average_specific_heat_plot.value > int(0):
                return True
            else:
                return False
        except:
            return False

    # def check_for_plot(self):
    #     if self.inputs.sk_plot.value > int(0):
    #         return True
    #     else:
    #         return False
    def generate_inputs(self):
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
        calculations = {}
        self.inputs.inpsd_dict = {}
        self.inputs.inpsd_dict.update(self.inputs.inpsd_skyr.get_dict())

        for idx_T, temperature in enumerate(self.inputs.temperatures):
            self.inputs.inpsd_dict['temp'] = temperature
            self.inputs.inpsd_dict['ip_temp'] = temperature
            for idx_B, externel_B in enumerate(self.inputs.external_fields):
                self.inputs.inpsd_dict['hfield'] = externel_B
                inputs = self.generate_inputs()
                future = self.submit(ASDBaseRestartWorkChain, **inputs)
                self.report(f'Running loop for T with value {temperature} and B with value {externel_B}')
                calculations['T' + str(idx_T) + '_' + 'B' + str(idx_B)] = future

        return ToContext(**calculations)

    def results_and_plot(self):
        sky_num_out = []
        sky_avg_num_out = []

        for idx_T, temperature in enumerate(self.inputs.temperatures):
            sm_list = []
            sam_list = []
            for idx_B, externel_B in enumerate(self.inputs.external_fields):
                coord_array = self.ctx['T' + str(idx_T) + '_' + 'B' +
                                       str(idx_B)].get_outgoing().get_node_by_label('coord')

                # Size of system for plot
                Nmax = 1000000
                mom_states = self.ctx['T' + str(idx_T) + '_' + 'B' +
                                      str(idx_B)].get_outgoing().get_node_by_label('mom_states_traj')
                points, nrAtoms = readAtoms(coord_array, Nmax)
                vectors, colors = readVectorsData(mom_states, nrAtoms, Nmax)
                plot_dir = self.inputs.plot_dir.value
                plot_realspace(points, nrAtoms, vectors, colors, Nmax, idx_T, idx_B, plot_dir)

                sk_data = self.ctx['T' + str(idx_T) + '_' + 'B' +
                                   str(idx_B)].get_outgoing().get_node_by_label('sk_num_out')
                sm_list.append(sk_data.get_array('sk_number_data')[0])
                sam_list.append(sk_data.get_array('sk_number_data')[1])

            sky_num_out.append(sm_list)
            sky_avg_num_out.append(sam_list)

        sk_num_for_plot = store_sknum_data(List(list=sky_num_out), List(list=sky_avg_num_out))

        self.out('sk_num_for_plot', sk_num_for_plot)

    def combined_for_PD(self):
        skyrmions_singleplot = []
        plot_dir = self.inputs.plot_dir.value
        for idx_T, temperature in enumerate(self.inputs.temperatures):
            for idx_B, externel_B in enumerate(self.inputs.external_fields):
                skyrmions_singleplot.append(
                    Image.open(f'{plot_dir}/T{idx_T}B{idx_B}.png').crop((200, 0, 848, 640))
                )  #we need to crop the white edges
        phase_diagram = Image.new('RGB', (648 * len(self.inputs.temperatures), 640 * len(self.inputs.external_fields)))
        for idx_T, temperature in enumerate(self.inputs.temperatures):
            for idx_B, externel_B in enumerate(self.inputs.external_fields):
                phase_diagram.paste(
                    skyrmions_singleplot[len(self.inputs.external_fields) * idx_T + idx_B],
                    (0 + 648 * idx_T, 0 + 640 * idx_B)
                )
        phase_diagram.save(f'{plot_dir}/PhaseDiagram.png', quality=100)

    def plot_skynumber(self):
        plot_dir = self.inputs.plot_dir.value
        fig, ax = plt.subplots()
        for idx_B, externel_B in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for idx_T, temperature in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx['T' + str(idx_T) + '_' + 'B' +
                             str(idx_B)].get_outgoing().get_node_by_label('sk_num_out').get_array('sk_number_data')[0]
                )
            ax.plot(self.inputs.temperatures.get_list(), sk_number_list, label=f'B: {externel_B[-6:]} T')
        ax.legend(fontsize='xx-small')
        ax.set_ylabel('Skyrmion number')
        ax.set_xlabel('Temperature')
        plt.savefig(f'{plot_dir}/sk_number.png')

    def plot_skynumber_avg(self):
        plot_dir = self.inputs.plot_dir.value
        fig, ax = plt.subplots()
        for idx_B, externel_B in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for idx_T, temperature in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx['T' + str(idx_T) + '_' + 'B' +
                             str(idx_B)].get_outgoing().get_node_by_label('sk_num_out').get_array('sk_number_data')[1]
                )
            ax.plot(
                self.inputs.temperatures.get_list(), sk_number_list, label=f'B: {externel_B[-6:]} T'
            )  #this should be changed with different B
        ax.legend(fontsize='xx-small')
        ax.set_ylabel('Skyrmion number')
        ax.set_xlabel('Temperature')
        plt.savefig(f'{plot_dir}/sk_number_avg.png')

    def plot_skynumber_number_heat_map(self):
        plot_dir = self.inputs.plot_dir.value
        heat_map = []
        for idx_B, externel_B in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for idx_T, temperature in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx['T' + str(idx_T) + '_' + 'B' +
                             str(idx_B)].get_outgoing().get_node_by_label('sk_num_out').get_array('sk_number_data')[0]
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

    def plot_skynumber_number_avg_heat_map(self):
        plot_dir = self.inputs.plot_dir.value
        heat_map = []
        for idx_B, externel_B in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for idx_T, temperature in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx['T' + str(idx_T) + '_' + 'B' +
                             str(idx_B)].get_outgoing().get_node_by_label('sk_num_out').get_array('sk_number_data')[1]
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
        plot_dir = self.inputs.plot_dir.value
        heat_map = []
        for idx_B, externel_B in enumerate(self.inputs.external_fields):
            average_magnetic_moment = []
            for idx_T, temperature in enumerate(self.inputs.temperatures):
                average_magnetic_moment.append(
                    self.ctx['T' + str(idx_T) + '_' + 'B' +
                             str(idx_B)].get_outgoing().get_node_by_label('cumulants')['magnetization']
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
        plot_dir = self.inputs.plot_dir.value

        if len(self.inputs.external_fields.get_list()) == 1:
            #plot the line for one B
            for idx_B, externel_B in enumerate(self.inputs.external_fields):
                energy_list = []
                tempture_list = self.inputs.temperatures.get_list()
                for idx_T, temperature in enumerate(self.inputs.temperatures):
                    energy_list.append(
                        self.ctx['T' + str(idx_T) + '_' + 'B' +
                                 str(idx_B)].get_outgoing().get_node_by_label('cumulants')['energy']
                    )
                de_dt = (np.gradient(np.array(energy_list)) / np.gradient(np.array(tempture_list)))
                de_dt = (de_dt / de_dt[0]).tolist()
            fig, ax = plt.subplots()
            ax.plot(self.inputs.temperatures.get_list(), de_dt)
            ax.legend(fontsize='xx-small')
            ax.set_ylabel('C')
            ax.set_xlabel('Temperature')
            plt.savefig(f'{plot_dir}/CV_T.png')

        else:
            heat_map = []
            for idx_B, externel_B in enumerate(self.inputs.external_fields):
                energy_list = []
                tempture_list = self.inputs.temperatures.get_list()

                for idx_T, temperature in enumerate(self.inputs.temperatures):
                    energy_list.append(
                        self.ctx['T' + str(idx_T) + '_' + 'B' +
                                 str(idx_B)].get_outgoing().get_node_by_label('cumulants')['energy']
                    )
                de_dt = (np.gradient(np.array(energy_list)) / np.gradient(np.array(tempture_list))).tolist()
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
