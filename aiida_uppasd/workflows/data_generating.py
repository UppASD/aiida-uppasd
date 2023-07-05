#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:03:40 2022

@author Qichen Xu
Workchain demo generating data for Machine Learning
"""

import numpy as np

from aiida.engine import ToContext, WorkChain, calcfunction, if_
from aiida.orm import ArrayData, Dict, Int, List, Str

from aiida_uppasd.workflows.base_restart import ASDBaseRestartWorkChain


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
            if_(cls.check_for_plot_sk)(cls.results_and_plot,),
        )

    def check_for_plot_sk(self):
        """Check if the skyrmion should plotted"""
        try:
            if self.inputs.sk_plot.value > int(0):
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

                sk_data = self.ctx[f'T{index_temperature}_B{index_field}'].get_outgoing(
                ).get_node_by_label('sk_num_out')
                sm_list.append(sk_data.get_array('sk_number_data')[0])
                sam_list.append(sk_data.get_array('sk_number_data')[1])

            sky_num_out.append(sm_list)
            sky_avg_num_out.append(sam_list)

        sk_num_for_plot = store_sknum_data(List(list=sky_num_out), List(list=sky_avg_num_out))

        self.out('sk_num_for_plot', sk_num_for_plot)
