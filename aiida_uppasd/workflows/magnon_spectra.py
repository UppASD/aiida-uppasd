#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:03:40 2022

@author Qichen Xu
Workchain demo for plot magnon spectra

In this workchain demo we want to use the same method like baserestartworkchain.
This workchain only takes output and do the magnon spectra plot task

For activate J1 J2 J3 .... mode(the nearest model, next nearest mode etc.)
We add one validation function here to check if people want to use this or
offer exchange file directly

several functions are base on codes from UppASD repo like preQ.py and postQ.py
which @Anders bergman
"""

import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_uppasd.workflows.base_restart import ASDBaseRestartWorkChain


class UppASDMagnonSpectraRestartWorkflow(WorkChain):
    """Base workchain first
    #Workchain to run an UppASD simulation with automated error handling and restarts.#"""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseRestartWorkChain)
        spec.expose_outputs(ASDBaseRestartWorkChain)

        spec.input(
            'J_model',
            valid_type=orm.Int,
            help='flag for choose nearest Jij depth',
            required=False,
        )
        spec.input(
            'exchange_ams',
            valid_type=orm.Dict,
            help='Dict to store Jij ',
            required=False,
        )

        spec.input(
            'plot_dir',
            valid_type=orm.Str,
            help='plot dir ',
            required=False,
        )

        spec.input(
            'inpsd_ams',
            valid_type=orm.Dict,
            help='temp dict of inpsd.dat',
            required=False,
        )
        spec.output(
            'ams',
            valid_type=orm.BandsData,
            required=True,
            help='Adiabatic magnon spectrum',
        )

        spec.outline(
            cls.submits,
            cls.results_and_plot,
        )

    def gen_inputs_validate_exch_model(self):
        """
        Generate the inputs needed for the workchain

        :return: the inputs to be used in the workchains
        :rtype: AttributeDict
        """
        inputs = self.exposed_inputs(
            ASDBaseRestartWorkChain
        )  #we need take input dict from the BaseRestartWorkchain, not a new one.
        inputs.code = self.inputs.code
        inputs.AMSplot = self.inputs.AMSplot
        inputs.inpsd_dict = self.inputs.inpsd_ams
        inputs.prepared_file_folder = self.inputs.prepared_file_folder
        inputs.except_filenames = self.inputs.except_filenames

        if self.inputs.J_model.value == -1:
            pass
        elif self.inputs.J_model.value == 0:
            inputs.exchange = self.inputs.exchange_ams
            self.report('Running AMS with all possible Jij')
        else:
            exchange_temp = {}
            for j_idx in range(self.inputs.J_model.value):
                exchange_temp.update(self.inputs.exchange_ams[str(j_idx + 1)])
            inputs.exchange = orm.Dict(dict=exchange_temp)
            self.report(f'Running AMS with J model {self.inputs.J_model.value}')
        inputs.retrieve_list_name = self.inputs.retrieve_list_name
        return inputs

    def submits(self):
        """
        Submit the calculations needed
        """
        calculations = {}
        inputs = self.gen_inputs_validate_exch_model()
        future = self.submit(ASDBaseRestartWorkChain, **inputs)
        #modify here to make a loop if needed
        calculations['AMS'] = future
        return ToContext(**calculations)

    def results_and_plot(self):  # pylint: disable=too-many-locals
        """Process results and basic plot"""
        ams_plot_var_out = self.ctx['AMS'].get_outgoing().get_node_by_label('AMS_plot_var')
        timestep = ams_plot_var_out.get_dict()['timestep']
        sc_step = ams_plot_var_out.get_dict()['sc_step']
        sqw_x = ams_plot_var_out.get_dict()['sqw_x']
        sqw_y = ams_plot_var_out.get_dict()['sqw_y']
        ams_t = ams_plot_var_out.get_dict()['ams']
        axidx_abs = ams_plot_var_out.get_dict()['axidx_abs']
        ams_dist_col = ams_plot_var_out.get_dict()['ams_dist_col']
        axlab = ams_plot_var_out.get_dict()['axlab']
        plot_dir = self.inputs.plot_dir.value
        _ = plt.figure(figsize=[8, 5])
        axes = plt.subplot(111)
        hbar = float(4.135667662e-15)
        emax = 0.5 * float(hbar) / (float(timestep) * float(sc_step)) * 1e3
        #print('emax_sqw=',emax)
        sqw_x = np.array(sqw_x)
        sqw_y = np.array(sqw_y)
        #ams = np.array(ams)
        sqw_temp = (sqw_x**2.0 + sqw_y**2.0)**0.5
        sqw_temp[:, 0] = sqw_temp[:, 0] / 100.0
        sqw_temp = sqw_temp.T / sqw_temp.T.max(axis=0)
        sqw_temp = ndimage.gaussian_filter1d(
            sqw_temp,
            sigma=1,
            axis=1,
            mode='constant',
        )
        sqw_temp = ndimage.gaussian_filter1d(
            sqw_temp,
            sigma=5,
            axis=0,
            mode='reflect',
        )
        plt.imshow(
            sqw_temp,
            cmap=plt.get_cmap('jet'),
            interpolation='antialiased',
            origin='lower',
            extent=[axidx_abs[0], axidx_abs[-1], 0, emax],
        )  # Note here, we could choose color bar here.
        ams_t = np.array(ams_t)
        plt.plot(
            ams_t[:, 0] / ams_t[-1, 0] * axidx_abs[-1],
            ams_t[:, 1:ams_dist_col],
            'white',
            lw=1,
        )
        plt.colorbar()
        plt.xticks(axidx_abs, axlab)
        plt.xlabel('q')
        plt.ylabel('Energy (meV)')
        plt.autoscale(tight=False)
        axes.set_aspect('auto')
        plt.grid(b=True, which='major', axis='x')
        plt.savefig(f'{plot_dir}/AMS.png')
        ams = self.ctx['AMS'].get_outgoing().get_node_by_label('ams')
        self.out('ams', ams)
