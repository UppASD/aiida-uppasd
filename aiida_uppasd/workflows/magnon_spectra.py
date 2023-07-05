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
        spec.output(
            'ams_plot_var',
            valid_type=orm.Dict,
            required=True,
            help='Adiabatic magnon spectrum plotting variables',
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

    def results(self):  # pylint: disable=too-many-locals
        """Process results and basic plot"""
        ams_plot_var_out = self.ctx['AMS'].get_outgoing().get_node_by_label('AMS_plot_var')
        ams = self.ctx['AMS'].get_outgoing().get_node_by_label('ams')
        self.out('ams', ams)
        self.out('ams_plot_var', ams_plot_var_out)
