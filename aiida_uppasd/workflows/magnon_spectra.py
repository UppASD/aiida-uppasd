#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:03:40 2022

@author Qichen Xu 
Workchain demo for plot magnon spectra

In this workchain demo we want to use the same method like baserestartworkchain.
This workchain only takes output and do the magnon spectra plot task

For activate J1 J2 J3 .... mode(the nearest model, next nearest mode etc.) We add one validation function here to check if people want to use this or offer exchange file directly

several functions are base on codes from UppASD repo like preQ.py and postQ.py which @Anders bergman
"""


# -*- coding: utf-8 -*-
import imp
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import ToContext, if_, while_, WorkChain,BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode, calcfunction
from aiida.plugins import CalculationFactory, GroupFactory
from aiida.orm import Code, SinglefileData, BandsData,Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
from aiida_uppasd.workflows.base_restart import ASDBaseRestartWorkChain
from numpy import gradient,asarray,insert
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
import json
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np

class UppASDMagnonSpectraRestartWorkflow(WorkChain):
    """Base workchain first
    #Workchain to run an UppASD simulation with automated error handling and restarts.#"""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseRestartWorkChain)
        spec.expose_outputs(ASDBaseRestartWorkChain)
        
        spec.input('J_model', valid_type=Int,
                   help='flag for choose nearest Jij deepth', required=False)
        spec.input('exchange_ams', valid_type=Dict,
                   help='Dict to store Jij ', required=False)
        
        spec.input("plot_dir", valid_type=Str, help="plot dir ", required=False)
        
        spec.input('inpsd_ams', valid_type=Dict,
                   help='temp dict of inpsd.dat', required=False)  # default=lambda: Dict(dict={})
        spec.output("ams",valid_type=BandsData, required=True,
                    help='Adiabatic magnon spectrum')

        spec.outline(
                cls.submits,
                cls.results_and_plot,
                )


    def generate_inputs_and_validation_J_model(self):
        inputs = self.exposed_inputs(ASDBaseRestartWorkChain)#we need take input dict from the BaseRestartWorkchain, not a new one.
        inputs.code = self.inputs.code
        inputs.AMSplot = self.inputs.AMSplot
        inputs.inpsd_dict =self.inputs.inpsd_ams
        inputs.prepared_file_folder = self.inputs.prepared_file_folder
        inputs.except_filenames = self.inputs.except_filenames
        #inputs.AMSplot = self.inputs.AMSplot
        #validation J model 
        #J_model should be Int(n) or Int(0)  n is a int which means 
        #Here Int(0) means calcuate all nearest

        #exchange_ams should be a dict that includes '1', '2'  '3' ..... 'n'   and  '-1' means use externel exchange

        if self.inputs.J_model.value == -1:
            pass
        elif self.inputs.J_model.value == 0:
            inputs.exchange =  self.inputs.exchange_ams
            self.report('Running AMS with all possible Jij')
        else:
            exchange_temp = {}
            for j_idx in range(self.inputs.J_model.value):
                exchange_temp.update(self.inputs.exchange_ams[str(j_idx+1)])
            inputs.exchange  = Dict(dict=exchange_temp)
            self.report('Running AMS with J model {}'.format(self.inputs.J_model.value))
        inputs.retrieve_list_name = self.inputs.retrieve_list_name
        return inputs


    def submits(self):
        calculations = {}
        inputs=self.generate_inputs_and_validation_J_model()
        future = self.submit(ASDBaseRestartWorkChain, **inputs)
        #modify here to make a loop if needed
        calculations['AMS'] = future
        return ToContext(**calculations)


    def results_and_plot(self):
        """Process results and basic plot"""
        AMS_plot_var_out = self.ctx['AMS'].get_outgoing().get_node_by_label('AMS_plot_var')
        timestep=AMS_plot_var_out.get_dict()['timestep']
        sc_step=AMS_plot_var_out.get_dict()['sc_step']
        sqw_x=AMS_plot_var_out.get_dict()['sqw_x']
        sqw_y=AMS_plot_var_out.get_dict()['sqw_y']
        ams_t=AMS_plot_var_out.get_dict()['ams']
        axidx_abs=AMS_plot_var_out.get_dict()['axidx_abs']
        ams_dist_col=AMS_plot_var_out.get_dict()['ams_dist_col']
        axlab=AMS_plot_var_out.get_dict()['axlab']
        plot_dir =  self.inputs.plot_dir.value
        fig = plt.figure(figsize=[8,5])
        ax=plt.subplot(111)
        hbar=float(4.135667662e-15)
        emax=0.5*float(hbar)/(float(timestep)*float(sc_step))*1e3
        #print('emax_sqw=',emax)
        sqw_x = np.array(sqw_x)
        sqw_y = np.array(sqw_y)
        #ams = np.array(ams)
        sqw_temp=(sqw_x**2.0+sqw_y**2.0)**0.5
        sqw_temp[:,0]=sqw_temp[:,0]/100.0
        sqw_temp=sqw_temp.T/sqw_temp.T.max(axis=0)
        sqw_temp=ndimage.gaussian_filter1d(sqw_temp,sigma=1,axis=1,mode='constant')
        sqw_temp=ndimage.gaussian_filter1d(sqw_temp,sigma=5,axis=0,mode='reflect')
        plt.imshow(sqw_temp, cmap=plt.get_cmap('jet'),interpolation='antialiased',origin='lower',extent=[axidx_abs[0],axidx_abs[-1],0,emax])# Note here, we could choose color bar here.
        ams_t = np.array(ams_t)
        plt.plot(ams_t[:,0]/ams_t[-1,0]*axidx_abs[-1],ams_t[:,1:ams_dist_col],'white',lw=1)
        plt.colorbar()
        plt.xticks(axidx_abs,axlab)
        plt.xlabel('q')
        plt.ylabel('Energy (meV)')
        plt.autoscale(tight=False)
        ax.set_aspect('auto')
        plt.grid(b=True,which='major',axis='x')
        plt.savefig("{}/AMS.png".format(plot_dir))
        ams = self.ctx['AMS'].get_outgoing().get_node_by_label('ams')
        self.out('ams',ams)
        return None
        
