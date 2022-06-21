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
from PIL import Image
from random import sample, choices
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np



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


def plot_pd(plot_dir,heat_map,x_label_list,y_label_list,plot_name,xlabel,ylabel):
    fig = go.Figure(data=go.Heatmap(
                    z=heat_map,
                    x=x_label_list,
                    y=y_label_list,
                    zsmooth = 'best',
                    ))
    fig.update_xaxes(range=[int(float(x_label_list[0])), int(float(x_label_list[-1]))])
    fig.update_yaxes(range=[int(float(y_label_list[0])), int(float(y_label_list[-1]))])
    fig.update_traces(colorscale='Jet', selector=dict(type='heatmap'))
    fig.update_layout(
                  yaxis={"title": '{}'.format(xlabel),"tickangle": -90},
                  xaxis={"title": '{}'.format(ylabel)} )
    fig.write_image('{}/{}.png'.format(plot_dir,plot_name.replace(' ','_')))

def plot_line(x,y,line_name_list,x_label,y_label,plot_path,plot_name,leg_name_list):
    #Since we need all lines in one plot here x and y should be a dict with the line name on that
    plt.figure()
    fig, ax = plt.subplots()
    for i in range(len(line_name_list)):
        ax.plot(
            x,
            y[line_name_list[i]],
            label="{}".format(leg_name_list[i]),
        )
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig("{}/{}.png".format(plot_path,plot_name))
    plt.close()

class ThermalDynamicWorkflow(WorkChain):
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

        spec.input('external_fields', valid_type=List,
                   help='task dict for inpsd.dat', required=False)
        spec.input('cell_size', valid_type=List,
                   help='task dict for inpsd.dat', required=False)

        spec.input("plot_dir", valid_type=Str, help="plot dir ", required=False)
        
                           
        #plot control flag:
        #only if we have M_T_plot > 0 and len(B) == 1 then we will plot M_T otherwise we will give phase diagram or not plot
        spec.input('M_T_plot', valid_type=Int,
                   help='flag for plotting skyrmions', required=False,default=lambda:Int(1))  # 
        spec.input('M_phase_diagram', valid_type=Int,
                   help='flag for ', required=False,default=lambda:Int(1))  # 

        spec.input('susceptibility_T_plot', valid_type=Int,
                   help='flag for plotting skyrmions', required=False,default=lambda:Int(1))  # 
        spec.input('susceptibility_phase_diagram', valid_type=Int,
                   help='flag for ', required=False,default=lambda:Int(1))  # 


        spec.input('specific_heat_T_plot', valid_type=Int,
                   help='flag for plotting skyrmions', required=False,default=lambda:Int(1))  # 
        spec.input('specific_heat_phase_diagram', valid_type=Int,
                   help='flag for ', required=False,default=lambda:Int(1))  # 



        spec.input('energy_T_plot', valid_type=Int,
                   help='flag for plotting skyrmions', required=False,default=lambda:Int(1))  # 
        spec.input('energy_phase_diagram', valid_type=Int,
                   help='flag for ', required=False,default=lambda:Int(1))  # 


        spec.input('free_e_T_plot', valid_type=Int,
                   help='flag for plotting skyrmions', required=False,default=lambda:Int(1))  # 
        spec.input('free_e_phase_diagram', valid_type=Int,
                   help='flag for ', required=False,default=lambda:Int(1))  # 


        spec.input('entropy_T_plot', valid_type=Int,
                   help='flag for plotting skyrmions', required=False,default=lambda:Int(1))  # 
        spec.input('entropy_phase_diagram', valid_type=Int,
                   help='flag for ', required=False,default=lambda:Int(1))  # 


        spec.input('dudt_T_plot', valid_type=Int,
                   help='flag for plotting skyrmions', required=False,default=lambda:Int(1))  # 
        spec.input('dudt_phase_diagram', valid_type=Int,
                   help='flag for ', required=False,default=lambda:Int(1))  # 


        spec.input('binder_cumulant_T_plot', valid_type=Int,
                   help='flag for plotting skyrmions', required=False,default=lambda:Int(1))  # 
        spec.input('binder_cumulant_phase_diagram', valid_type=Int,
                   help='flag for ', required=False,default=lambda:Int(1))  # 
        
        spec.output('thermal_dynamic_output', valid_type=Dict, help='Result Dict for temperature')  
        spec.exit_code(701, "ThermalDynamic_T_error", message="IN TD CALC T LENGTH SHOULD LAGRER THAN 1")
        spec.outline(
                cls.load_tasks,
                cls.loop_temperatures,
                cls.results,
                if_(cls.check_M_phase_diagram)(
                    cls.plot_M_phase_diagram
                ).elif_(cls.check_M_T)(
                    cls.plot_M_T
                ).else_(
                    cls.error_report
                ),    
                if_(cls.check_susceptibility_phase_diagram)(
                    cls.plot_susceptibility_phase_diagram
                ).elif_(cls.check_susceptibility_T)(
                    cls.plot_susceptibility_T
                ).else_(
                    cls.error_report
                ),  

                if_(cls.check_specific_heat_phase_diagram)(
                    cls.plot_specific_heat_phase_diagram
                ).elif_(cls.check_specific_heat_T)(
                    cls.plot_specific_heat_T
                ).else_(
                    cls.error_report
                ),  


                if_(cls.check_free_e_phase_diagram)(
                    cls.plot_free_e_phase_diagram
                ).elif_(cls.check_free_e_T)(
                    cls.plot_free_e_T
                ).else_(
                    cls.error_report
                ),  


                if_(cls.check_entropy_phase_diagram)(
                    cls.plot_entropy_phase_diagram
                ).elif_(cls.check_entropy_T)(
                    cls.plot_entropy_T
                ).else_(
                    cls.error_report
                ),  


                if_(cls.check_energy_phase_diagram)(
                    cls.plot_energy_phase_diagram
                ).elif_(cls.check_energy_T)(
                    cls.plot_energy_T
                ).else_(
                    cls.error_report
                ),  

                if_(cls.check_dudt_phase_diagram)(
                    cls.plot_dudt_phase_diagram
                ).elif_(cls.check_dudt_T)(
                    cls.plot_dudt_T
                ).else_(
                    cls.error_report
                ),  




                if_(cls.check_binder_cumulant_phase_diagram)(
                    cls.plot_binder_cumulant_phase_diagram
                ).elif_(cls.check_binder_cumulant_T)(
                    cls.plot_binder_cumulant_T
                ).else_(
                    cls.error_report
                ),  


                )
    def error_report(self):
        return self.exit_codes.ThermalDynamic_T_error

    def check_M_phase_diagram(self):
        if (self.inputs.M_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) > 1 :
            return True
        else:
            return False 
    def plot_M_phase_diagram(self):
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for idx,cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['magnetization'])
            plot_pd(self.inputs.plot_dir.value,pd_for_plot,self.inputs.temperatures.get_list(),y_label_list,('M_T'+str(cell_size)),'B','T')

    
    def check_M_T(self):
        if (self.inputs.M_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) == 1:
            return True
        else:
            return False 

    def plot_M_T(self):
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for idx,cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            leg_name_list.append(('Cell:'+cell_size.replace(' ','_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot['{}_{}'.format(cell_size.replace(' ','_'),index)] = line_dict[i]['magnetization']
                line_name_list.append('{}_{}'.format(cell_size.replace(' ','_'),index))
                index = index+1
        plot_line(self.inputs.temperatures.get_list(),line_for_plot,line_name_list,'T','M',self.inputs.plot_dir.value,'M_T',leg_name_list)


        

        #specific_heat
    def check_specific_heat_phase_diagram(self):
        if (self.inputs.specific_heat_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) > 1 :
            return True
        else:
            return False 
    def plot_specific_heat_phase_diagram(self):
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for idx,cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['specific_heat'])
            plot_pd(self.inputs.plot_dir.value,pd_for_plot,self.inputs.temperatures.get_list(),y_label_list,('specific_heat_T'+str(cell_size)),'B','T')

    
    def check_specific_heat_T(self):
        if (self.inputs.specific_heat_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) == 1:
            return True
        else:
            return False 

    def plot_specific_heat_T(self):
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for idx,cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            leg_name_list.append(('Cell:'+cell_size.replace(' ','_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot['{}_{}'.format(cell_size.replace(' ','_'),index)] = line_dict[i]['specific_heat']
                line_name_list.append('{}_{}'.format(cell_size.replace(' ','_'),index))
                index = index+1
        plot_line(self.inputs.temperatures.get_list(),line_for_plot,line_name_list,'T','specific_heat',self.inputs.plot_dir.value,'specific_heat_T',leg_name_list)

    

        #susceptibility
    def check_susceptibility_phase_diagram(self):
        if (self.inputs.susceptibility_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) > 1 :
            return True
        else:
            return False 
    def plot_susceptibility_phase_diagram(self):
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for idx,cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['susceptibility'])
            plot_pd(self.inputs.plot_dir.value,pd_for_plot,self.inputs.temperatures.get_list(),y_label_list,('susceptibility_T'+str(cell_size)),'B','T')

    
    def check_susceptibility_T(self):
        if (self.inputs.susceptibility_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) == 1:
            return True
        else:
            return False 

    def plot_susceptibility_T(self):
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for idx,cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            leg_name_list.append(('Cell:'+cell_size.replace(' ','_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot['{}_{}'.format(cell_size.replace(' ','_'),index)] = line_dict[i]['susceptibility']
                line_name_list.append('{}_{}'.format(cell_size.replace(' ','_'),index))
                index = index+1
        plot_line(self.inputs.temperatures.get_list(),line_for_plot,line_name_list,'T','Susceptibility',self.inputs.plot_dir.value,'Susceptibility_T',leg_name_list)








        #free_e
    def check_free_e_phase_diagram(self):
        if (self.inputs.free_e_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) > 1 :
            return True
        else:
            return False 
    def plot_free_e_phase_diagram(self):
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for idx,cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['free_e'])
            plot_pd(self.inputs.plot_dir.value,pd_for_plot,self.inputs.temperatures.get_list(),y_label_list,('free_e_T'+str(cell_size)),'B','T')

    
    def check_free_e_T(self):
        if (self.inputs.free_e_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) == 1:
            return True
        else:
            return False 

    def plot_free_e_T(self):
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for idx,cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            leg_name_list.append(('Cell:'+cell_size.replace(' ','_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot['{}_{}'.format(cell_size.replace(' ','_'),index)] = line_dict[i]['free_e']
                line_name_list.append('{}_{}'.format(cell_size.replace(' ','_'),index))
                index = index+1
        plot_line(self.inputs.temperatures.get_list(),line_for_plot,line_name_list,'T','free_e',self.inputs.plot_dir.value,'free_e_T',leg_name_list)





        #entropy
    def check_entropy_phase_diagram(self):
        if (self.inputs.entropy_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) > 1 :
            return True
        else:
            return False 
    def plot_entropy_phase_diagram(self):
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for idx,cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['entropy'])
            plot_pd(self.inputs.plot_dir.value,pd_for_plot,self.inputs.temperatures.get_list(),y_label_list,('entropy_T'+str(cell_size)),'B','T')

    
    def check_entropy_T(self):
        if (self.inputs.entropy_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) == 1:
            return True
        else:
            return False 

    def plot_entropy_T(self):
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for idx,cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            leg_name_list.append(('Cell:'+cell_size.replace(' ','_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot['{}_{}'.format(cell_size.replace(' ','_'),index)] = line_dict[i]['entropy']
                line_name_list.append('{}_{}'.format(cell_size.replace(' ','_'),index))
                index = index+1
        plot_line(self.inputs.temperatures.get_list(),line_for_plot,line_name_list,'T','Entropy',self.inputs.plot_dir.value,'Entropy_T',leg_name_list)




        #energy
    def check_energy_phase_diagram(self):
        if (self.inputs.energy_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) > 1 :
            return True
        else:
            return False 
    def plot_energy_phase_diagram(self):
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for idx,cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['energy'])
            plot_pd(self.inputs.plot_dir.value,pd_for_plot,self.inputs.temperatures.get_list(),y_label_list,('energy_T'+str(cell_size)),'B','T')

    
    def check_energy_T(self):
        if (self.inputs.energy_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) == 1:
            return True
        else:
            return False 

    def plot_energy_T(self):
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for idx,cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            leg_name_list.append(('Cell:'+cell_size.replace(' ','_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot['{}_{}'.format(cell_size.replace(' ','_'),index)] = line_dict[i]['energy']
                line_name_list.append('{}_{}'.format(cell_size.replace(' ','_'),index))
                index = index+1
        plot_line(self.inputs.temperatures.get_list(),line_for_plot,line_name_list,'T','energy',self.inputs.plot_dir.value,'energy_T',leg_name_list)





        #dudt
    def check_dudt_phase_diagram(self):
        if (self.inputs.dudt_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) > 1 :
            return True
        else:
            return False 
    def plot_dudt_phase_diagram(self):
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for idx,cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['dudt'])
            plot_pd(self.inputs.plot_dir.value,pd_for_plot,self.inputs.temperatures.get_list(),y_label_list,('dudt_T'+str(cell_size)),'B','T')

    
    def check_dudt_T(self):
        if (self.inputs.dudt_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) == 1:
            return True
        else:
            return False 

    def plot_dudt_T(self):
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for idx,cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            leg_name_list.append(('Cell:'+cell_size.replace(' ','_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot['{}_{}'.format(cell_size.replace(' ','_'),index)] = line_dict[i]['dudt']
                line_name_list.append('{}_{}'.format(cell_size.replace(' ','_'),index))
                index = index+1
        plot_line(self.inputs.temperatures.get_list(),line_for_plot,line_name_list,'T','dudt',self.inputs.plot_dir.value,'dudt_T',leg_name_list)





        #binder_cumulant
    def check_binder_cumulant_phase_diagram(self):
        if (self.inputs.binder_cumulant_phase_diagram.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) > 1 :
            return True
        else:
            return False 
    def plot_binder_cumulant_phase_diagram(self):
        y_label_list = []
        for i in self.inputs.external_fields.get_list():
            y_label_list.append(float(max(np.array(i.split()))))

        for idx,cell_size in enumerate(self.inputs.cell_size):
            pd_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            pd_for_plot = []
            for i in pd_dict.keys():
                pd_for_plot.append(pd_dict[i]['binder_cumulant'])
            plot_pd(self.inputs.plot_dir.value,pd_for_plot,self.inputs.temperatures.get_list(),y_label_list,('binder_cumulant_T'+str(cell_size)),'B','T')

    
    def check_binder_cumulant_T(self):
        if (self.inputs.binder_cumulant_T_plot.value > int(0) and len(self.inputs.temperatures.get_list()) > 1 and len(self.inputs.external_fields.get_list())) == 1:
            return True
        else:
            return False 

    def plot_binder_cumulant_T(self):
        line_name_list = []
        leg_name_list = []
        line_for_plot = {}
        for idx,cell_size in enumerate(self.inputs.cell_size):
            line_dict = self.ctx.TD_dict['{}'.format(cell_size)]
            leg_name_list.append(('Cell:'+cell_size.replace(' ','_')))
            index = 0
            for i in line_dict.keys():
                line_for_plot['{}_{}'.format(cell_size.replace(' ','_'),index)] = line_dict[i]['binder_cumulant']
                line_name_list.append('{}_{}'.format(cell_size.replace(' ','_'),index))
                index = index+1
        plot_line(self.inputs.temperatures.get_list(),line_for_plot,line_name_list,'T','binder_cumulant',self.inputs.plot_dir.value,'binder_cumulant_T',leg_name_list)











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
        inputs = self.exposed_inputs(ASDBaseRestartWorkChain)#we need take input dict from the BaseRestartWorkchain, not a new one.
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
        for idx,cell_size in enumerate(self.inputs.cell_size):
            for idx_1, eB in enumerate(self.inputs.external_fields):
                for idx_2, temperature in enumerate(self.inputs.temperatures):
                    #self.report('Running loop for temperature with value {} and B with value {} and cell size{}'.format(temperature,eB,cell_size))
                    self.inputs.inpsd_dict['temp'] = temperature
                    self.inputs.inpsd_dict['ip_temp'] = temperature
                    self.inputs.inpsd_dict['ip_hfield'] = eB
                    self.inputs.inpsd_dict['ncell'] = cell_size
                    inputs=self.generate_inputs()
                    future = self.submit(ASDBaseRestartWorkChain, **inputs)
                    calculations['C'+str(idx)+'B'+str(idx_1)+'T'+str(idx_2)] = future
                    self.report('{} is run'.format(('C'+str(idx)+'B'+str(idx_1)+'T'+str(idx_2))))
        return ToContext(**calculations)


    def results(self):
        """Process results."""
        TD_out = {}
        for idx,cell_size in enumerate(self.inputs.cell_size):
            TD_out['{}'.format(cell_size)]={}
            for idx_1, eB in enumerate(self.inputs.external_fields):
                outputs = {}
                for idx_2, temperature in enumerate(self.inputs.temperatures):
                    outputs['C'+str(idx)+'B'+str(idx_1)+'T'+str(idx_2)]= self.ctx['C'+str(idx)+'B'+str(idx_1)+'T'+str(idx_2)].get_outgoing().get_node_by_label('cumulants')
                temperature_output = get_temperature_data(**outputs)
                TD_out['{}'.format(cell_size)]['{}'.format(eB)] = temperature_output.get_dict()
        self.ctx.TD_dict = TD_out
        outdict = Dict(dict=TD_out).store()
        self.out('thermal_dynamic_output',outdict)

