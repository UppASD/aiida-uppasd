# -*- coding: utf-8 -*-
"""
This demo we only need temperature list and N(cell size)
@author Qichen Xu
Workchain demo for plot M(T) with different cell setting
"""
import os

from aiida import load_profile, orm
from aiida.engine import WorkChain

from aiida_uppasd.workflows.temperature_restart import UppASDTemperatureRestartWorkflow

load_profile()


class MCVariableCellWorkchain(WorkChain):
    """This demo we only need temperature list and N(cell size)

    list as extra controllable input"""

    @classmethod
    def define(cls, spec):
        """This demo we only need temperature list and N(cell size)

        list as extra controllable input"""
        super().define(spec)
        spec.input(
            'temp_list',
            valid_type=orm.List,
            help='temp list',
            required=False,
        )
        spec.input(
            'N_list',
            valid_type=orm.List,
            help='N list ',
            required=False,
        )
        spec.input(
            'plot_dir',
            valid_type=orm.Str,
            help='plot dir ',
            required=False,
        )
        spec.outline(cls.submit_workchains,)

    def submit_workchains(self):
        """submit UppASDTemperatureRestartWorkflow with input"""
        ncell_list = self.inputs.N_list
        temp_list = self.inputs.temp_list
        for cell_size in ncell_list:
            input_uppasd = {
                'inpsd_temp':
                orm.Dict(
                    dict={
                        'simid':
                        orm.Str('bccFe100'),
                        'ncell':
                        orm.Str(f'{cell_size} {cell_size} {cell_size}'),
                        'BC':
                        orm.Str('P         P         P '),
                        'cell':
                        orm.Str(
                            """1.00000 0.00000 0.00000
                        0.00000 1.00000 0.00000
                        0.00000 0.00000 1.00000"""
                        ),
                        'sym':
                        orm.Int(1),
                        'maptype':
                        orm.Int(1),
                        'Initmag':
                        orm.Int(3),
                        'alat':
                        orm.Float(2.87e-10),
                    }
                ),
                'num_machines':
                orm.Int(1),
                'num_mpiprocs_per_machine':
                orm.Int(1),
                'max_wallclock_seconds':
                orm.Int(1800),
                #'code' :Code.get_from_string('uppasd_dev@uppasd_local'),
                'code':
                orm.Code.get_from_string('uppasd_nsc_2021_test@nsc_uppasd_2021'),
                'input_filename':
                orm.Str('inpsd.dat'),
                'parser_name':
                orm.Str('uppasd.uppasd_parser'),
                'label':
                orm.Str('uppasd_base_workflow_demo'),
                'description':
                orm.Str('Test base workflow'),
                'prepared_file_folder':
                orm.Str(os.path.join(os.getcwd(), 'task1_input')),
                'except_filenames':
                orm.List(list=[]),
                'retrieve_list_name':
                orm.List(list=[('*.out', '.', 0), ('*.json', '.', 0)]),
                'tasks':
                orm.List(list=['mc', 'thermodynamics']),
                'temperatures':
                temp_list,
            }

            future = self.submit(UppASDTemperatureRestartWorkflow, **input_uppasd)
            key = f'workchain_with_N_{cell_size}'
            self.to_context(**{key: future})
