# -*- coding: utf-8 -*-
"""
This demo we only need temperature list and N(cell size)
@author Qichen Xu
Workchain demo for plot M(T) with different cell setting
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from aiida import orm, load_profile
from aiida.engine import ExitCode, WorkChain
from aiida_uppasd.workflows.temperature_restart import UppASDTemperatureRestartWorkflow

load_profile()


def cal_node_query(workchain_pk):
    """Simple query function that helps find output dict"""
    qb = orm.QueryBuilder()
    qb.append(
        orm.WorkChainNode,
        filters={'id': str(workchain_pk)},
        tag='workflow_node',
    )
    qb.append(orm.Dict, with_incoming='workflow_node', tag='workdict')

    return qb.all()


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
        spec.outline(
            cls.submit_workchains,
            cls.plot_result,
        )

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

    def plot_result(self):
        """
        Basic plot function
        Note that we use normalized axis like T/Tc in all figures
        """
        plot_path = self.inputs.plot_dir.value
        ncell_list = self.inputs.N_list
        _ = self.inputs.temp_list
        curie_temperature = 1043  # BCC Fe Curie temperature

        plt.figure()
        _, axes = plt.subplots()
        for i in ncell_list:
            key = f'workchain_with_N_{i}'
            sub_workchain_node = self.ctx[key]
            sub_workchain_pk = sub_workchain_node.pk
            self.report(f'sub_workchain {sub_workchain_pk}')
            data = cal_node_query(sub_workchain_pk)[0][0].get_dict()
            axes.plot(
                np.array(data['temperature']) / curie_temperature,
                np.array(data['magnetization']) / np.array(data['magnetization'][0]),
                label=f'N={i}',
            )
        axes.legend()
        axes.set_xlabel('T/Tc')
        axes.set_ylabel('M/Mc')
        # plt.title('Temperature-magnetization')
        plt.savefig(f'{plot_path}/temperature-magnetization.png')
        plt.close()

        plt.figure()
        _, axes = plt.subplots()
        for i in ncell_list:
            key = f'workchain_with_N_{i}'
            sub_workchain_node = self.ctx[key]
            sub_workchain_pk = sub_workchain_node.pk
            self.report(f'sub_workchain {sub_workchain_pk}')
            data = cal_node_query(sub_workchain_pk)[0][0].get_dict()
            axes.plot(
                np.array(data['temperature']) / curie_temperature,
                data['energy'],
                label=f'N={i}',
            )
        axes.legend()
        axes.set_xlabel('T/Tc')
        axes.set_ylabel('Energy(mRy)')
        # plt.title('temperature-energy')
        plt.savefig(f'{plot_path}/temperature-energy.png')
        plt.close()

        return ExitCode(0)
