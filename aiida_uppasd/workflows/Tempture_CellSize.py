# -*- coding: utf-8 -*-
from aiida import orm
import aiida
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import (
    run,
    submit,
    ToContext,
    if_,
    while_,
    BaseRestartWorkChain,
    process_handler,
    ProcessHandlerReport,
    ExitCode,
)
from aiida.plugins import CalculationFactory, GroupFactory
from aiida.orm import (
    Code,
    SinglefileData,
    Int,
    Float,
    Str,
    Bool,
    List,
    Dict,
    ArrayData,
    XyData,
    SinglefileData,
    FolderData,
    RemoteData,
)
from aiida_uppasd.workflows.temperature_restart import UppASDTemperatureRestartWorkflow
import os
from aiida.engine import WorkChain

aiida.load_profile()
from aiida.orm.nodes import WorkChainNode
from aiida.orm import QueryBuilder, Dict
import matplotlib.pyplot as plt
import numpy as np
import json

"""
@author Qichen Xu 
Workchain demo for plot M(T) with different cell setting
"""


def cal_node_query(workchain_pk, attribute_name):
    """Simple query function that helps find output dict"""
    qb = QueryBuilder()
    qb.append(WorkChainNode, filters={"id": str(workchain_pk)}, tag="workflow_node")
    qb.append(Dict, with_incoming="workflow_node", tag="workdict")

    return qb.all()


class N_workchain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        """This demo we only need temperature list and N(cell size) list as extra controllable input"""
        spec.input("temp_list", valid_type=List, help="temp list", required=False)
        spec.input("N_list", valid_type=List, help="N list ", required=False)
        spec.input("plot_dir", valid_type=Str, help="plot dir ", required=False)
        spec.outline(
            cls.submit_workchains,
            cls.plot_result,
        )

    def submit_workchains(self):
        """submit UppASDTemperatureRestartWorkflow with input"""
        N_list = self.inputs.N_list
        temp_list = self.inputs.temp_list
        for i in N_list:
            input_uppasd = {
                "inpsd_temp": orm.Dict(
                    dict={
                        "simid": Str("bccFe100"),
                        "ncell": Str("{} {} {}".format(i, i, i)),
                        "BC": Str("P         P         P "),
                        "cell": Str(
                            """1.00000 0.00000 0.00000
                        0.00000 1.00000 0.00000
                        0.00000 0.00000 1.00000"""
                        ),
                        "sym": Int(1),
                        "maptype": Int(1),
                        "Initmag": Int(3),
                        "alat": Float(2.87e-10),
                    }
                ),
                "num_machines": orm.Int(1),
                "num_mpiprocs_per_machine": orm.Int(1),
                "max_wallclock_seconds": orm.Int(1800),
                #'code' :Code.get_from_string('uppasd_dev@uppasd_local'),
                "code": Code.get_from_string("uppasd_nsc_2021_test@nsc_uppasd_2021"),
                "input_filename": orm.Str("inpsd.dat"),
                "parser_name": orm.Str("UppASD_core_parsers"),
                "label": orm.Str("uppasd_base_workflow_demo"),
                "description": orm.Str("Test base workflow"),
                "prepared_file_folder": Str(os.path.join(os.getcwd(), "task1_input")),
                "except_filenames": List(list=[]),
                "retrieve_list_name": List(
                    list=[("*.out", ".", 0), ("*.json", ".", 0)]
                ),
                "tasks": List(list=["mc", "thermodynamics"]),
                "temperatures": temp_list,
                #'temperatures' : List(list=[ 0.001, 100,200,300,400,500,600,700,800,850,900,950,1000,1100,1200,1300,1400,1500])
            }

            future = self.submit(UppASDTemperatureRestartWorkflow, **input_uppasd)
            key = f"workchain_with_N_{i}"
            self.to_context(**{key: future})

    def plot_result(self):
        """
        Basic plot function
        Note that we use normalized axis like T/Tc in all figures
        """
        plot_path = self.inputs.plot_dir.value
        N_list = self.inputs.N_list
        temp_list = self.inputs.temp_list
        T_C = 1043  # BCC Fe Curie temperature

        plt.figure()
        fig, ax = plt.subplots()
        for i in N_list:
            key = f"workchain_with_N_{i}"
            sub_workchain_node = self.ctx[key]
            sub_workchain_pk = sub_workchain_node.pk
            self.report("sub_workchain {}".format(sub_workchain_pk))
            data = cal_node_query(sub_workchain_pk, "temperature_output")[0][
                0
            ].get_dict()
            ax.plot(
                np.array(data["temperature"]) / T_C,
                np.array(data["magnetization"]) / np.array(data["magnetization"][0]),
                label="N={}".format(i),
            )
        ax.legend()
        ax.set_xlabel("T/Tc")
        ax.set_ylabel("M/Mc")
        # plt.title('Temperature-magnetization')
        plt.savefig("{}/temperature-magnetization.png".format(plot_path))
        plt.close()

        plt.figure()
        fig, ax = plt.subplots()
        for i in N_list:
            key = f"workchain_with_N_{i}"
            sub_workchain_node = self.ctx[key]
            sub_workchain_pk = sub_workchain_node.pk
            self.report("sub_workchain {}".format(sub_workchain_pk))
            data = cal_node_query(sub_workchain_pk, "temperature_output")[0][
                0
            ].get_dict()
            ax.plot(
                np.array(data["temperature"]) / T_C,
                data["energy"],
                label="N={}".format(i),
            )
        ax.legend()
        ax.set_xlabel("T/Tc")
        ax.set_ylabel("Energy(mRy)")
        # plt.title('temperature-energy')
        plt.savefig("{}/temperature-energy.png".format(plot_path))
        plt.close()

        return ExitCode(0)
