# -*- coding: utf-8 -*-
"""Temperature sweep test"""
import matplotlib.pyplot as plt

import aiida
from aiida.orm import Dict, QueryBuilder
from aiida.orm.nodes import WorkChainNode

aiida.load_profile()


def cal_node_query(workchain_pk):
    """Query node"""

    q_build = QueryBuilder()
    q_build.append(WorkChainNode, filters={'id': str(workchain_pk)}, tag='workflow_node')
    q_build.append(Dict, with_incoming='workflow_node', tag='workdict')

    return q_build.all()


with open('UppASDTemperatureWorkflow_jobPK.csv', 'r', encoding='utf-8') as f:
    workchain_node = f.read()
try:
    data = cal_node_query(workchain_node)[0][0].get_dict()
    plt.figure()
    plt.plot(data['temperature'], data['magnetization'], 'x-')
    plt.title('temperature-magnetization')
    plt.savefig('temperature-magnetization.png')
    plt.close()

    plt.figure()
    plt.plot(data['temperature'], data['energy'], 'x-')
    plt.title('temperature-energy')
    plt.savefig('temperature-energy.png')
    plt.close()
except:  # pylint: disable=bare-except
    print(f'Workchain {workchain_node} has some errors or not finished yet')
    #print('Workchain {} has some errors or not finished yet'.format(workchain_node))
