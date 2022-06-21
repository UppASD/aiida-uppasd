# -*- coding: utf-8 -*-
from aiida.orm.nodes import WorkChainNode
from aiida.orm import QueryBuilder, Dict
import matplotlib.pyplot as plt
import numpy as np
import aiida
import json

aiida.load_profile()


def cal_node_query(workchain_pk, attribute_name):

    qb = QueryBuilder()
    qb.append(WorkChainNode,
              filters={'id': str(workchain_pk)},
              tag='workflow_node')
    qb.append(Dict, with_incoming='workflow_node', tag='workdict')

    return qb.all()


with open('UppASDTemperatureWorkflow_jobPK.csv', 'r') as f:
    workchain_node = f.read()
try:
    data = cal_node_query(workchain_node,
                          'temperature_output')[0][0].get_dict()
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
except:
    print(f'Workchain {workchain_node} has some errors or not finished yet')
