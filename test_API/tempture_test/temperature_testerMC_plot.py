# -*- coding: utf-8 -*-
"""
Perform a plot from the data that results from a given query
"""
import typing
import matplotlib.pyplot as plt
from aiida import load_profile, orm

load_profile()


def cal_node_query(workchain_pk: typing.Union[int, str]) -> list:
    """
    Get dictionary nodes that result from a query

    :param workchain_pk: pk number identifying a given workchain node
    :type workchain_pk: typing.Union[int, str]
    :return: list of nodes resulting from the query
    :rtype: list
    """
    qb = orm.QueryBuilder()
    qb.append(orm.WorkChainNode, filters={'id': str(workchain_pk)}, tag='workflow_node')
    qb.append(orm.Dict, with_incoming='workflow_node', tag='workdict')

    return qb.all()


with open('UppASDTemperatureWorkflow_jobPK.csv', 'r') as f:
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
except BaseException:  # pylint: disable=broad-except
    print(f'Workchain {workchain_node} has some errors or not finished yet')
