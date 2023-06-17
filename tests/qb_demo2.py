# -*- coding: utf-8 -*-
"""
Set of helper functions to plot data from the calculations
"""
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aiida import load_profile, orm

load_profile()


def group_query(group_name: str) -> list:
    """
    Get the list of nodes belonging to a group

    :param group_name: name of the group being searched
    :type group_name: str
    :return: list of CalcJobNode which belong to a group
    :rtype: list
    """
    qb = orm.QueryBuilder()
    qb.append(orm.Group, filters={'label': str(group_name)}, tag='group')
    qb.append(orm.CalcJobNode, tag='UppASD_demo_cal', with_group='group')
    pk_list = []
    for calc_job in qb.all():
        pk_list.append(calc_job[0].pk)
    #staff in pk_list here are cal_node
    return pk_list


def cal_node_query(
    cal_node_pk: typing.Union[str, int],
    attribute_name: str,
) -> typing.Union[np.ndarray, None]:
    """
    Get an output array for a given calculation node

    :param cal_node_pk: identified for the calculation node
    :type cal_node_pk: typing.Union[str, int]
    :param attribute_name: Name of the array that one wants to get
    :type attribute_name: str
    :return: get the first array that matches the name
    :rtype: typing.Union[np.ndarray, None]
    """
    qb = orm.QueryBuilder()
    qb.append(orm.CalcJobNode, filters={'id': str(cal_node_pk)}, tag='cal_node')
    qb.append(orm.ArrayData, with_incoming='cal_node', tag='arrays')
    all_array = qb.all()
    for array in all_array:
        for name in array[0].get_arraynames():
            if name == attribute_name:
                return array[0].get_array(attribute_name)
    return None


def demo_plot1(data: pd.DataFrame, title: str, path: str):
    """
    Generate a plot of the magnetization

    :param data: data to be plotted
    :type data: pd.DataFrame
    :param title: plot title
    :type title: str
    :param path: path where the plot will be stored
    :type path: str
    """
    sns.set_theme(style='darkgrid')
    _plot = sns.lineplot(
        data=data,
        palette='tab10',
        linewidth=2,
    )
    _plot.set(xlabel='Lter_num', ylabel=r'$Magnetization (\mu_B)$')
    _plot.set(title=f'{title}')
    _plot.figure.savefig(f'{path}/{title}.png', dpi=800, bbox_inches='tight')
    plt.close()


GROUP_NAME = 'demo_group2'

PK_LIST = group_query(GROUP_NAME)
PLOT_DF_1 = pd.DataFrame()
PLOT_DF_2 = pd.DataFrame()
PLOT_DF_3 = pd.DataFrame()
PLOT_DF_4 = pd.DataFrame()
for pk in PK_LIST:
    cal_node = orm.load_node(pk)

    label = cal_node.label
    Iter_num_list = cal_node_query(pk, 'Iter_num_average')
    M_x_list = cal_node_query(pk, 'M_x')
    PLOT_DF_1[label] = list(map(float, M_x_list.tolist()))
    M_y_list = cal_node_query(pk, 'M_y')
    PLOT_DF_2[label] = list(map(float, M_y_list.tolist()))
    M_z_list = cal_node_query(pk, 'M_z')
    PLOT_DF_3[label] = list(map(float, M_z_list.tolist()))
    M_list = cal_node_query(pk, 'M')
    PLOT_DF_4[label] = list(map(float, M_list.tolist()))

    df = pd.DataFrame({
        'M_y': list(map(float, M_y_list.tolist())),
        'M_z': list(map(float, M_z_list.tolist())),
        'M_x': list(map(float, M_x_list.tolist())),
        'M': list(map(float, M_list.tolist())),
    })
    df.set_axis(list(map(int, Iter_num_list)))
    demo_plot1(df, label, './demo3_plots')
PLOT_DF_1 = PLOT_DF_1.set_axis(list(map(int, Iter_num_list)))
demo_plot1(PLOT_DF_1, 'M_x', './demo3_plots')

PLOT_DF_2 = PLOT_DF_2.set_axis(list(map(int, Iter_num_list)))
demo_plot1(PLOT_DF_2, 'M_y', './demo3_plots')

PLOT_DF_3 = PLOT_DF_3.set_axis(list(map(int, Iter_num_list)))
demo_plot1(PLOT_DF_3, 'M_z', './demo3_plots')

PLOT_DF_4 = PLOT_DF_4.set_axis(list(map(int, Iter_num_list)))
demo_plot1(PLOT_DF_4, 'M', './demo3_plots')
