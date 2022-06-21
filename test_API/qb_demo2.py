# -*- coding: utf-8 -*-
#%% import package
from aiida.orm import QueryBuilder
import aiida
aiida.load_profile()
from aiida.orm.nodes.process.calculation.calcjob import CalcJobNode
from aiida.orm import Group
from aiida.orm import load_node, ArrayData
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def group_query(group_name):

    qb = QueryBuilder()
    qb.append(Group, filters={'label': str(group_name)}, tag='group')
    qb.append(CalcJobNode, tag='UppASD_demo_cal', with_group='group')
    pk_list = []
    for cj in qb.all():
        pk_list.append(cj[0].pk)
    #staff in pk_list here are cal_node
    return pk_list


def cal_node_query(cal_node_pk, attribute_name):

    qb = QueryBuilder()
    qb.append(CalcJobNode, filters={'id': str(cal_node_pk)}, tag='cal_node')
    qb.append(ArrayData, with_incoming='cal_node', tag='arrays')
    all_array = qb.all()
    for array in all_array:
        for name in array[0].get_arraynames():
            if name == attribute_name:
                return array[0].get_array(attribute_name)


def demo_plot1(df, title, path):
    sns.set_theme(style='darkgrid')
    g = sns.lineplot(
        data=df,
        palette='tab10',
        linewidth=2,
    )
    g.set(xlabel='Lter_num', ylabel=r'$Magnetization (\mu_B)$')
    g.set(title=f'{title}')
    g.figure.savefig(f'{path}/{title}.png', dpi=800, bbox_inches='tight')
    plt.close()
    return


group_name = 'demo_group2'

pk_list = group_query(group_name)
plot_df_1 = pd.DataFrame()
plot_df_2 = pd.DataFrame()
plot_df_3 = pd.DataFrame()
plot_df_4 = pd.DataFrame()
for pk in pk_list:
    cal_node = load_node(pk)

    label = cal_node.label
    Iter_num_list = cal_node_query(pk, 'Iter_num_average')
    M_x_list = cal_node_query(pk, 'M_x')
    plot_df_1[label] = list(map(float, M_x_list.tolist()))
    M_y_list = cal_node_query(pk, 'M_y')
    plot_df_2[label] = list(map(float, M_y_list.tolist()))
    M_z_list = cal_node_query(pk, 'M_z')
    plot_df_3[label] = list(map(float, M_z_list.tolist()))
    M_list = cal_node_query(pk, 'M')
    plot_df_4[label] = list(map(float, M_list.tolist()))

    df = pd.DataFrame({
        'M_y': list(map(float, M_y_list.tolist())),
        'M_z': list(map(float, M_z_list.tolist())),
        'M_x': list(map(float, M_x_list.tolist())),
        'M': list(map(float, M_list.tolist())),
    })
    df.set_axis(list(map(int, Iter_num_list)))
    demo_plot1(df, label, './demo3_plots')
plot_df_1 = plot_df_1.set_axis(list(map(int, Iter_num_list)))
demo_plot1(plot_df_1, 'M_x', './demo3_plots')

plot_df_2 = plot_df_2.set_axis(list(map(int, Iter_num_list)))
demo_plot1(plot_df_2, 'M_y', './demo3_plots')

plot_df_3 = plot_df_3.set_axis(list(map(int, Iter_num_list)))
demo_plot1(plot_df_3, 'M_z', './demo3_plots')

plot_df_4 = plot_df_4.set_axis(list(map(int, Iter_num_list)))
demo_plot1(plot_df_4, 'M', './demo3_plots')
