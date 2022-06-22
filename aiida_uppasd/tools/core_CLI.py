# -*- coding: utf-8 -*-
"""
Set of command line functions for the handling of aiida-uppasd
"""
import typing
import click
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
from aiida import orm
import plotext


@click.group()
def uppasd_cli():
    '''help'''


@uppasd_cli.command('visualization_observations')
@click.option('-iter_slice', default=-1)
@click.option('--y_axis', default=['Tot'], multiple=True)
@click.option('-plot_style', default='line')
@click.option('-plot_name', default='None')
@click.option('-width', default=100)
@click.option('-height', default=20)
@click.argument('pk')
def visualization_observations(
    iter_slice: int,
    y_axis: list,
    plot_style: str,
    plot_name: str,
    width: int,
    height: int,
    pk: int,
):  # pylint: disable=too-many-arguments, too-many-branches
    """
    Visualize a given observable

    :param iter_slice: which iteration to visualize
    :type iter_slice: int
    :param y_axis: what quantity to visualize
    :type y_axis: list
    :param plot_style: which kind of plot style (matplotlib) to use
    :type plot_style: str
    :param plot_name: name of the plot
    :type plot_name: str
    :param width: width of the figure
    :type width: int
    :param height: height of the figure
    :type height: int
    :param pk: pk number of the calculation that one wishes to visualize
    :type pk: int
    """
    auto_name = locals()
    cal_node = orm.load_node(pk)
    if iter_slice != -1:

        for name in y_axis:
            _name = str(name)
            auto_name[_name] = cal_node.get_array(_name)[:int(iter_slice)].astype(float)
        if plot_style == 'line':

            iter_list = cal_node.get_array('iterations')[:int(iter_slice)].astype(int)
            for name in y_axis:
                _name = str(name)
                plotext.plot(iter_list, eval(_name), label=_name)
            plotext.plotsize(width, height)
            if plot_name != 'None':
                plotext.title(f'{plot_name}')
            else:
                plotext.title('Result')
            plotext.show()
        elif plot_style == 'scatter':
            iter_list = cal_node.get_array('iterations')[:int(iter_slice)].astype(int)
            for name in y_axis:
                _name = str(name)
                plotext.scatter(iter_list, eval(_name), label=_name)
            plotext.plotsize(width, height)
            if plot_name != 'None':
                plotext.title(f'{plot_name}')
            else:
                plotext.title('Result')
            plotext.show()
        else:
            print('We only support line or scatter plot now')
    else:

        for name in y_axis:
            _name = str(name)
            auto_name[_name] = cal_node.get_array(_name).astype(float)
        if plot_style == 'line':

            iter_list = cal_node.get_array('iterations').astype(int)
            for name in y_axis:
                _name = str(name)
                plotext.plot(iter_list, eval(_name), label=_name)
            plotext.plotsize(width, height)
            if plot_name != 'None':
                plotext.title(f'{plot_name}')
            else:
                plotext.title('Result')
            plotext.show()
        elif plot_style == 'scatter':
            iter_list = cal_node.get_array('iterations')[:int(iter_slice)].astype(int)
            for name in y_axis:
                _name = str(name)
                plt.scatter(iter_list, eval(_name), label=_name)
            plotext.plotsize(width, height)
            if plot_name != 'None':
                plotext.title(f'{plot_name}')
            else:
                plotext.title('Result')
            plotext.show()
        else:
            print('We only support line or scatter plot now')


def output_node_query(
    cal_node_pk: typing.Union[int, str],
    output_array_name: str,
    attribute_name: str,
) -> np.ndarray:
    """
    Get the array output of a given calculation node

    :param cal_node_pk: pk number for the node that is being queried
    :type cal_node_pk: typing.Union[int, str]
    :param output_array_name: name of the array that we are looking for
    :type output_array_name: str
    :param attribute_name: specific entry in the array that one is looking for
    :type attribute_name: str
    :return: get an array for a corresponding calculation node
    :rtype: np.ndarray
    """
    qb = orm.QueryBuilder()
    qb.append(
        orm.CalcJobNode,
        filters={'id': str(cal_node_pk)},
        tag='cal_node',
    )
    qb.append(
        orm.ArrayData,
        with_incoming='cal_node',
        edge_filters={'label': {
            '==': output_array_name
        }},
    )
    all_array = qb.all()
    return all_array[0][0].get_array(attribute_name)


def trajectory_parser(
    mom_x: np.array,
    mom_y: np.array,
    mom_z: np.array,
    atoms_total: int,
) -> np.ndarray:
    """
    Get a compound array that contains all the moments information.

    :param mom_x: array with the moments along x-direction
    :type mom_x: np.array
    :param mom_y: array with the moments along y-direction
    :type mom_y: np.array
    :param mom_z: array with the moments along z-direction
    :type mom_z: np.array
    :param atoms_total: total number of atoms
    :type atoms_total: int
    :return: compounded array with the moments
    :rtype: np.ndarray
    """
    mom_states = np.array([mom_x, mom_y, mom_z]).transpose()
    #because trajectory includes first state we need to do that.
    mom_states = np.array(np.split(mom_states, len(mom_x) / atoms_total))
    return mom_states


def get_arrow_next(array_name: str):
    """
    Get the coordinates of an array in cartesian and in rotated coordinates.

    :param array_name: bane if the array that one is parsing
    :type array_name: str
    :return: coordinates in cartesian and rotated coordinates
    :rtype: typing.Union[np.array,np.array,np.array,np.array, np.array, np.array]
    """
    rot_mom_array = r.apply(mom_array_from_result[array_name])
    return (
        coord_r[:, 0],
        coord_r[:, 1],
        coord_r[:, 2],
        rot_mom_array[:, 0],
        rot_mom_array[:, 1],
        rot_mom_array[:, 2],
    )


def animate(data):
    """
    Animate the magnetic moments
    """
    global quivers
    quivers.remove()
    quivers = ax.quiver(
        *get_arrow_next(data),
        arrow_length_ratio=arrow_ratio_arr,
        length=length_arr,
        colors=colors_arr,
        normalize=normalize_flag_arr
    )


@uppasd_cli.command('visualization_motion')
@click.option('-rotation_axis', default='x')
@click.option('-rotation_matrix', default=[0])  #note that it should fit the rotation axis length
@click.option('-color', default='b')  # color_bar are test feature, works not well now
@click.option('-arrow_head_ratio', default=0.3)
@click.option('-length_ratio', default=0.3)
@click.option('-normalize_flag', default=True)
@click.option('-height', default=20)
@click.option('-width', default=20)
@click.option('-color_bar_axis', default='x')
@click.option('-path_animation', default='./motion.gif')
@click.option('-interval_time', default=200)
@click.option('-dpi_setting', default=100)
@click.option('-path_frame', default='./motion.png')
@click.option('-frame_number', default=0)
@click.option('-animation_flag', default=False)
@click.argument('pk')
def visualization_motion(
    rotation_axis: str,
    rotation_matrix: list,
    color: str,
    arrow_head_ratio: float,
    length_ratio: float,
    normalize_flag: bool,
    height: int,
    width: int,
    color_bar_axis: str,
    path_animation: str,
    interval_time: int,
    dpi_setting: int,
    path_frame: str,
    frame_number: int,
    animation_flag: bool,
    pk: int,
):  # pylint: disable=too-many-arguments, too-many-locals
    """
    Visualize the magnetic moments of the calculation node

    :param rotation_axis: which axis to rotate over
    :type rotation_axis: str
    :param rotation_matrix: matrix to perform the vector rotation
    :type rotation_matrix: list
    :param color: color for the spins
    :type color: str
    :param arrow_head_ratio: ratio between the arrow head and the body
    :type arrow_head_ratio: float
    :param length_ratio: ratio for the length of the arrow
    :type length_ratio: float
    :param normalize_flag: whether or not to normalize the vectors
    :type normalize_flag: bool
    :param height: height of the plot
    :type height: int
    :param width: width of the plot
    :type width: int
    :param color_bar_axis: which axis determines the color bar
    :type color_bar_axis: str
    :param path_animation: path to store the animation
    :type path_animation: str
    :param interval_time: how often one performs the animation
    :type interval_time: int
    :param dpi_setting: dpi of the figure
    :type dpi_setting: int
    :param path_frame: path to store a single frame of the figure
    :type path_frame: str
    :param frame_number: which frame to render
    :type frame_number: int
    :param animation_flag: whether or not to animate the figure
    :type animation_flag: bool
    :param pk: pk number of the calculation that one wishes to animate the moments
    :type pk: int
    """
    global coord_r, mom_array_from_result, r, quivers, axis_to_colorbar, ax
    global arrow_ratio_arr, length_arr, colors_arr, normalize_flag_arr
    r = Rotation.from_euler(rotation_axis, rotation_matrix, degrees=True)
    mom_states_x = output_node_query(pk, 'trajectories_moments', 'moments_x')
    mom_states_y = output_node_query(pk, 'trajectories_moments', 'moments_y')
    mom_states_z = output_node_query(pk, 'trajectories_moments', 'moments_z')
    coord = output_node_query(pk, 'coord', 'coord')[:, 1:4]

    coord_r = r.apply(coord)
    atoms_total = len(coord)
    arrow_ratio_arr = arrow_head_ratio
    length_arr = length_ratio
    colors_arr = color
    normalize_flag_arr = normalize_flag
    mom_array_from_result = trajectory_parser(
        mom_states_x,
        mom_states_y,
        mom_states_z,
        atoms_total,
    )

    if color_bar_axis == 'x':
        axis_to_colorbar = 0
    elif color_bar_axis == 'y':
        axis_to_colorbar = 1
    else:
        axis_to_colorbar = 2

    fig = plt.figure(figsize=(height, width))
    ax = fig.gca(projection='3d')

    if not animation_flag:
        quivers = ax.quiver(
            *get_arrow_next(frame_number),
            arrow_length_ratio=arrow_ratio_arr,
            length=length_arr,
            colors=colors_arr,
            normalize=normalize_flag_arr
        )
        fig.savefig(path_frame)

    if animation_flag:
        quivers = ax.quiver(
            *get_arrow_next(0),
            arrow_length_ratio=arrow_ratio_arr,
            length=length_arr,
            colors=colors_arr,
            normalize=normalize_flag_arr
        )
        ani = FuncAnimation(
            fig,
            animate,
            frames=list(range(len(mom_array_from_result))),
            interval=interval_time,
        )
        ani.save(path_animation, dpi=dpi_setting)
