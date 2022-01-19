import click
from aiida.orm import load_node
import numpy as np
import plotext
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from aiida.orm import load_node,QueryBuilder,ArrayData,Dict
from aiida.orm.nodes.process.calculation.calcjob import CalcJobNode
import aiida
#some code in motion part are modified from https://stackoverflow.com/questions/48911643/set-uvc-equivilent-for-a-3d-quiver-plot-in-matplotlib?answertab=active#tab-top

@click.group()
def UppASD_cli():
    '''help'''

@UppASD_cli.command('visualization_observations')
@click.option('-iter_slice',default = -1)#means we need all iterations -1 just a placeholder
@click.option('--y_axis',default = ['Tot'],multiple=True)# here the input should be a tuple -- means multiple
@click.option('-plot_style',default='line')
@click.option('-plot_name',default='None')
@click.option('-width',default=100)
@click.option('-height',default=20)
@click.argument('pk')
def visualization_observations(iter_slice,y_axis,plot_style,plot_name,width,height,pk):
    """help"""
    auto_name=locals()
    cal_node = load_node(pk)
    if iter_slice != -1:

        for name in y_axis:
            '''
            All array name here:
            ['BQ','DM','PD','Ani','Dip',' ','LSF','Tot','Chir','BiqDM','Zeeman','Iter_num_totenergy']
            '''
            auto_name[str(name)] = cal_node.get_array(str(name))[:int(iter_slice)].astype(float)
        if plot_style == 'line':

            iter_list = cal_node.get_array('Iter_num_totenergy')[:int(iter_slice)].astype(int)
            for name in y_axis:
                plotext.plot(iter_list,eval(str(name)), label = str(name))
            plotext.plotsize(width, height)
            if plot_name != 'None':
                plotext.title('{}'.format(plot_name))
            else:
                plotext.title('Result')
            plotext.show()
        elif plot_style == 'scatter':
            iter_list = cal_node.get_array('Iter_num_totenergy')[:int(iter_slice)].astype(int)
            for name in y_axis:
                plotext.scatter(iter_list,eval(str(name)), label = str(name))
            plotext.plotsize(width, height)
            if plot_name != 'None':
                plotext.title('{}'.format(plot_name))
            else:
                plotext.title('Result')
            plotext.show()
        else:
            print("We only support line or scatter plot now")
    else:

        for name in y_axis:
            '''
            All array name here:
            ['BQ','DM','PD','Ani','Dip','Exc','LSF','Tot','Chir','BiqDM','Zeeman','Iter_num_totenergy']
            '''
            auto_name[str(name)] = cal_node.get_array(str(name)).astype(float)
        if plot_style == 'line':

            iter_list = cal_node.get_array('Iter_num_totenergy').astype(int)
            for name in y_axis:
                plotext.plot(iter_list,eval(str(name)), label = str(name))
            plotext.plotsize(width, height)
            if plot_name != 'None':
                plotext.title('{}'.format(plot_name))
            else:
                plotext.title('Result')
            plotext.show()
        elif plot_style == 'scatter':
            iter_list = cal_node.get_array('Iter_num_totenergy')[:int(iter_slice)].astype(int)
            for name in y_axis:
                plt.scatter(iter_list,eval(str(name)), label = str(name))
            plotext.plotsize(width, height)
            if plot_name != 'None':
                plotext.title('{}'.format(plot_name))
            else:
                plotext.title('Result')
            plotext.show()
        else:
            print("We only support line or scatter plot now")


def output_node_query(cal_node_pk,output_array_name,attribute_name):
    qb = QueryBuilder()
    qb.append(CalcJobNode, filters={'id': str(cal_node_pk)}, tag='cal_node')
    qb.append(
        ArrayData,
        with_incoming='cal_node',
        edge_filters={'label': {'==':output_array_name}}
        )
    all_array = qb.all()
    return all_array[0][0].get_array(attribute_name)

def trajectory_parser(mom_x,mom_y,mom_z,atoms_total):
    mom_states =np.array([mom_x,mom_y,mom_z]).transpose() 
    mom_states = np.array(np.split(mom_states,len(mom_x)/atoms_total)) #because trajectory includes first state we need to do that.
    return mom_states

def get_arrow_next(i):
    x = coord_r[:,0]
    y = coord_r[:,1]
    z = coord_r[:,2]
    rot_mom_array =  r.apply(mom_array_from_result[i])
    u = rot_mom_array[:,0] 
    v = rot_mom_array[:,1] 
    w = rot_mom_array[:,2] 
    return x,y,z,u,v,w
def animate(i):
    global quivers
    quivers.remove()
    colors = cm.jet(mom_array_from_result[i][axis_to_colorbar])
    quivers= ax.quiver(*get_arrow_next(i),arrow_length_ratio=arrow_ratio_arr,length=length_arr ,colors =colors_arr,normalize = normalize_flag_arr)


@UppASD_cli.command('visualization_motion')
@click.option('-rotation_axis',default = 'x')
@click.option('-rotation_matrix',default = [0])#note that it should fit the rotation axis length
@click.option('-color',default = 'b')# color_bar are test feature, works not well now
@click.option('-arrow_head_ratio',default=0.3)
@click.option('-length_ratio',default=0.3)
@click.option('-normalize_flag',default=True)
@click.option('-height',default=20)
@click.option('-width',default=20)
@click.option('-color_bar_axis',default='x')
@click.option('-path_animation',default='./motion.gif')
@click.option('-interval_time',default=200)
@click.option('-dpi_setting',default=100)
@click.option('-path_frame',default='./motion.png')
@click.option('-frame_number',default=0)
@click.option('-animation_flag',default=False)
@click.argument('pk')

def visualization_motion(rotation_axis,
                        rotation_matrix,color,
                        arrow_head_ratio,length_ratio,
                        normalize_flag,height,
                        width,color_bar_axis,path_animation,interval_time,
                        dpi_setting,path_frame,frame_number,animation_flag,pk):
    global coord_r,mom_array_from_result,r,quivers,axis_to_colorbar,ax,arrow_ratio_arr,length_arr,colors_arr,normalize_flag_arr
    r = R.from_euler(rotation_axis,rotation_matrix , degrees=True)
    mom_states_x = output_node_query(pk,'mom_states_traj','mom_states_x')
    mom_states_y = output_node_query(pk,'mom_states_traj','mom_states_y')
    mom_states_z = output_node_query(pk,'mom_states_traj','mom_states_z')
    coord = output_node_query(pk,'coord','coord')[:,1:4]

    coord_r = r.apply(coord)
    atoms_total = len(coord)
    arrow_ratio_arr=arrow_head_ratio
    length_arr=length_ratio
    colors_arr =color
    normalize_flag_arr=normalize_flag
    mom_array_from_result = trajectory_parser(mom_states_x,mom_states_y,mom_states_z,atoms_total)
    
    if color_bar_axis == 'x':
        axis_to_colorbar = 0
    elif color_bar_axis == 'y':
        axis_to_colorbar = 1
    else:
        axis_to_colorbar = 2
    colors = cm.jet(mom_array_from_result[1][axis_to_colorbar])

    fig = plt.figure(figsize=(height,width))
    ax = fig.gca(projection='3d')

    if animation_flag == False:
        quivers = ax.quiver(*get_arrow_next(frame_number),arrow_length_ratio=arrow_ratio_arr,length=length_arr ,colors =colors_arr,normalize = normalize_flag_arr)
        fig.savefig(path_frame)
    #quivers = ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.3,length=0.8, colors = colors)

    elif animation_flag == True:
        quivers = ax.quiver(*get_arrow_next(0),arrow_length_ratio=arrow_ratio_arr,length=length_arr ,colors =colors_arr,normalize = normalize_flag_arr)
        ani = FuncAnimation(fig, animate, frames =list(range(len(mom_array_from_result))), interval = interval_time)
        ani.save(path_animation,dpi=dpi_setting)

