#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:53:23 2021

@author: qichen
"""
import numpy as np
import pandas as pd
import dgl 
import torch 
from torch.utils.data import Dataset, DataLoader




from aiida.orm import load_node,QueryBuilder,ArrayData,Dict
from aiida.orm.nodes.process.calculation.calcjob import CalcJobNode
import aiida
aiida.load_profile()
    
    
    
'''
files that we need to parse:

'''
'''
#prototype
def build_graph(dmdata_file=None,aiida_output=None,
                mom_out_file = None,previous_state = None,
                atoms_number = None):
    if dmdata_file != None:
        
        data_full = pd.read_csv(dmdata_file,sep='\s+', header=None,skiprows=5)
        atom_site_A = list(data_full[0])
        atom_site_B = list(data_full[1])
        #because dgl nodes is start from 0 so we need to use atom site minius 1 to fit DGL
        atom_site_A[:] = [i-1 for i in atom_site_A]
        atom_site_B[:] = [i-1 for i in atom_site_B]

        Jij_x = np.array(data_full[4])
        Jij_y = np.array(data_full[5])
        Jij_z = np.array(data_full[6])
        DM_x  = np.array(data_full[7])
        DM_y  = np.array(data_full[8])
        DM_z  = np.array(data_full[9])
        
    if previous_state != None:
        if mom_out_file == None:
            print("Error, we need some previous state that calculated from UppASD")
            
        else:
            mom_output = pd.read_csv(mom_out_file,sep='\s+', header=None,skiprows=7)
            mom_x = mom_output[4][:atoms_number*previous_state]
            mom_y = mom_output[5][:atoms_number*previous_state]
            mom_z = mom_output[6][:atoms_number*previous_state]
            
            mom_states =np.array([mom_x,mom_y,mom_z]).transpose() 
            mom_states = np.concatenate(np.split(mom_states,previous_state),axis=1)
    
    
 
    
    g = dgl.graph((atom_site_A,atom_site_B))
    g.edata['J_ij'] = torch.from_numpy(np.array([Jij_x,Jij_y,Jij_z]).transpose())
    g.edata['DM'] = torch.from_numpy(np.array([DM_x,DM_y,DM_z]).transpose())
    g.ndata['mom'] = torch.from_numpy(mom_states)
    
    return g
'''
'''
for test:
g = build_graph(dmdata_file='./uppasd_opfile/dmdata.SCsurf_T.out',mom_out_file='./uppasd_opfile/moment.SCsurf_T.out',atoms_number=900,previous_state=5)
'''   

'''    
def build_graph(dmdata_out_file= None,state_array= None, 
                globle_attribute_array= None):
    #dm out data includes all edge information, we need this to build the graph itself (connections)
    if dmdata_out_file != None:
        data_full = pd.read_csv(dmdata_out_file,sep='\s+', header=None,skiprows=5)
        atom_site_A = list(data_full[0])
        atom_site_B = list(data_full[1])
        #because dgl nodes is start from 0 so we need to use atom site minius 1 to fit DGL
        atom_site_A[:] = [i-1 for i in atom_site_A]
        atom_site_B[:] = [i-1 for i in atom_site_B]

        Jij_x = np.array(data_full[4])
        Jij_y = np.array(data_full[5])
        Jij_z = np.array(data_full[6])
        DM_x  = np.array(data_full[7])
        DM_y  = np.array(data_full[8])
        DM_z  = np.array(data_full[9])
    else: 
        print('error, lack of edge-information')
        
    
    node_attr = torch.from_numpy(state_array[0]).float()
    i = 1 
    previous_state = len(state_array)
    while i < previous_state:
        node_attr = torch.cat((node_attr, torch.from_numpy(state_array[i]).float()),1)
        i = i+1

    #node_attr = torch.cat((node_attr, torch.from_numpy(globle_attribute_array).float()),1)

    g = dgl.graph((atom_site_A,atom_site_B))
    g.edata['J_ij'] = torch.from_numpy(np.array([Jij_x,Jij_y,Jij_z]).transpose())
    g.edata['DM'] = torch.from_numpy(np.array([DM_x,DM_y,DM_z]).transpose())
    g.ndata['mom'] = node_attr
    
    return g
            
'''
    

def build_graph(atom_site_A,atom_site_B,rij,mom_array,J_ij):
    g = dgl.graph((atom_site_A,atom_site_B))
    g.edata['r_ij_J_ij'] = torch.from_numpy(np.array([rij,J_ij]).transpose())  # not sure if we need transpose here
    g.ndata['mom'] = torch.from_numpy(mom_array)
    #g.ndata['globle'] = torch.from_numpy(globle_attribute_array)
    return g
    



def define_globle_attribute(g_attribute_list,node_number):
    """
    here we use the method in the paper <Learning to Simulate Complex Physics with Graph Networks>
    put the globle attribute into the node attribute.

    Parameters
    ----------
    g_attribute_list : TYPE
        things like temperature, externel magnetic field or others
        this should be a list like [temp,mf, .., .., ..]
    node_number : TYPE
        here node number is the total number of atoms in the system

    Returns
    -------
    globle_attribute_array : TYPE
        DESCRIPTION.

    """
    #
    init_m = np.ones([node_number,1])
    temp_m = init_m.copy()
    for attribute in g_attribute_list:
        temp_m = np.concatenate( [temp_m,init_m*attribute],1)
    globle_attribute_array = temp_m[:,1:]
    return globle_attribute_array
    

    
def trajectory_parser(mom_x,mom_y,mom_z,atoms_total):
    mom_states =np.array([mom_x,mom_y,mom_z]).transpose() 
    mom_states = np.array(np.split(mom_states,len(mom_x)/atoms_total)) #because trajectory includes first state we need to do that.
    return mom_states



def split_mom_training_step_pairs(mom_statas,previous_steps):
    
    split_mom_list = []
    if previous_steps != 0:
        init_mom_states = mom_statas[0:previous_steps]
    else:
        init_mom_states = mom_statas[0]
    total_step = len(mom_statas)
    index_init = 1
    if previous_steps != 0:
        while (index_init+previous_steps) <= total_step :
            split_mom_list.append(mom_statas[index_init:index_init+previous_steps])
            index_init = index_init+1
    else:
        while (index_init+previous_steps) < total_step :
            split_mom_list.append(mom_statas[index_init])                
            index_init = index_init+1
    temp_a = split_mom_list[0:-1]
    temp_b = split_mom_list[1:]
    mom_pair_list = []
    for i in list(range(len(temp_a))):
        mom_pair_list.append([temp_a[i],temp_b[i]])
    return init_mom_states,mom_pair_list




def query_finished_node(wf_node_pk):
    #inquire all node that exitcode = 0
    workfunctionnode = load_node(wf_node_pk)
    #finished_node_list = []
    finished_pk_list = []
    for c_node in workfunctionnode.called:
        if c_node.exit_status == 0:
            #finished_node_list.append(c_node)
            finished_pk_list.append(c_node.pk)
    #return finished_node_list,finished_pk_list
    return finished_pk_list


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
            


def input_node_query(cal_node_pk,input_node_name,attribute_name):
    qb = QueryBuilder()
    qb.append(CalcJobNode, filters={'id': str(cal_node_pk)}, tag='cal_node')
    qb.append(
        Dict,
        with_outgoing='cal_node',
        edge_filters={'label': {'==':input_node_name}}
        )
    all_dict = qb.all()
    return all_dict[0][0][attribute_name]

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)#back L2 norm
    r = np.hypot(hxy, z)
    az = np.arctan2(y, x) #phi
    el = np.arctan2(z, hxy) # theta   elevation angle
    return az, el, r

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def spherical_coordinate_trajectory_parser(mom_x,mom_y,mom_z,atoms_total):
    mom_states =np.array([mom_x,mom_y,mom_z]).transpose() 
    mom_states_sc = []
    for i in mom_states:
        mom_states_sc_temp = list(cart2sph(*i))
        mom_states_sc.append(mom_states_sc_temp[0:2])
    mom_states_sc = np.array(mom_states_sc)
    
    mom_states_sc = np.array(np.split(mom_states_sc,len(mom_x)/atoms_total)) #because trajectory includes first state we need to do that.
    
    return mom_states_sc


def aiida_trajectory_to_graph_list(pk):

    
    
    mom_states_x = output_node_query(pk,'mom_states_traj','mom_states_x')
    mom_states_y = output_node_query(pk,'mom_states_traj','mom_states_y')
    mom_states_z = output_node_query(pk,'mom_states_traj','mom_states_z')
    #atoms_n_list = input_node_query(pk,'inpsd','ncell').split()#looks like ['1', '2', '3']
    #atoms_total = int(atoms_n_list[0])*int(atoms_n_list[1])*int(atoms_n_list[2])
    atoms_total = 2# this is for dipole so here it should be 2
    mom_array_sc = spherical_coordinate_trajectory_parser(mom_states_x,mom_states_y,mom_states_z,atoms_total)
    init_mom_states,mom_pair_list = split_mom_training_step_pairs(mom_array_sc,10)
    
    atom_site_A = output_node_query(pk,'struct_out','atom_site_A')
    atom_site_B = output_node_query(pk,'struct_out','atom_site_B')
    rij_x= output_node_query(pk,'struct_out','rij_x')
    rij_y= output_node_query(pk,'struct_out','rij_y')
    rij_z= output_node_query(pk,'struct_out','rij_z')
    rij= output_node_query(pk,'struct_out','rij')
    J_ij= output_node_query(pk,'struct_out','J_ij')
    
    

    
    graph_pair_list = []
    for mom_pair in mom_pair_list:
        g1 = build_graph(atom_site_A,atom_site_B,rij,np.concatenate(mom_pair[0],1),J_ij)   #np.concatenate(mom_pair[0],1) means we want to make all previous stop into 1d vector
        g2 = build_graph(atom_site_A,atom_site_B,rij,np.concatenate(mom_pair[1],1),J_ij)
        graph_pair_list.append([g1,g2])
    return graph_pair_list





    
    
def generate_train_test_set(input_pk_list):  
    from sklearn.model_selection import train_test_split
    train_pk_list,test_pk_list = train_test_split(input_pk_list,test_size = 0.1)#this will split the list randomly
    return train_pk_list,test_pk_list
'''
    
def generate_graph_pair_list_for_dataset(splited_pk_list):
    whole_graph_pair_list = []
    for pk in splited_pk_list:
        whole_graph_pair_list = whole_graph_pair_list + aiida_trajectory_to_graph_list(pk)
    return whole_graph_pair_list
'''
def generate_graph_pair_list_for_dataset(splited_pk_list):
    whole_graph_pair_list = []
    for pk in splited_pk_list:
        whole_graph_pair_list.append(aiida_trajectory_to_graph_list(pk))
    return whole_graph_pair_list
    
'''
from dgl.data import DGLDataset
class Graph_pairs_for_one_trajectory(DGLDataset):
    def __init__(self,g_list):
        self.gplist = g_list
    def __getitem__(self, index):
        return self.gplist[index]
    def __len__(self):
        return len(self.gplist)
'''

    
def store_graph_pair_list(sote_path,g_list,file_name):
    torch.save(g_list,sote_path+file_name)

'''
test:
x = trajectory_parser('./uppasd_opfile/moment.SCsurf_T.out',100)
'''
wf_node_pk = 111239
split_number = 100 #this means when we store our dataset we split it and store 8 trajectory into 1 split_dataset

input_pk_list = query_finished_node(wf_node_pk)
train_pk_list,test_pk_list = generate_train_test_set(input_pk_list)


#split train pk list into small set
name_machine = locals()
train_split_name_list = []
for list_name in range(len(train_pk_list)//split_number+1):
    if list_name != len(train_pk_list)//split_number:
        name_machine['train_set_{}'.format(list_name)] = train_pk_list[list_name*split_number:(list_name+1)*split_number]
        train_split_name_list.append('train_set_{}'.format(list_name))
    else:
        name_machine['train_set_{}'.format(list_name)] = train_pk_list[list_name*split_number:len(train_pk_list)]
        train_split_name_list.append('train_set_{}'.format(list_name))
        
#split test pk list into small set
name_machine = locals()
test_split_name_list = []
for list_name in range(len(test_pk_list)//split_number+1):
    if list_name != len(test_pk_list)//split_number:
        name_machine['test_set_{}'.format(list_name)] = test_pk_list[list_name*split_number:(list_name+1)*split_number]
        test_split_name_list.append('test_set_{}'.format(list_name))
    else:
        name_machine['test_set_{}'.format(list_name)] = test_pk_list[list_name*split_number:len(test_pk_list)]
        test_split_name_list.append('test_set_{}'.format(list_name))
     
        
        
#store the dataset:
data_set_store_path ='/Users/qichen/dataset/dipole/test/' 
for dataset_name in test_split_name_list:
    test_graph_pair_set =  generate_graph_pair_list_for_dataset(eval(dataset_name))
    store_graph_pair_list(data_set_store_path,test_graph_pair_set,dataset_name+'.pt')
    print('{} is stored'.format(dataset_name))

        

data_store_path_2 = '/Users/qichen/dataset/dipole/train/'    
for dataset_name in train_split_name_list:
    train_graph_pair_set =  generate_graph_pair_list_for_dataset(eval(dataset_name))
    store_graph_pair_list(data_store_path_2,train_graph_pair_set,dataset_name+'.pt')
    print('{} is stored'.format(dataset_name))



        
test_graph_pair_set =  generate_graph_pair_list_for_dataset(test_pk_list)
#we need more than the graph pairs for test~ maybe we need the trajctory too
train_graph_pair_set =  generate_graph_pair_list_for_dataset(train_pk_list)


store_graph_pair_list('/Volumes/Untitled/datastore',test_graph_pair_set,'test_list.pt')
dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=10, shuffle=True)

a = torch.utils.data.DataLoader(dataset,batch_size=10)


for i,n in a:
    nn=i
    break

def creat_dataset_for_one_trajectory(dataset_name,):
    
def visualization_graph(G):
    import matplotlib.pyplot as plt
    nx_G = G.to_networkx()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]],font_size=1)
    plt.savefig("graph.png", dpi=1000)
'''