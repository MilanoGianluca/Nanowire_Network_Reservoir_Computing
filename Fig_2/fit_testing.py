#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os, sys

#import user-defined functions
from functions import define_grid_graph_2,initialize_graph_attributes, mod_voltage_node_analysis, calculate_network_resistance, calculate_Vsource, calculate_Isource, update_edge_weigths

#%%
out_dir = './out_data/' 
if not os.path.exists(r'./out_data/'):
    os.makedirs(r'./out_data/')
    
step = 15 # sampling experimental data

data = np.loadtxt('./raw_data/pulse_2V.txt')
title = 'Pulseshape (2V)'
time = data[::step, 0]
time = time - time[0]
V = data[::step, 2]
I_exp = data[::step, 1]
G_exp = I_exp/V

kp0 =  8.422409820914783e-10
kd0 =  0.048697579017353006
eta_p =  0.19999370301178968
eta_d =  158.02444821482402
Y_0 =  0.0008866475159246351
Y_min =  0.001123825331225794
Y_max =  0.0030515679724941363

#%% INITIALIZATION
xdim = 21                                                         
ydim = 21

padI = 31  
padII = 409

sourcenode_pos = [padI]                                                          
groundnode_pos = [padII]    

G = define_grid_graph_2(xdim, ydim)

#%% APPLIED VOLTAGE LIST

Vin_list = V
timesteps = len(time)
print('Timesteps: '+str(timesteps))
t_list = time

#%% GRAPH INITIALIZATION    
G = initialize_graph_attributes(G, Y_0)

#%% GROWTH OF THE CONDUCTIVE PATH

####Initialization of list over time
H_list = [[] for t in range(0, timesteps)]
I_list = [[] for t in range(0, timesteps)]
V_list = [[] for t in range(0, timesteps)]
Ynetwork_list = [[] for t in range(0, timesteps)]
Rnetwork_list = [[] for t in range(0, timesteps)]

####Pristine state
H_list[0] = mod_voltage_node_analysis(G, [V[0]], [sourcenode_pos[0]], [groundnode_pos[0]])
I_list[0] = calculate_Isource(H_list[0], sourcenode_pos[0])
V_list[0] = calculate_Vsource(H_list[0], sourcenode_pos[0])
Rnetwork_list[0] = calculate_network_resistance(H_list[0], sourcenode_pos[0])
Ynetwork_list[0] = 1/Rnetwork_list[0]

####Growth over time
for i in range(1, int(timesteps)):

    delta_t = t_list[i] - t_list[i-1]

    update_edge_weigths(G, delta_t, Y_min, Y_max, kp0, eta_p, kd0, eta_d)                                  #update edges
    
    H_list[i] = mod_voltage_node_analysis(G, [V[i]], [sourcenode_pos[0]], [groundnode_pos[0]])
    I_list[i] = calculate_Isource(H_list[i], sourcenode_pos[0])
    V_list[i] = calculate_Vsource(H_list[i], sourcenode_pos[0])
    Rnetwork_list[i] = calculate_network_resistance(H_list[i], sourcenode_pos[0])
    Ynetwork_list[i] = 1/Rnetwork_list[i]
    
    sys.stdout.write("\rNetwork Stimulation: "+str(i+1)+'/'+str(timesteps))

#%% Conductance plot

plt.figure(figsize=(10,7))
plt.title(title, fontsize=30)
plt.plot(t_list, G_exp, 'b', label='exp', linewidth = 2.2)
plt.plot(t_list, Ynetwork_list, '--r', label='model', linewidth = 2.2)
plt.legend(fontsize=30)
plt.grid()
plt.xlabel('time [s]', fontsize=30)
plt.ylabel('Conductance [S]', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(out_dir+'conductance_model_vs_exp')
#  plt.show()

#%% conductive path frames
def plot(H):

    fig4, ax = plt.subplots(figsize=(10, 10))
    plt.cla()
    pos=nx.get_node_attributes(H,'pos')
    
    nx.draw_networkx(H, pos,
                      #NODES
                      node_size=60,
                      node_color=[H.nodes[n]['V'] for n in H.nodes()],
                      cmap=plt.cm.Blues,
                      vmin=0,
                      vmax=2.0,
                      #EDGES
                      width=4,
                      edge_color=[H[u][v]['Y'] for u,v in H.edges()],
                      edge_cmap=plt.cm.Reds,
                      edge_vmin=Y_min,
                      edge_vmax=Y_max,
                      with_labels=False,   #Set TRUE to see node numbers
                      font_size=6,)
    plt.title('t = '+str(round(time[i],2))+' s', fontsize = 15)
    nx.draw_networkx_nodes(H, pos, nodelist=[31, 409], node_size=100, node_color='k')
 
list_index = [2, 15, 131, 137, 144, 179] 
count = 0   
for i in list_index:
    plot(H_list[i])
    count += 1
    name = 'conductive_path_'+str(count)
    plt.savefig(out_dir+name)   

#%% Export Data

my_data=np.vstack((t_list,V_list,I_exp, G_exp, I_list, Ynetwork_list))
my_data=my_data.T
np.savetxt(out_dir+'2V_pulse_data.txt',my_data, delimiter=',', header='time, V, I_exp, G_exp, I_model, G_model',comments='')

