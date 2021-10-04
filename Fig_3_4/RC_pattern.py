#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#import user-defined functions
from functions import define_grid_graph_2,initialize_graph_attributes, mod_voltage_node_analysis, update_edge_weigths
from functions_reservoir import dataset_to_pulse, insert_R_to_graph, remove_R_from_graph
#%%
def plot(H):
    remove_R_from_graph(H, src, new_nodes, gnd)
    fig3, ax = plt.subplots(figsize=(10, 10))
    plt.cla()
    pos=nx.get_node_attributes(H,'pos')
    
    nx.draw_networkx(H, pos,
                      #NODES
                      node_size=60,
                      node_color=[H.nodes[n]['V'] for n in H.nodes()],
                      cmap=plt.cm.Blues,
                      vmin=0,
                      vmax=pulse_amplitude+V_read,
                      #EDGES
                      width=4,
                      edge_color=[H[u][v]['Y'] for u,v in H.edges()],
                      edge_cmap=plt.cm.Reds,
                      edge_vmin=g_min,
                      edge_vmax=g_max,
                      with_labels=False,   #Set TRUE to see node numbers
                      font_size=6,)
    nx.draw_networkx_nodes(H, pos, nodelist=src, node_size=100, node_color='k')


#%% OUTPUT DIRECTORIES
    
out_dir_3 = './out_data/Fig3/'
out_dir_4 = './out_data/Fig4/'

if not os.path.exists(r'./out_data/Fig3/'):
    os.makedirs(r'./out_data/Fig3/')
if not os.path.exists(r'./out_data/Fig4/'):
    os.makedirs(r'./out_data/Fig4/')

#%% DATASET LOAD & DISPLAY

file_to_train = './raw_data/pattern'

file_train = file_to_train+'.txt'
file_train_class = file_to_train+'_class.txt'

digit_train = np.loadtxt(file_train)
digit_train_class = np.loadtxt(file_train_class)

digit_rows = 4
digit_cols = 4

total_rows_train = int(len(digit_train))
num_digits_train = len(digit_train_class)

digit_list_train = [[] for i in range(0, num_digits_train)]
for i in range(0, num_digits_train):
    digit_list_train[i] =digit_train[digit_rows*i:digit_rows*(i+1)][:]
    
color_n = ['--*b','--*r','--*g']
name = ['North', 'East', 'South']
pattern_name = ['diag1', 'diag2', 'horz', 'vert']
    
#%% NETWORK PROPERTIES DEFINITION
# north pulse fit
kp0 = 2.555173332603108574e-06
kd0 = 6.488388862524891465e+01
eta_p = 3.492155165334443012e+01
eta_d = 5.590601016803570467e+00
g_min = 1.014708121672117710e-03
g_max = 2.723493729125820492e-03
g0 = 5.602507668026886038e-04
g0 = g_min
   
xdim = 21  # graph dimension
ydim = 21
frame = 2 #number of frame rows/columns

left_pads = [(xdim-1)*(frame+1)-2-3*i for i in range(0, 5)] # (from top to bottom)
right_pads = [(xdim*xdim-1)-xdim*frame-frame-2-3*i for i in range(0, 5)] # (from top to bottom)
top_pads = [(xdim-1)+xdim*(frame+2)-frame+xdim*3*i for i in range(0, 5)] # (from left to right)
bottom_pads = [xdim*(frame+2)+frame+xdim*3*i for i in range(0, 5)] # (from left to right)

pad_N = top_pads[2]  
pad_E = right_pads[2]
pad_S = bottom_pads[2]  
pad_W = left_pads[2]

src = [pad_N, pad_E, pad_S, pad_W]
new_nodes = [xdim*xdim+nn for nn in range(4)]
gnd = [new_nodes[-1]+1]

#%% NETWORK INPUTs

####### CUSTOMIZE YOUR PULSE SHAPE ###############
R_read = [82]*4
V_read = 100e-3
pulse_amplitude = 5 # Volts

delta_pot = 250e-6 # distance for potentiation points
delta_dep = 250e-6 # distance for depression points
delta_read = delta_dep
delta = 250e-6 # distance for transition from low to high signal and viceversa 

pulse_time = 10e-3-delta
idle_time = 5e-4-delta 
read_time = 5.5e-3 #  seconds between write and read

####################################################
read_timesteps = int(read_time/delta_read)-1
pulse_timesteps = int(pulse_time/delta_pot) 
idle_timesteps = int(idle_time/delta_dep)

one_pulse = 2*idle_timesteps+pulse_timesteps+3 # points of a single pulse


time_write_1 = np.linspace(0, idle_time, idle_timesteps+1)
time_write_2 = np.linspace(idle_time+delta, idle_time+delta+pulse_time, pulse_timesteps+1)
time_write_3 = np. linspace(idle_time+pulse_time+2*delta, idle_time+pulse_time+2*delta+idle_time, idle_timesteps+1)
time_write_tot = np.append(np.append(time_write_1, time_write_2), time_write_3)

time_write = time_write_tot

for i in range(1, digit_cols):
    time_write = np.append(time_write, time_write_tot+time_write[-1]+delta_dep)

timesteps_write = len(time_write)

int_point = [[] for ip in range(digit_cols+1)]

for ip in range(0, digit_cols+1):
    int_point[ip] = one_pulse*ip+idle_timesteps 

int_point[-1] = timesteps_write+read_timesteps

dataset_readout = [[] for i in range(0, num_digits_train)] #  input for readout function
dataset_to_save = [[] for i in range(0, num_digits_train)] #  input for readout function

hist = np.zeros((5,4,3))
H_to_plot = [[[] for j in range(5)] for i in range(4)]
V_list_read = [[np.zeros((4, timesteps_write+read_timesteps+1))] for i in range(4)]

for digit in range(0, num_digits_train):
    
    print('Simulating pattern '+ pattern_name[digit])

    G = define_grid_graph_2(xdim, ydim)
    G = initialize_graph_attributes(G, g0) 
    input_digit = int(digit_train_class[digit])


    H_list_write = [[] for t in range(0, timesteps_write+read_timesteps+1)]
    
    train_pulse, _ = dataset_to_pulse(digit_rows, digit_cols, timesteps_write, pulse_timesteps+1, idle_timesteps+1, digit_list_train, digit, pulse_amplitude)
    
    Vin_list_write = [[] for t in range(0, timesteps_write)]

    for t in range(0, timesteps_write):
        for r in range(0, digit_rows):
            if train_pulse[r][t] == 0:
                Vin_list_write[t] = list(Vin_list_write[t])+[int(r==digit_rows-1)*V_read]
            else:
                Vin_list_write[t] = list(Vin_list_write[t])+[int(r==digit_rows-1)*V_read+np.multiply(pulse_amplitude,(train_pulse[r][t]))]

        
    insert_R_to_graph(G, R_read, src, new_nodes, gnd)
    
    H_list_write[0] = mod_voltage_node_analysis(G, Vin_list_write[0], new_nodes, gnd)
    
    for i in range(1, timesteps_write):
    
        delta_t = time_write[i] - time_write[i-1]
        
        remove_R_from_graph(G, src, new_nodes, gnd)        
        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)                                  #update edges
        insert_R_to_graph(G, R_read, src, new_nodes, gnd)
        
        H_list_write[i] = mod_voltage_node_analysis(G, Vin_list_write[i], new_nodes, gnd)
        for c in range(4):
            V_list_read[digit][0][c, i] = H_list_write[i].nodes[src[c]]['V']
            
    for i in range(timesteps_write, timesteps_write+read_timesteps+1):
    
        delta_t = delta_read
        
        remove_R_from_graph(G, src, new_nodes, gnd)        
        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)                                  #update edges
        insert_R_to_graph(G, R_read, src, new_nodes, gnd)
        
        H_list_write[i] = mod_voltage_node_analysis(G, Vin_list_write[-1], new_nodes, gnd)
        for c in range(4):
            V_list_read[digit][0][c, i] = H_list_write[i].nodes[src[c]]['V']
    
    time_int = int_point

    for t_int in range(5):
        H_to_plot[digit][t_int] = H_list_write[time_int[t_int]]
        for n in range(3):
            hist[t_int,digit,n] = H_list_write[time_int[t_int]].nodes[src[n]]['V']
       
    print('Pattern '+pattern_name[digit]+' completed\n')
    
#%% FIG 3

plt.figure(figsize=(18,9))

for n in range(3):
    plt.plot(range(5), hist[:,0,n], color_n[n], label=name[n], linewidth=1.5)
plt.title(pattern_name[0], fontsize=20)
plt.grid()
plt.xticks(range(5), ['t0', 't1', 't2', 't3', 't4'], fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Voltage [V]', fontsize=15)
plt.legend(fontsize=15)
plt.ylim([ np.min(hist[:,:,:])-0.001, np.max(hist[:,:,:])])  
plt.savefig(out_dir_3+'diag1_Vout_vs_time_model.png')

for t_int in range(5):
    plot(H_to_plot[0][t_int])
    name = 'pattern_'+pattern_name[0]+'_t_int_'+str(t_int)+'.png'
    plt.savefig(out_dir_3+name) 
    
fsave = hist[:,0,0]
fname = 'pattern_'+pattern_name[0]+'_Vout_vs_time_model.txt'
for n in range(1, 3):
    fsave = np.vstack((fsave, hist[:,0,n]))
np.savetxt(out_dir_3+fname, fsave.T, header='n1 - n2 - n3') 

#%% FIG 4

plt.figure(figsize=(18,9))    
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.bar([1, 2, 3], hist[-1,i,:], color='dodgerblue', alpha=0.5)
    plt.ylim([0, np.max(hist[[0,-1],:,:])])
    plt.xticks([1,2,3], fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Neuron', fontsize = 15)
    plt.ylabel('Voltage [V]', fontsize = 15)
    plt.title('Pattern: '+pattern_name[i], fontsize = 20)
plt.tight_layout()
plt.savefig(out_dir_4+'hist_model.png')

for p in range(4):
    plot(H_to_plot[p][-1])
    name = 'pattern_'+pattern_name[p]+'_t_int_'+str(t_int)+'.png'
    plt.savefig(out_dir_4 + name)
           
file_to_save = hist[-1,:,:]
np.savetxt(out_dir_4+'hist_model.txt', file_to_save.T, header='t1-t2-t3-t4', delimiter=' ')
       