#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing
#import user-defined functions
from functions import define_grid_graph_2,initialize_graph_attributes, mod_voltage_node_analysis, update_edge_weigths
from functions_reservoir import dataset_to_pulse, insert_R_to_graph, remove_R_from_graph
#%% function to plot the graph
def plot(H):
    remove_R_from_graph(H, src, new_nodes, gnd)
    fig3, ax = plt.subplots(figsize=(10, 10))
    plt.cla()
    pos=nx.get_node_attributes(H,'pos')
    
    nx.draw_networkx(H, pos,
                      #NODES
                      node_size=60,
                      node_color=[H.nodes[n]['V']*int(n not in src)+5*int(n in src) for n in H.nodes()],
                      cmap=plt.cm.Blues,
                      vmin=0,
                      vmax=pulse_amplitude+V_read,
                      #EDGES
                      width=4,
                      edge_color=[H[u][v]['Y'] for u,v in H.edges()],
                      edge_cmap=plt.cm.Reds,
                      edge_vmin=g_min,
                      edge_vmax=g_max,
                      with_labels=True,   #Set TRUE to see node numbers
                      font_size=6,)
    
#%% DATASET LOAD & DISPLAY
    
##############################################################################
############################### USER SETUP ###################################
##############################################################################
    
num_digits_train = 60000
num_digits_test = 10000
start_train = 0
start_test = 0

##############################################################################
##############################################################################
##############################################################################

digit_rows_no_chop = 28 
digit_cols_no_chop = 28

digit_rows = 196 
digit_cols = 4

file_train = './raw_data/mnist_train.csv'
file_test = './raw_data/mnist_test.csv'

read_train = pd.read_csv(file_train)
read_train = read_train.to_numpy()
read_test = pd.read_csv(file_test)
read_test = read_test.to_numpy()

digit_train_no_border= [[] for i in range(0, num_digits_train)]
digit_test_no_border = [[] for i in range(0, num_digits_test)]

digit_train_no_chop = [np.zeros(shape=(digit_rows_no_chop, digit_cols_no_chop)) for i in range(0, num_digits_train)]
digit_test_no_chop = [np.zeros(shape=(digit_rows_no_chop, digit_cols_no_chop)) for i in range(0, num_digits_test)]

digit_train = [[np.zeros(shape=(digit_rows, digit_cols))] for i in range(0, num_digits_train)]
digit_test = [[np.zeros(shape=(digit_rows, digit_cols))] for i in range(0, num_digits_test)]

digit_train_class = read_train[0+start_train:num_digits_train+start_train, 0]
for i in range(0, num_digits_train):
    digit_train_no_chop[i] = np.reshape(np.round(read_train[i+start_train][1:]/256), (28, 28))
    digit_train[i] = np.reshape(digit_train[i], (digit_rows,digit_cols))
    for j in range(digit_cols_no_chop//digit_cols):
        digit_train[i][digit_rows_no_chop*j:digit_rows_no_chop*(j+1), :] = digit_train_no_chop[i][:, digit_cols*j:digit_cols*(j+1)]
    
digit_test_class = read_test[0+start_test:num_digits_test+start_test, 0]
for i in range(0, num_digits_test):
    digit_test_no_chop[i] = np.reshape(np.round(read_test[i+start_test][1:]/256), (28, 28))
    digit_test[i] = np.reshape(digit_test[i], (digit_rows,digit_cols))
    for j in range(digit_cols_no_chop//digit_cols):
        digit_test[i][digit_rows_no_chop*j:digit_rows_no_chop*(j+1), :] = digit_test_no_chop[i][:, digit_cols*j:digit_cols*(j+1)]
    

#%% NETWORK PROPERTIES DEFINITION

kp0 = 2.555173332603108574e-06
kd0 = 6.488388862524891465e+01
eta_p = 3.492155165334443012e+01
eta_d = 5.590601016803570467e+00
g_min = 1.014708121672117710e-03
g_max = 2.723493729125820492e-03
g0 = g_min


frame = 1 #number of frame rows/columns
inter_nodes = 1

ydim = frame*2+14+13*inter_nodes  # graph dimension
xdim = frame*2+14+13*inter_nodes

pad_rows = [[] for i in range(14)]

pad_rows[0] = [(frame+1)*(xdim-1)+(inter_nodes+1)*xdim*i for i in range(14)]
for i in range(1,14):
    pad_rows[i] = [pad_rows[0][j]-(inter_nodes+1)*i for j in range(14)]
    
src = []
for sublist in pad_rows:
    for item in sublist:
        src.append(item)

bias_nodes = [src[105]]
read_nodes = src[0:105]+src[106:]
new_nodes = [xdim*ydim+nn for nn in range(196)]
gnd = [new_nodes[-1]+1]
#%% NETWORK INPUTs

####### CUSTOMIZE YOUR PULSE SHAPE ###############
R_read = [82]*196
V_read = 100e-3
pulse_amplitude = 5 # Volts

delta_pot = 250e-6 # distance for potentiation points
delta_dep = 250e-6 # distance for depression points
delta_read = delta_dep
delta = 250e-6 # distance for transition from low to high signal and viceversa 

pulse_time = 10e-3-delta
idle_time = 5e-4-delta 
read_time = 5.5e-3 #  seconds between write and read

###################################################
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

wave_test = [0]*(idle_timesteps+1)+[1]*(pulse_timesteps+1)+[0]*(idle_timesteps+1)
wave_test = wave_test*4
wave_test = wave_test+[0]*(read_timesteps+1)

int_point = [[] for ip in range(digit_cols+1)]

for ip in range(1, digit_cols+2):
    int_point[ip-1] = one_pulse*ip+idle_timesteps 

int_point[-1] = timesteps_write+read_timesteps

time_test = [i*delta_pot for i in range(len(wave_test))]
plt.figure()
plt.plot(time_test, wave_test, '*b')
plt.plot(np.array(time_test)[int_point], [0]*len(int_point), '*r')

newpath = r'./output_MNIST_graph/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
#%% TRAIN digits

def train_mp(digit):

    G = define_grid_graph_2(xdim, ydim)
    G = initialize_graph_attributes(G, g0) 
    
    input_digit = int(digit_train_class[digit])
    
    H_list_write = [[] for t in range(0, timesteps_write+read_timesteps+1)]
    
    train_pulse, _ = dataset_to_pulse(digit_rows, digit_cols, timesteps_write, pulse_timesteps+1, idle_timesteps+1, digit_train, digit, pulse_amplitude)
    
    Vin_list_write = [[] for t in range(0, timesteps_write)]

    for t in range(0, timesteps_write):
        for r in range(0, digit_rows):
            if train_pulse[r][t] == 0:
                Vin_list_write[t] = list(Vin_list_write[t])+[int(src[r] in bias_nodes)*V_read]
            else:
                Vin_list_write[t] = list(Vin_list_write[t])+[int(src[r] in bias_nodes)*V_read+np.multiply(pulse_amplitude,(train_pulse[r][t]))]

    # WRITE
    insert_R_to_graph(G, R_read, src, new_nodes, gnd)
    
    H_list_write[0] = mod_voltage_node_analysis(G, Vin_list_write[0], new_nodes, gnd)
    
    for i in range(1, timesteps_write):
            
        delta_t = time_write[i] - time_write[i-1]
        
        remove_R_from_graph(G, src, new_nodes, gnd)
        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)                                  #update edges
        
        if i in int_point:
            fname = './output_MNIST_graph/'+'train_'+str(digit+start_train)+'_t_'+str(int_point.index(i))+'.txt'
            pickle.dump(G, open(fname, 'wb'))
            
        insert_R_to_graph(G, R_read, src, new_nodes, gnd)
       
        H_list_write[i] = mod_voltage_node_analysis(G, Vin_list_write[i], new_nodes, gnd)
    
    for i in range(timesteps_write, timesteps_write+read_timesteps+1):
    
        delta_t = delta_read
        
        remove_R_from_graph(G, src, new_nodes, gnd)        
        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)                                  #update edges
        
        if i in int_point:
            fname = './output_MNIST_graph/'+'train_'+str(digit+start_train)+'_t_'+str(int_point.index(i))+'.txt'
            pickle.dump(G, open(fname, 'wb'))
                
        insert_R_to_graph(G, R_read, src, new_nodes, gnd)
        
        H_list_write[i] = mod_voltage_node_analysis(G, Vin_list_write[-1], new_nodes, gnd)
    
    print('Train Digit '+str(input_digit)+' completed'+' ('+str(np.round((digit+1)/num_digits_train*100,2))+'%)')

if __name__ == '__main__':
    k = multiprocessing.cpu_count()
    mypool = multiprocessing.Pool(k-1)
    mypool.map(train_mp, range(0, num_digits_train))
    

#%% TEST digits

def test_mp(digit):
    
    G = define_grid_graph_2(xdim, ydim)
    G = initialize_graph_attributes(G, g0) 
    
    input_digit = int(digit_test_class[digit])
    
    H_list_write = [[] for t in range(0, timesteps_write+read_timesteps+1)]
    
    test_pulse, _ = dataset_to_pulse(digit_rows, digit_cols, timesteps_write, pulse_timesteps+1, idle_timesteps+1, digit_test, digit, pulse_amplitude)
    
    Vin_list_write = [[] for t in range(0, timesteps_write)]

    for t in range(0, timesteps_write):
        for r in range(0, digit_rows):
            if test_pulse[r][t] == 0:
                Vin_list_write[t] = list(Vin_list_write[t])+[int(src[r] in bias_nodes)*V_read]
            else:
                Vin_list_write[t] = list(Vin_list_write[t])+[int(src[r] in bias_nodes)*V_read+np.multiply(pulse_amplitude,(test_pulse[r][t]))]

    # WRITE
    insert_R_to_graph(G, R_read, src, new_nodes, gnd)
    
    H_list_write[0] = mod_voltage_node_analysis(G, Vin_list_write[0], new_nodes, gnd)
    
    for i in range(1, timesteps_write):
            
        delta_t = time_write[i] - time_write[i-1]
        
        remove_R_from_graph(G, src, new_nodes, gnd)
        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)                                  #update edges
        
        if i in int_point:
            fname = './output_MNIST_graph/'+'test_'+str(digit+start_test)+'_t_'+str(int_point.index(i))+'.txt'
            pickle.dump(G, open(fname, 'wb'))
            
        insert_R_to_graph(G, R_read, src, new_nodes, gnd)
       
        H_list_write[i] = mod_voltage_node_analysis(G, Vin_list_write[i], new_nodes, gnd)
    
    for i in range(timesteps_write, timesteps_write+read_timesteps+1):
    
        delta_t = delta_read
        
        remove_R_from_graph(G, src, new_nodes, gnd)        
        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)                                  #update edges
        
        if i in int_point:
            fname = './output_MNIST/'+'test_'+str(digit+start_test)+'_t_'+str(int_point.index(i))+'.txt'
            pickle.dump(G, open(fname, 'wb'))
                
        insert_R_to_graph(G, R_read, src, new_nodes, gnd)
        
        H_list_write[i] = mod_voltage_node_analysis(G, Vin_list_write[-1], new_nodes, gnd)
       
    print('Test Digit '+str(input_digit)+' completed'+' ('+str(np.round((digit+1)/num_digits_test*100,2))+'%)')


if __name__ == '__main__':
    k = multiprocessing.cpu_count()
    mypool = multiprocessing.Pool(k-1)
    mypool.map(test_mp, range(0, num_digits_test))
