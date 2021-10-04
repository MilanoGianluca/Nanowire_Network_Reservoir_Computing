#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import multiprocessing
import pickle
import numpy as np
#import user-defined functions
from functions import mod_voltage_node_analysis
from functions_reservoir import insert_R_to_graph

#%%
newpath = r'./output_MNIST_hist/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

#%% 
num_digits_train = 60000
num_digits_test = 10000
start_train = 0
start_test = 0

#%% NETWORK PROPERTIES DEFINITION

# from north fitting, november measurement
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

R_read = [82]*196
V_read = 100e-3
pulse_amplitude = 5 # Volts

#%% read hist from G maps

bias_idx = [105]
read_idx = list(range(105))+list(range(106,196))

bias_nodes = list(np.array(src)[bias_idx])
read_nodes = list(np.array(src)[read_idx])

V_list_read = []
src_vec = []
for n in range(196):
    if src[n] in bias_nodes:
        V_list_read += [V_read]
        src_vec += [new_nodes[n]]
    elif src[n] in read_nodes:
        V_list_read += [0]
        src_vec += [new_nodes[n]]

def hist_train_fun(digit):
    
    t_int = 4
    hist_train = np.zeros((len(read_nodes)+1,))

    directory = './output_MNIST_graph/'        
    
    fname = directory+'train_'+str(digit+start_train)+'_t_'+str(t_int)+'.txt'
    G_read = pickle.load(open(fname, 'rb'))
    
    insert_R_to_graph(G_read, R_read, src, new_nodes, gnd)
    H_read = mod_voltage_node_analysis(G_read, V_list_read, src_vec, gnd)
    for n in range(len(read_nodes)):
        hist_train[n+1] = H_read.nodes[read_nodes[n]]['V']
                
    hist_train[0] = digit+start_train
    print(str(digit)+'/'+str(num_digits_train))        
    return hist_train  

def hist_test_fun(digit):
    
    t_int = 4
    hist_test = np.zeros((len(read_nodes)+1,))

    directory = './output_MNIST_graph/'
    
    fname = directory+'test_'+str(digit+start_test)+'_t_'+str(t_int)+'.txt'
    G_read = pickle.load(open(fname, 'rb'))
                
    insert_R_to_graph(G_read, R_read, src, new_nodes, gnd)
    H_read = mod_voltage_node_analysis(G_read, V_list_read, src_vec, gnd)
    for n in range(len(read_nodes)):
        hist_test[n+1] = H_read.nodes[read_nodes[n]]['V']
    hist_test[0] = digit+start_test
    print(str(digit)+'/'+str(num_digits_test))        
    return hist_test


if __name__ == '__main__':
    k = multiprocessing.cpu_count()
    
    mypool = multiprocessing.Pool(k-1)
    d = np.array(mypool.map(hist_train_fun, range(num_digits_train)))
    idx = np.argsort(d[:,0])
    d = d[idx,:]
    np.savetxt('./output_MNIST_hist/hist_train.txt', d)

    mypool = multiprocessing.Pool(k-1)
    d = np.array(mypool.map(hist_test_fun, range(num_digits_test)))
    idx = np.argsort(d[:,0])
    d = d[idx,:]
    np.savetxt('./output_MNIST_hist/hist_test.txt', d)
    