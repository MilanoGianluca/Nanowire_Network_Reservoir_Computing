#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def dataset_to_pulse(digit_rows, digit_cols, timesteps_write, pulse_timesteps, idle_timesteps, digit_list, input_digit, pulse_amplitude):
    
    Vin_list_write = [[] for t in range(0, timesteps_write)]
    train_pulse = [[[]for i in range(0, digit_cols)] for i in range(0, digit_rows)]

    bit_0 = [0]*(idle_timesteps + pulse_timesteps + idle_timesteps)
    bit_1 = [0]*idle_timesteps + [1]*pulse_timesteps + [0]*idle_timesteps
    
    for i in range(0, digit_rows):
        for j in range(0, digit_cols):
             digit_cell = digit_list[input_digit][i][j]
             cell_value = int(digit_cell == 1)
             train_pulse[i][j] = bit_0*(1-cell_value)+bit_1*(cell_value)
        train_pulse[i] = [element for item in train_pulse[i] for element in item] # make it a single list
    
    for t in range(0, timesteps_write):
        for r in range(0, digit_rows):
            if train_pulse[r][t] == 0:
                Vin_list_write[t] = list(Vin_list_write[t])+['f']
            else:
                Vin_list_write[t] = list(Vin_list_write[t])+[np.multiply(pulse_amplitude,(train_pulse[r][t]))]
    
    
    return train_pulse, Vin_list_write

def insert_R_to_graph(G, R, src_nodes, new_nodes, gnd):
    
    G.add_nodes_from(new_nodes)
    G.add_nodes_from(gnd)
    
    for i in range(len(new_nodes)):    
        G.add_edge(new_nodes[i], src_nodes[i])
        
    for u in new_nodes:
        for v in src_nodes:  
            if G.has_edge(u, v):
                idx = new_nodes.index(u)
                G[u][v]['Y'] = 1/R[idx]
    return G

def remove_R_from_graph(G, src_nodes, new_nodes, gnd):
    
    G.remove_nodes_from(new_nodes)
    G.remove_nodes_from(gnd)
    
    for u in new_nodes:
        for v in src_nodes:  
            if G.has_edge(u, v):
                G.remove_edge(u, v)
    return G
