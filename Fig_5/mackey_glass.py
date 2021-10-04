#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import datetime
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import signalz
#  import time as t
#import user-defined functions
from functions import define_grid_graph_2,initialize_graph_attributes, mod_voltage_node_analysis, update_edge_weigths_2
from functions_reservoir import insert_R_to_graph, remove_R_from_graph

#%% SETUP
    
g0_std = 0 # percentage with respect to its mean value
gmax_std = 0 # percentage with respect to its mean value
eta_p_std = 0 # percentage with respect to its mean vale
eta_d_std = 0 # percentage with respect to its mean value
round_decimal = 2

beta = 0.2
gamma = 0.9
tau = 18
n = 10

num_virtual_nodes = 20
steps_training = 900+num_virtual_nodes
steps_forecasting = 1200 
init_duration = 400
mg_boundary = 0.6
update_active = 0
init_active = 1 
update_steps_on = 100
update_steps_off = 200

#%% NETWORK PARAMETER

# network prameters
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
seed = 2

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

R = [82]*196
V_read = 100e-3
pulse_amplitude_min = 1 # Volts
pulse_amplitude_max = 6 # Volts

#%% NETWORK STIMULATION FOR TRAINING

delta_t_pot = 1e-3 #seconds
delta_t_dep = 2e-3 #seconds
pot_points = 4 
dep_points = 2

steps_discard_initial_training = num_virtual_nodes

current_date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
current_date_and_time_string = str(current_date_and_time)
folder = r'./out_data/Fig5/mackey_glass/simulation_'+current_date_and_time_string+'/' 
directory = './out_data/Fig5/mackey_glass/simulation_'+current_date_and_time_string+'/' 
if not os.path.exists(folder):
    os.makedirs(folder)
        
try:
    
    if num_virtual_nodes > steps_discard_initial_training:
        raise Exception("ERROR: num virtual nodes too high with respect to steps_discard_initial_training")
    
    steps = steps_training+steps_forecasting
    
    mg_target = signalz.mackey_glass(steps, a=beta, b=0.8, c=gamma, d=tau, e=n, initial=mg_boundary)
    mg_target_norm = (mg_target-min(mg_target))/((max(mg_target)-min(mg_target)))
    
    
    plt.figure()
    plt.title("Mackey Glass Numeric Solution", fontsize = 20)
    plt.plot(mg_target_norm, 'b', linewidth = 2) 
    plt.xlabel("steps", fontsize = 15)  


    pulse_stream = pulse_amplitude_min+mg_target_norm*(pulse_amplitude_max-pulse_amplitude_min)
    H_train = [[] for s in range(steps_training)]
    H_train_read = [[] for s in range(steps_training)]
    output_train = np.zeros((steps_training, len(new_nodes)))
    
    G = define_grid_graph_2(xdim, ydim)
    initialize_graph_attributes(G, g0)
    
    for s in range(steps_training):
        pulse_stream[s] = round(pulse_stream[s], round_decimal)
         
        input_list = []
        
        # selecting pads to stimulate --> chessboard fashion
        for count in range(196):
            if ((count%2 == 0) & ((count//14)%2 == 0)) or ((count%2 == 1) & ((count//14)%2 == 1)) :
                input_list += [pulse_stream[s]]
            else:
                input_list += [0]
               
        for pp in range(pot_points):
            insert_R_to_graph(G, R, src, new_nodes, gnd)
            H_train[s] = mod_voltage_node_analysis(G, input_list, new_nodes, gnd)
    
            remove_R_from_graph(G, R, new_nodes, gnd)
            G = update_edge_weigths_2(G, delta_t_pot/pot_points, g_min, g_min*g0_std, g_max, g_max*gmax_std, kp0, eta_p, eta_p*eta_p_std, kd0, eta_d, eta_d*eta_d_std)
    
        for p in range(len(src)):
            output_train[s, p] = H_train[s].nodes[src[p]]['V']
            
        print(str(s)+"/"+str(steps_training)+ " (training)")
        
    
    #%% TRAINING
        
    # selecting pads for output reading --> chessboard fashion (as in stimulation)
    read_idx = []  
    for count in range(196):
        if not(((count%2 == 0) & ((count//14)%2 == 0)) or ((count%2 == 1) & ((count//14)%2 == 1))) :
            read_idx += [count]
        
    train_in = np.zeros((steps_training-steps_discard_initial_training-1, len(read_idx)+len(read_idx)*num_virtual_nodes))
    
    for s in range(len(train_in)):
        train_in[s,:] = np.hstack((np.reshape(output_train[steps_discard_initial_training+s, read_idx], (1,len(read_idx))), np.reshape(output_train[steps_discard_initial_training+s-num_virtual_nodes:steps_discard_initial_training+s, read_idx], (1, len(read_idx)*num_virtual_nodes))))
    
    train_out = np.zeros((steps_training-1-steps_discard_initial_training,1))
    train_out[:,0] = np.array(mg_target[steps_discard_initial_training+1:steps_training])
    
    reg = LinearRegression().fit(train_in, train_out)
    pred_train = reg.predict(train_in)
    
    plt.figure()
    plt.title('Training', fontsize = 20)
    plt.plot(train_out, 'b', label = 'true')
    plt.plot(pred_train, 'r', label='predicted')
    plt.xlabel('Time-step', fontsize = 20)
    plt.ylabel('Value', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)  
    plt.legend(fontsize = 15)   
    plt.grid()
    
    plt.savefig(directory+'plot_training.png')
    np.savetxt(directory+'plot_training.txt', np.hstack((np.reshape(pred_train, (len(pred_train),1)), np.reshape(train_out, (len(train_out),1)))))
    
    coef = reg.coef_
    
    #%% NETWORK STIMULATION FOR INITIALIZATION

    if init_active:
        G_new = define_grid_graph_2(xdim, ydim)
        initialize_graph_attributes(G_new, g0)
        output_new = np.zeros((steps_training+steps_forecasting, len(src)))
        H_new = [[] for s in range(steps_training+steps_forecasting)]
        H_new_init = [[] for s in range(init_duration)]
        output_new_init = np.zeros((init_duration, len(src)))
        
        for init in range(init_duration):
            
            input_list_target = []
            pulse_stream[init+steps_training-init_duration] = round(pulse_stream[init+steps_training-init_duration], round_decimal)
        
            for count in range(196):
                if ((count%2 == 0) & ((count//14)%2 == 0)) or ((count%2 == 1) & ((count//14)%2 == 1)) :
                    input_list_target += [pulse_stream[init+steps_training-init_duration]]
                else:
                    input_list_target += [0]
                
            for pp in range(pot_points):
                insert_R_to_graph(G_new, R, src, new_nodes, gnd)
                H_new_init[init] = mod_voltage_node_analysis(G_new, input_list_target, new_nodes, gnd)
                remove_R_from_graph(G_new, R, new_nodes, gnd)
                G_new = update_edge_weigths_2(G, delta_t_pot/pot_points, g_min, g_min*g0_std, g_max, g_max*gmax_std, kp0, eta_p, eta_p*eta_p_std, kd0, eta_d, eta_d*eta_d_std)
            for p in range(len(src)):
                output_new_init[init, p] = H_new_init[init].nodes[src[p]]['V']
            print(str(init)+"/"+str(init_duration)+ " (initialization)") 

        H_new[steps_training-init_duration:steps_training] = H_new_init
        output_new[steps_training-init_duration:steps_training, :] = output_new_init
    else:
        G_new = G
        H_new = [[] for s in range(steps_training+steps_forecasting)]
        H_new[:steps_training] = H_train
        output_new = np.zeros((steps_training+steps_forecasting, len(src)))
        output_new[:steps_training, :] = output_train

    
    #%% FORECASTING
    counter = 0
    
    for s in range(steps_training, steps_training+steps_forecasting):
        
        train_in_predict = np.zeros((1, len(read_idx)+len(read_idx)*num_virtual_nodes))
        
        train_in_predict[0,:] = np.hstack((np.reshape(output_new[s-1, read_idx], (1, len(read_idx))), np.reshape(output_new[s-2-num_virtual_nodes:s-2, read_idx], (1, len(read_idx)*num_virtual_nodes))))
    
        pulse_stream_new = reg.predict(train_in_predict)[0][0]
    
        pulse_stream_new = (pulse_stream_new-min(mg_target))/((max(mg_target)-min(mg_target)))
        pulse_stream_new = pulse_amplitude_min+pulse_stream_new*(pulse_amplitude_max-pulse_amplitude_min)
        pulse_stream_new = round(pulse_stream_new, round_decimal)
        
        input_list = []
        
        for count in range(196):
            if ((count%2 == 0) & ((count//14)%2 == 0)) or ((count%2 == 1) & ((count//14)%2 == 1)) :
                input_list += [pulse_stream_new]
            else:
                input_list += [0]
        
        pulse_stream[s] = round(pulse_stream[s], round_decimal)
         
        input_list_target = []
        
        for count in range(196):
            if ((count%2 == 0) & ((count//14)%2 == 0)) or ((count%2 == 1) & ((count//14)%2 == 1)) :
                input_list_target += [pulse_stream[s]]
            else:
                input_list_target += [0]
                
        for pp in range(pot_points):
            insert_R_to_graph(G_new, R, src, new_nodes, gnd)
            
                                     
            if (update_active):
                if (counter < update_steps_off): 
                    H_new[s] = mod_voltage_node_analysis(G_new, input_list, new_nodes, gnd)
                else: 
                    H_new[s] = mod_voltage_node_analysis(G_new, input_list_target, new_nodes, gnd)
            else:
                H_new[s] = mod_voltage_node_analysis(G_new, input_list, new_nodes, gnd)
            
            remove_R_from_graph(G_new, R, new_nodes, gnd)
            G_new = update_edge_weigths_2(G, delta_t_pot/pot_points, g_min, g_min*g0_std, g_max, g_max*gmax_std, kp0, eta_p, eta_p*eta_p_std, kd0, eta_d, eta_d*eta_d_std)
        
        for p in range(len(src)):
            output_new[s, p] = H_new[s].nodes[src[p]]['V']
        
        counter += 1
        if counter == update_steps_off + update_steps_on:
            counter = 0
        print(str(s-steps_training)+"/"+str(steps_forecasting)+ " (forecasting)") 
    
    #%% PLOT  
    if init_active: # include init region in the plot
        forecast_plot = np.zeros((init_duration+steps_forecasting, len(read_idx)+len(read_idx)*num_virtual_nodes))
        for s in range(len(forecast_plot)):
            forecast_plot[s,:] = np.hstack((np.reshape(output_new[steps_training-init_duration+s, read_idx], (1,len(read_idx))), np.reshape(output_new[steps_training-init_duration+s-num_virtual_nodes:steps_training-init_duration+s, read_idx], (1, len(read_idx)*num_virtual_nodes))))
    
    else: # plot only the forecast region
        forecast_plot = np.zeros((steps_forecasting, len(read_idx)+len(read_idx)*num_virtual_nodes))
        for s in range(len(forecast_plot)):
            forecast_plot[s,:] = np.hstack((np.reshape(output_new[steps_training+s, read_idx], (1,len(read_idx))), np.reshape(output_new[steps_training+s-num_virtual_nodes:steps_training+s, read_idx], (1, len(read_idx)*num_virtual_nodes)))) 
    
    train_plot = np.zeros((steps_training-steps_discard_initial_training, len(read_idx)+len(read_idx)*num_virtual_nodes))
    for s in range(len(train_plot)):
        train_plot[s,:] = np.hstack((np.reshape(output_train[steps_discard_initial_training+s, read_idx], (1,len(read_idx))), np.reshape(output_train[steps_discard_initial_training+s-num_virtual_nodes:steps_discard_initial_training+s, read_idx], (1, len(read_idx)*num_virtual_nodes))))
    
    pred_forecasting = reg.predict(forecast_plot)
    pred_train = reg.predict(train_plot)
    
    target = np.zeros((steps-steps_discard_initial_training,1))
    target[:,0] = np.array(mg_target[steps_discard_initial_training:])
    
    RMSE_train = mean_squared_error(target[0:steps_training-steps_discard_initial_training], pred_train, squared=False)
    RMSE_forecast = mean_squared_error(target[steps_training-steps_discard_initial_training:], pred_forecasting[init_duration*init_active:], squared=False)
        
    plt.figure(figsize=(18,9))
    plt.title('Forecasting', fontsize = 20)
    plt.plot(target[steps_training-steps_discard_initial_training-init_active*init_duration:], 'b', label = 'true')
    plt.plot(pred_forecasting[:], 'r', label='predicted')
    if init_active:
        plt.axvline(x=init_duration, color = 'black', linewidth=2)
    plt.xlabel('Time-step', fontsize = 20)
    plt.ylabel('Value', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)  
    plt.legend(fontsize = 15)   
    plt.grid()
    plt.savefig(directory+'plot_forecast.png')
    
    np.savetxt(directory + 'plot_forecasting.txt', np.hstack((pred_forecasting[:], target[steps_training-steps_discard_initial_training-init_active*init_duration:])))
    np.savetxt(directory+'output_train.txt', output_train)
    np.savetxt(directory+'output_new.txt', output_new)
    np.savetxt(directory+'coef.txt', coef)
    np.savetxt(directory+'target.txt', target)
    np.savetxt(directory+'predicted.txt', pred_forecasting)
    if init_active:
        np.savetxt(directory+'output_new_init.txt', output_new_init)
    
    original_stdout = sys.stdout
    file_name =  'setup.txt'
    file = open(directory+file_name, 'w')
    
    sys.stdout = file 
    print('SIMULATION SETUP\n')
    
    print('tau = ' +  str(tau))
    print('num_virtual_nodes = ' +str(num_virtual_nodes))
    print('steps_training = ' +  str(steps_training))
    print('steps_forecasting = ' +  str(steps_forecasting))
    print('init_active = ' +  str(init_active))
    print('update_active = ' +  str(update_active))
    print('init_duration = ' +  str(init_duration))
    print('update_steps_on = ' +  str(update_steps_on))
    print('update_steps_off = ' +  str(update_steps_off))
    print('pulse_amplitude_min = ' +  str(pulse_amplitude_min))
    print('pulse_amplitude_max = ' +  str(pulse_amplitude_max))
    print('delta_t_pot = ' +  str(delta_t_pot))
    print('\n')
    print('TRAIN RMSE: '+str(RMSE_train))
    print('FORECAST RMSE: '+str(RMSE_forecast))
    print('\n')
    print('Accuracy Train: '+str(1-RMSE_train))
    print('Accuracy Forecast: '+str(1-RMSE_forecast))
    
    file.close()
    sys.stdout = original_stdout 
    
except Exception as e:

    original_stdout = sys.stdout
    file_name =  'error.txt'
    file = open(directory+file_name, 'w')
    
    sys.stdout = file 
    
    print("ERROR")
    print(e)
    print('\n')
    print('SIMULATION SETUP\n')
    
    print('tau = ' +  str(tau))
    print('num_virtual_nodes = ' +str(num_virtual_nodes))
    print('steps_training = ' +  str(steps_training))
    print('steps_forecasting = ' +  str(steps_forecasting))
    print('init_active = ' +  str(init_active))
    print('update_active = ' +  str(update_active))
    print('init_duration = ' +  str(init_duration))
    print('update_steps_on = ' +  str(update_steps_on))
    print('update_steps_off = ' +  str(update_steps_off))
    print('pulse_amplitude_min = ' +  str(pulse_amplitude_min))
    print('pulse_amplitude_max = ' +  str(pulse_amplitude_max))
    print('delta_t_pot = ' +  str(delta_t_pot))
                                   
    file.close()
    sys.stdout = original_stdout 
                
            




        





