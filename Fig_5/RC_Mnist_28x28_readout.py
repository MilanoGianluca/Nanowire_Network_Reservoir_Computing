#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#import user-defined functions
from functions_reservoir import remove_R_from_graph
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
    # # ax.set_title("".format(round(1, 6)))

#%%
    
out_dir_5 = './out_data/Fig5/'
if not os.path.exists(r'./out_data/Fig5/'):
    os.makedirs(r'./out_data/Fig5/')

#%% DATASET LOAD & DISPLAY
new_training = 0 # if put to 1, output of readout may differ a bit from the paper ones due to randomicity of the process

num_digits_train = 60000
num_digits_test = 10000
start_train = 0
start_test = 0

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

#%%    

time_istant = [0, 1, 2, 4]
for i in range(18, 19):
    for t_int in time_istant:
        fname = './raw_data/'+'train_'+str(i)+'_t_'+str(t_int)+'.txt'
        G_read = pickle.load(open(fname, 'rb'))
        plot(G_read)
        fname = out_dir_5+'train_digit_'+str(i)+'_reservoir_evolution_t_int_'+str(t_int)+'.png'
        plt.savefig(fname)
#%% NN
if new_training == 1:
    
    epochs = 800
    batch_size = 1500
    num_digits_train = 60000
    num_digits_test = 10000
    start_train= 0
    start_test = 0
    
    direct = './raw_data/'
    
    hist_train = np.loadtxt(direct+'hist_train.txt')
    hist_test = np.loadtxt(direct+'hist_test.txt')
    
    train_in = hist_train[start_train:start_train+num_digits_train,1:]
    test_in = hist_test[start_test:start_test+num_digits_test,1:]
    
    train_out = np.reshape(digit_train_class[start_train:start_train+num_digits_train], (num_digits_train,1))
    test_out = np.reshape(digit_test_class[start_test:start_test+num_digits_test], (num_digits_test,1))
    
    sc = StandardScaler()
    ohe = OneHotEncoder()
    
    scaler = sc.fit(train_in)
    train_in = scaler.transform(train_in)
    
    train_out = ohe.fit_transform(train_out).toarray()
    
    model = Sequential()
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(train_in, train_out, epochs=epochs, batch_size=batch_size)
        
    plt.figure()
    plt.plot(history.history['accuracy'], 'b', linewidth=2)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.1])
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train'], loc='upper left')

    plt.figure()
    plt.plot(history.history['loss'], 'b', linewidth=2)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train'], loc='upper left')
    
    test_in = scaler.transform(test_in)

    for layer in model.layers:
        weights = layer.get_weights()
        
    y_pred = model.predict(test_in)    
    predicted = 0
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
        if np.argmax(y_pred[i]) == test_out[i]:
            predicted += 1
    print(pred)   
    print('accuracy = '+str(predicted/len(y_pred)))
    
    conf_matrix_norm=confusion_matrix(test_out, pred, normalize='true')
    conf_matrix=confusion_matrix(test_out, pred, normalize=None)
    
    plt.figure(figsize=(18,9))
    plt.title('Confusion Matrix', fontsize=25)
    plt.imshow(conf_matrix_norm, cmap='Blues')
    plt.colorbar(label='%')
    plt.xlabel('PREDICTED', fontsize=20)
    plt.ylabel('TRUE', fontsize=20)
    plt.xticks(range(10), fontsize = 15)
    plt.yticks(range(10), fontsize = 15, rotation=90)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    for i in range(10):
        for j in range(10):
            if round(conf_matrix[i, j],0) != 0:
                text = plt.text(j, i, round(conf_matrix[i, j],0),
                                ha="center", va="center", color="w", fontsize=15)
    plt.savefig(out_dir_5+'conf_matrix.png')
            
    for layer in model.layers:
        weights = layer.get_weights()
    np.savetxt(out_dir_5+'conf_matrix_ep'+str(epochs)+'_batch'+str(batch_size)+'.txt', conf_matrix)
    np.savetxt(out_dir_5+'conf_matrix_norm_ep'+str(epochs)+'_batch'+str(batch_size)+'.txt', conf_matrix_norm)
    
elif new_training == 0:
    backup = './training_data_backup/'
    conf_matrix = np.loadtxt(backup+'conf_matrix_ep800_batch1500.txt')
    conf_matrix_norm = np.loadtxt(backup+'conf_matrix_norm_ep800_batch1500.txt')
    train_loss = np.loadtxt(backup+'train_loss_ep800_batch1500.txt')
    train_accuracy = np.loadtxt(backup+'train_accuracy_ep800_batch1500.txt')
    pred = np.loadtxt(backup+'predicted_28x28.txt')
    direct = './raw_data/'
    
    hist_train = np.loadtxt(direct+'hist_train.txt')
    hist_test = np.loadtxt(direct+'hist_test.txt')
    
    plt.figure()
    plt.plot(train_accuracy, 'b', linewidth=2)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.1])
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train'], loc='upper left')
    
    plt.figure()
    plt.plot(train_loss, 'b', linewidth=2)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train'], loc='upper left')
    
    plt.figure(figsize=(18,9))
    plt.title('Confusion Matrix', fontsize=25)
    plt.imshow(conf_matrix_norm, cmap='Blues')
    plt.colorbar(label='%')
    plt.xlabel('PREDICTED', fontsize=20)
    plt.ylabel('TRUE', fontsize=20)
    plt.xticks(range(10), fontsize = 15)
    plt.yticks(range(10), fontsize = 15, rotation=90)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    for i in range(10):
        for j in range(10):
            if round(conf_matrix[i, j],0) != 0:
                text = plt.text(j, i, int(conf_matrix[i, j]),
                                ha="center", va="center", color="w", fontsize=15)
    plt.savefig(out_dir_5+'conf_matrix.png')       

#%% digit 6 hist
    
num_digit_to_plot = 1
n_to_plot = 1
m_to_plot= 1
plt.figure(figsize=(12,10))    
for i in range(18,19):
    plt.subplot(n_to_plot,m_to_plot,1)
    plt.bar(range(1,len(hist_train[i,1:])+1), hist_train[i,1:], color='red', alpha=1)
    plt.ylim([np.min(hist_train[:,1:]), np.max(hist_train[:,1:])])
    plt.xticks([0, 65, 130, 195], fontsize = 15)
    if i >= (n_to_plot-1)*m_to_plot:
        plt.xlabel('Neuron', fontsize = 15)
    if i%m_to_plot != 0:
        plt.yticks([],fontsize = 15)
    else:
        plt.yticks(fontsize = 15)
        plt.ylabel('Voltage [V]', fontsize = 15)
    plt.title('Digit: '+str(digit_train_class[i]), fontsize = 20)
plt.savefig(out_dir_5+'hist_train_digit_6.png')