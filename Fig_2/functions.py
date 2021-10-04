#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import exit
import math
import random
from networkx import grid_graph
import numpy as np

import networkx as nx


####################    -    GRAPH DEFINITION     -    ########################
def define_grid_graph(xdim, ydim):

    Ggrid = grid_graph(dim=[xdim, ydim])
    G = nx.convert_node_labels_to_integers(Ggrid, first_label=0, ordering='default', label_attribute='pos')

    return G

###########  -    GRAPH DEFINITION_2  (with random diagonals) -    ############


def define_grid_graph_2(xdim,ydim):

    ##define a grid graph

    Ggrid = grid_graph(dim=[xdim, ydim])
    random.seed(2)
    ##define random diagonals
    for x in range (xdim-1):
        for y in range(ydim-1):
            k = random.randint(0, 1)
            if k == 0:
                Ggrid.add_edge((x, y), (x+1, y+1))
            else:
                Ggrid.add_edge((x+1, y), (x, y+1))
            

    ##define a graph with integer nodes and positions of a grid graph
    G=nx.convert_node_labels_to_integers(Ggrid, first_label=0, ordering='default', label_attribute='pos')

    return G




########################  - GRAPH INITIALIZATION     -    #####################
    
def initialize_graph_attributes(G,Yin):
    #add the initial conductance
    for u,v in G.edges():
        G[u][v]['Y']=Yin                                                        #assign initial ammittance to all edges
        G[u][v]['Filament']= False 
        
        G[u][v]['X']=0
        G[u][v]['Xlocal']=0                                                     #assign initial high resistance state in all junctions
        G[u][v]['deltaV']=0
        G[u][v]['g']=0
        
        
    ##initialize
    for n in G.nodes():
        G.nodes[n]['pad']=False
        G.nodes[n]['source_node']= False
        G.nodes[n]['ground_node']= False
        
    return G

###############################################################################

def mod_voltage_node_analysis(G, V_list, src_list, gnd_list):
    ## MODIFIED VOlTAGE NODE ANALYSIS
    Vin_list = V_list.copy()
    sourcenode_list = src_list.copy()
    groundnode_list = gnd_list.copy()
    len_src = len(sourcenode_list)
    len_gnd = len(groundnode_list)
    
    i = 0
    while(i < len(Vin_list)):
        if Vin_list[i] == 'f':
            del Vin_list[i]
            del sourcenode_list[i]
            if len_gnd == len_src:
                del groundnode_list[i]
        else:
            i = i + 1 
            
    if len(groundnode_list) != 0:
                
        node_list = list(G.nodes()) #  real node list
        node_map = np.arange(0, G.number_of_nodes()) # real nodes mapped on consecutive integers
        num_src = len(sourcenode_list) # number of src nodes
        num_gnd = len(groundnode_list) # number of gnd nodes
        src_map = [[] for i in range(0, num_src)] # initialize src mapping from consecutive integer nodes
        gnd_map = [[] for i in range(0, num_gnd)] # initialize gnd mapping from consecutive integer nodes
        num_supply_gnd = num_gnd-1 # number of zero voltage supplies to connect gnd nodes each other
        num_supply_src = num_src # number of non-zero supply to connect between src and gnd
        num_supply = num_supply_gnd + num_supply_src
        supply_map = [[] for i in range(0, num_supply)]
        
        if len(Vin_list) is not num_src:
            print('Error: Input Voltage list and source node list must be equal in length!')
            exit()
        
        # define which nodes from integer list are src
        pos_node = -1
        for node in G.nodes():
            pos_node = pos_node + 1
            for i in range(0, num_src):
                if node == sourcenode_list[i]:
                    src_map[i] = pos_node
        # define which nodes from integer list are gnd            
        pos_node = -1
        for node in G.nodes():
            pos_node = pos_node + 1
            for i in range(0, num_gnd):
                if node == groundnode_list[i]:
                    gnd_map[i] = pos_node
                    
        # definition of matrices
        matZ = np.zeros(shape=(G.number_of_nodes()-1 + num_supply, 1))
        matG = np.zeros(shape=(G.number_of_nodes(), G.number_of_nodes()))
        matB = np.zeros(shape=(G.number_of_nodes(), num_supply))
        matD = np.zeros(shape=(num_supply, num_supply))
    
    
        # filling Y matrix as a combination of G B D in the form [(G B) ; (B' D)]
    
        # elements of G
        for k in range(0, G.number_of_nodes()):
            real_node = list(G.nodes())[k]
            real_neighs = list(G.neighbors(real_node))  # list of neighbors nodes
            k_map = k
            for m in range(0, len(real_neighs)):
                neigh_map = list(G.nodes()).index(real_neighs[m])
                matG[k_map][k_map] = matG[k_map][k_map] + G[real_node][real_neighs[m]]['Y']  # divided by 1
                if neigh_map != gnd_map[0]:
                    matG[k_map][neigh_map] = -G[real_node][real_neighs[m]]['Y']
                        
        #  for ground in gnd_map:
        matG = np.delete(matG, gnd_map[0], 0)
        matG = np.delete(matG, gnd_map[0], 1)
        
        #  matZ and matB
        pos = 0
        gnd_pos = 0
        for node in node_map:
            if node in src_map:
                src_pos = src_map.index(node)
                matZ[-num_supply + pos] = Vin_list[src_pos]
                supply_map[pos] = node
                matB[node][pos] = 1
                pos = pos + 1
            if node in gnd_map[1:]:
                matZ[-num_supply + pos] = 0
                gnd_pos = gnd_pos + 1 
                supply_map[pos] = node
                matB[node][pos] = 1
                pos = pos + 1
        
              
        matB = np.delete(matB, gnd_map[0], 0)
        
    
        # matY
        
        submat1 = np.hstack((matG, matB))
        submat2 = np.hstack((np.transpose(matB), matD))
        matY = np.vstack((submat1, submat2))
        
    
        # solve X matrix from Yx = z
        invmatY = np.linalg.inv(matY)  # inverse of matY
        
        matX = np.matmul(invmatY, matZ)  # Ohm law
    
        # add voltage as a node attribute
        
        flag = 0 #  how many times I find a ground node
        for n in range(0, G.number_of_nodes()):
            if n == gnd_map[0]:
                G.nodes[node_list[n]]['V'] = 0
                flag = flag+1
            else:
                G.nodes[node_list[n]]['V'] = matX[n-flag][0]
    
    else:
        node_list = list(G.nodes()) #  real node list
        for n in range(0, G.number_of_nodes()):
            G.nodes[node_list[n]]['V'] = 0
    ###DEFINE CURRENT DIRECTION

    # transform G to a direct graph H

    H = G.to_directed()  # transform G to a direct graph

    # add current as a node attribute

    for u, v in H.edges():
        H[u][v]['I'] = (H.nodes[u]['V'] - H.nodes[v]['V']) * H[u][v]['Y']
        H[u][v]['Irounded'] = np.round(H[u][v]['I'], 2)

    #  set current direction
    for u in H.nodes():  # select current direction
        for v in H.nodes():
            if H.has_edge(u, v) and H.has_edge(v, u):
                if H[u][v]['I'] < 0:
                    H.remove_edge(u, v)
                else:
                    H.remove_edge(v, u)
    return H

#################  - CALCULATE  NETWORK RESISTANCE     -    ####################
    

def calculate_network_resistance(H, sourcenode):
    
    I_fromsource = 0
    for u,v in H.edges(sourcenode):
        a= H[u][v]['I']
        I_fromsource=I_fromsource+a

    
    Rnetwork=H.nodes[sourcenode]['V']/I_fromsource
    
    return Rnetwork

def calculate_network_resistance_2(H, sourcenode, groundnode):
    
    V_read = 1 #  arbitrary
    H = H.to_undirected()

    H_pad = mod_voltage_node_analysis(H, [V_read], sourcenode, groundnode)
    
    I_fromsource = 0
    for u,v in H_pad.edges(sourcenode):
        a=H_pad[u][v]['I']
        I_fromsource=I_fromsource+a
    
    Rnetwork=V_read/I_fromsource
    
    return Rnetwork

###############################################################################


#######################  - CALCULATE V source     -    ########################
    
    
def calculate_Vsource(H, sourcenode):

    Vsource=H.nodes[sourcenode]['V']
    
    return Vsource

###############################################################################





#######################  - CALCULATE I source     -    ########################

def calculate_Isource(H, sourcenode):
    
    I_from_source=0
    
    for u,v in H.edges(sourcenode):
        a= H[u][v]['I']
        I_from_source=I_from_source+a
    return I_from_source

###############################################################################
    




#################  - UPDATE EDGE WEIGHT (Miranda's model)   -    ##############
    
def update_edge_weigths(G,delta_t,Y_min, Y_max,kp0,eta_p,kd0,eta_d):
    
    
    for u,v in G.edges():

        G[u][v]['deltaV']=abs(G.nodes[u]['V']-G.nodes[v]['V'])
    
        G[u][v]['kp']= kp0*math.exp(eta_p*G[u][v]['deltaV'])
        
        G[u][v]['kd']= kd0*math.exp(-eta_d*G[u][v]['deltaV'])
        
        G[u][v]['g']= (G[u][v]['kp']/(G[u][v]['kp']+G[u][v]['kd']))*(1-(1-(1+(G[u][v]['kd']/G[u][v]['kp'])*G[u][v]['g']))*math.exp(-(G[u][v]['kp']+G[u][v]['kd'])*delta_t))
    
        G[u][v]['Y']= Y_min*(1-G[u][v]['g'])+Y_max*G[u][v]['g']
        
    return G


