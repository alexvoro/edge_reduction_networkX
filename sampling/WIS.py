import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import datetime
import random
import bisect
import itertools


def WIS_reduce_test(graph, sizes, weight_attr):   # weighted    
    #print(nx.info(graph))
    n = graph.number_of_nodes() 
    cum_weights = [0]*n
    tot_weight = 0
    index_count = 0
    nodes = [0]*n
    node_indexes = []
    node_prob = [] 
 
    for i,v in enumerate(list(graph.nodes(data=True))): 
        tot_weight += graph.degree(v[0], weight=weight_attr) 
        cum_weights[i] = graph.degree(v[0], weight=weight_attr)
        node_indexes.append(i)
        nodes[i] = v[0]
        node_prob.append(graph.degree(v[0], weight=weight_attr))  
    
    node_prob[:] = [x / tot_weight for x in node_prob]  
      
    edge_cuts_percentage = []
    total_weight = []
    in_degree = []
    out_degree = []
    running_time = []
    average_clustering = []
    nn = []
    ne = []
    wcc = []

    for size in sizes:
        print("size", size)
        c = 0 
        integers = []
        while(c < size):
            if c >= size: return  
            i = np.random.choice(node_indexes, p=node_prob) 
            c = c + 1 
            if nodes[i] not in integers:
                integers.append(nodes[i]) 
        G_reduced = graph.copy()
        nodes_to_remove = [x for x in G_reduced.nodes if x not in integers] 
        G_reduced.remove_nodes_from(nodes_to_remove)
        actual_edge_cut = 1 - (graph.number_of_edges() - G_reduced.number_of_edges()) / graph.number_of_edges() 
        
        edge_cuts_percentage.append(actual_edge_cut) 
        total_weight.append(G_reduced.size(weight=weight_attr))
        
        in_degree.append(get_in_degree(G_reduced))
        out_degree.append(get_out_degree(G_reduced))
        #average_clustering.append(nx.average_clustering(G_reduced.to_undirected()))
        nn.append(G_reduced.number_of_nodes())
        ne.append(G_reduced.number_of_edges())
        wcc.append(len(list(nx.weakly_connected_component_subgraphs(G_reduced))))

    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc

def get_in_degree(G):
    if (len(G) == 0):
        return len(G)
    return (sum(d for n, d in G.in_degree())/float(len(G)))

def get_out_degree(G): 
    if(len(G) == 0):
        return len(G)
    return (sum(d for n, d in G.out_degree())/float(len(G)))
    
def WIS_test(G, edge_percentages, weight_attr='transferred'): 
    edge_cuts= [int(G.number_of_nodes() * x) for x in edge_percentages] 
    G_r = G.copy()
    edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc = WIS_reduce_test(G_r, edge_cuts, weight_attr=weight_attr)
    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc


def WIS_reduce_test_with_graphs(graph, sizes, weight_attr):   # weighted    
    #print(nx.info(graph))
    n = graph.number_of_nodes() 
    cum_weights = [0]*n
    tot_weight = 0
    index_count = 0
    integers = []
    nodes = [0]*n
    node_indexes = []
    node_prob = [] 
 
    for i,v in enumerate(list(graph.nodes(data=True))): 
        tot_weight += graph.degree(v[0], weight=weight_attr) 
        cum_weights[i] = graph.degree(v[0], weight=weight_attr)
        node_indexes.append(i)
        nodes[i] = v[0]
        node_prob.append(graph.degree(v[0], weight=weight_attr))  
    
    node_prob[:] = [x / tot_weight for x in node_prob]  
      
    edge_cuts_percentage = []
    total_weight = []
    in_degree = []
    out_degree = []
    running_time = []
    average_clustering = []
    nn = []
    ne = []
    wcc = []
    graphs = []

    for size in sizes:
        print("size", size)
        c = 0 
        while(c < size):
            if c >= size: return  
            i = np.random.choice(node_indexes, p=node_prob) 
            c = c + 1 
            if nodes[i] not in integers:
                integers.append(nodes[i]) 
        G_reduced = graph.copy()
        nodes_to_remove = [x for x in G_reduced.nodes if x not in integers] 
        G_reduced.remove_nodes_from(nodes_to_remove)
        actual_edge_cut = 1 - (graph.number_of_edges() - G_reduced.number_of_edges()) / graph.number_of_edges() 
        
        edge_cuts_percentage.append(actual_edge_cut) 
        total_weight.append(G_reduced.size(weight=weight_attr))
        
        in_degree.append(get_in_degree(G_reduced))
        out_degree.append(get_out_degree(G_reduced))
        average_clustering.append(nx.average_clustering(G_reduced.to_undirected()))
        nn.append(G_reduced.number_of_nodes())
        ne.append(G_reduced.number_of_edges())
        wcc.append(len(list(nx.weakly_connected_component_subgraphs(G_reduced))))
        graphs.append(G_reduced)
    
    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, graphs


def WIS_test_with_graph(G, edge_percentages, weight_attr='transferred'): 
    edge_cuts= [int(G.number_of_nodes() * x) for x in edge_percentages] 
    G_r = G.copy()
    edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, graphs = WIS_reduce_test_with_graphs(G_r, edge_cuts, weight_attr=weight_attr)
    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, graphs
