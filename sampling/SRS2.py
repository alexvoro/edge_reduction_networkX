import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import datetime
import random
import bisect
import itertools

def SRS2(graph, weight_attr, cut_size): 
    sorted_edges_by_weight = sorted(graph.edges.data(weight_attr),reverse=True, key=lambda x: x[2])
 
    edges_to_remove = list(sorted_edges_by_weight)[int(cut_size):] 
    graph.remove_edges_from(edges_to_remove)
    return graph

def get_in_degree(G):
    return (sum(d for n, d in G.in_degree())/float(len(G)))

def get_out_degree(G):
    return (sum(d for n, d in G.out_degree())/float(len(G)))

def SRS2_test(G, edge_cuts, weight_attr='transferred'):
    #G_r = G.copy()
    total_weight = [] 
    in_degree = []
    out_degree = []
    average_clustering = []
    nn = []
    ne = []
    wcc = [] 
    running_time = []

    for edge_cut in edge_cuts:  
        edges_max_goal = G.number_of_edges() * edge_cut 
        
        current_time = time.time()
        G_reduced = SRS2(G.copy(), weight_attr, edges_max_goal)
        time_spent = time.time()-current_time
        
        total_weight.append(G_reduced.size(weight=weight_attr)) 
        in_degree.append(get_in_degree(G_reduced))
        out_degree.append(get_out_degree(G_reduced))
        running_time.append(time_spent)
        #average_clustering.append(nx.average_clustering(G_reduced.to_undirected()))

        nn.append(G_reduced.number_of_nodes())
        ne.append(G_reduced.number_of_edges())
        wcc.append(len(list(nx.weakly_connected_component_subgraphs(G_reduced)))) 

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc

 
def SRS2_test_with_graphs(G, edge_cuts, weight_attr='transferred'):
    G_r = G.copy()
    total_weight = [] 
    in_degree = []
    out_degree = []
    average_clustering = []
    nn = []
    ne = []
    wcc = [] 
    running_time = []
    graphs = []

    for edge_cut in edge_cuts:  
        edges_max_goal = G.number_of_edges() * edge_cut 
        
        current_time = time.time()
        G_reduced = SRS2(G.copy(), weight_attr, edges_max_goal)
        time_spent = time.time()-current_time
        
        total_weight.append(G_reduced.size(weight=weight_attr)) 
        in_degree.append(get_in_degree(G_reduced))
        out_degree.append(get_out_degree(G_reduced))
        running_time.append(time_spent)
        average_clustering.append(nx.average_clustering(G_reduced.to_undirected()))

        nn.append(G_reduced.number_of_nodes())
        ne.append(G_reduced.number_of_edges())
        wcc.append(len(list(nx.weakly_connected_component_subgraphs(G_reduced))))
        graphs.append(G_reduced)

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc,graphs

 