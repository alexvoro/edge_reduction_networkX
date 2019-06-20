from __future__ import division, absolute_import, print_function
import sys
import os
# We need to import the graph_tool module itself 
# sys.path.append('/usr/local/Cellar/graph-tool/2.27_6/lib/python3.7/site-packages/')
from graph_tool.all import *  
import numpy as np
import json
import time
import datetime
import random
import bisect
import itertools 

def WIS_reduce_test(graph, sizes, weight_attr):   
    n = graph.num_vertices()
    node_indexes = np.arange(n) 

    nodes = graph.get_vertices()
    in_degrees =  np.array(graph.get_in_degrees(graph.get_vertices()))
    out_degrees =  np.array(graph.get_out_degrees(graph.get_vertices())) 
    cum_weights = np.add(in_degrees, out_degrees) 
    tot_weight = sum(cum_weights) 
    node_prob = [x / int(tot_weight) for x in cum_weights]    
 
    edge_cuts_percentage = []
    total_weight = []
    in_degree = []
    out_degree = []
    running_time = []
    average_clustering = []
    nn = []
    ne = []
    wcc = []
    assortativity = []

    for size in sizes:
        print("size", size) 
        current_time = time.time() 
        indexes = np.random.choice(node_indexes, size, replace=False, p=node_prob) 
         
        nodes_to_keep = nodes[indexes]
        nodes_to_keep = list(set(nodes_to_keep))
        
        G_reduced = Graph(graph)
        nodes_to_remove = [x for x in G_reduced.get_vertices() if x not in nodes_to_keep]

        for v in reversed(sorted(nodes_to_remove)):
            G_reduced.remove_vertex(v, fast=True)
 
        actual_edge_cut = 1 - (graph.num_edges() - G_reduced.num_edges()) / graph.num_edges()

        graph_reduced = Graph(G_reduced, prune=True)
        sim.append(graph_tool.topology.similarity(graph, graph_reduced))
     
        time_spent = time.time()-current_time
        edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, assortativity = get_stats(G_reduced, actual_edge_cut, weight_attr, edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, time_spent, assortativity)

    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, assortativity

def get_stats(G_reduced, actual_edge_cut, weight_attr, edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, time_spent, assortativity):
    edge_weight = G_reduced.edge_properties[weight_attr]
    t_weight = sum(edge_weight[edge] for edge in G_reduced.edges())

    edge_cuts_percentage.append(actual_edge_cut) 
    total_weight.append(t_weight) 
    in_degree.append(get_in_degree(G_reduced))
    out_degree.append(get_out_degree(G_reduced)) 
    running_time.append(time_spent)
    average_clustering.append(graph_tool.clustering.global_clustering(G_reduced))
    
    nn.append(G_reduced.num_vertices())
    ne.append(G_reduced.num_edges())
    wcc.append(len(label_components(G_reduced, directed=False)[1])) 

    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, assortativity
 
def get_in_degree(G): 
    if (G.num_vertices() == 0):
        return 0
    return (sum(G.get_in_degrees(G.get_vertices()) )/float(G.num_vertices()))

def get_out_degree(G):  
    if (G.num_vertices() == 0):
        return 0
    return (sum(G.get_out_degrees(G.get_vertices())/float(G.num_vertices())))
    
def WIS_test(G, edge_percentages, weight_attr='transferred'): 
    edge_cuts= [int(G.num_vertices() * x) for x in edge_percentages] 
    G_r = G.copy()
    edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, assortativity = WIS_reduce_test(G_r, edge_cuts, weight_attr=weight_attr)
    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, assortativity

def WIS_reduce_test_with_graphs(graph, sizes, weight_attr):   # weighted    
    n = graph.num_vertices()
    node_indexes = np.arange(n) 

    edge_weight = graph.edge_properties[weight_attr]
    nodes = graph.get_vertices()
    in_degrees =  np.array(graph.get_in_degrees(graph.get_vertices(), edge_weight))
    out_degrees =  np.array(graph.get_out_degrees(graph.get_vertices(), edge_weight)) 
    cum_weights = np.add(in_degrees, out_degrees) 
    tot_weight = sum(cum_weights) 
    node_prob = [x / int(tot_weight) for x in cum_weights]    
 
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
    sim = []

    for size in sizes:
        current_time = time.time() 
        indexes = np.random.choice(node_indexes, size, replace=False, p=node_prob) 
        nodes_to_keep = nodes[indexes]
        nodes_to_keep = list(set(nodes_to_keep))
        
        G_reduced = Graph(graph)
        nodes_to_remove = [x for x in G_reduced.get_vertices() if x not in nodes_to_keep]

        for v in reversed(sorted(nodes_to_remove)):
            G_reduced.remove_vertex(v, fast=True)
 
        actual_edge_cut = 1 - (graph.num_edges() - G_reduced.num_edges()) / graph.num_edges()

        time_spent = time.time()-current_time
        edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time = get_stats(G_reduced, actual_edge_cut, weight_attr, edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, time_spent)
        graphs.append(Graph(G_reduced, prune=True))
        G_reduced.clear_filters()
        
    print("<<<<<wis sim: ", sim)
    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, graphs


def WIS_run(graph, sizes, weight_attr):   # weighted    
    n = graph.num_vertices()
    node_indexes = np.arange(n) 

    edge_weight = graph.edge_properties[weight_attr]
    nodes = graph.get_vertices()
    in_degrees =  np.array(graph.get_in_degrees(graph.get_vertices(), edge_weight))
    out_degrees =  np.array(graph.get_out_degrees(graph.get_vertices(), edge_weight)) 
    cum_weights = np.add(in_degrees, out_degrees) 
    tot_weight = sum(cum_weights) 
    node_prob = [x / int(tot_weight) for x in cum_weights]    
  
    graphs = []

    for size in sizes:
        current_time = time.time() 
        indexes = np.random.choice(node_indexes, size, replace=False, p=node_prob) 
        nodes_to_keep = nodes[indexes]
        nodes_to_keep = list(set(nodes_to_keep))
        
        G_reduced = Graph(graph)
        nodes_to_remove = [x for x in G_reduced.get_vertices() if x not in nodes_to_keep]

        for v in reversed(sorted(nodes_to_remove)):
            G_reduced.remove_vertex(v, fast=True)
  
        graphs.append(Graph(G_reduced, prune=True))
        print("num_vertices", G_reduced.num_vertices())
        print("num_edges", G_reduced.num_edges())
        G_reduced.clear_filters()
        
    return graphs

def WIS_test_with_graph(G, edge_percentages, weight_attr='transferred'): 
    edge_cuts= [int(G.num_vertices() * x) for x in edge_percentages]   
    edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, graphs = WIS_reduce_test_with_graphs(G, edge_cuts, weight_attr=weight_attr)
    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, graphs

def WIS_graph(G, edge_percentages, weight_attr='transferred'): 
    edge_cuts= [int(G.num_vertices() * x) for x in edge_percentages]   
    return WIS_run(G, edge_cuts, weight_attr=weight_attr)
