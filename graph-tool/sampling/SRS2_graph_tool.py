from graph_tool.all import * 
import numpy as np
import json
import time
import datetime
import random
import bisect
import itertools
 
def SRS2(G_reduced, e_delete, weight_attr, cut_size): 
    edge_weight = G_reduced.edge_properties[weight_attr]
    edges_by_weight = [(edge, edge_weight[edge]) for edge in G_reduced.edges()]
    sorted_edges_by_weight = sorted(edges_by_weight,reverse=True, key=lambda x: x[1])
     
    edges_to_remove = list(sorted_edges_by_weight)[int(cut_size):] 

    for edge in edges_to_remove:
        e_delete[edge[0]] = False 

    G_reduced.set_edge_filter(e_delete) 
    return G_reduced

def get_in_degree(G): 
    return (sum(G.get_in_degrees(G.get_vertices()) )/float(G.num_vertices()))

def get_out_degree(G): 
    return (sum(G.get_out_degrees(G.get_vertices())/float(G.num_vertices())))

def run_SRS2(G, edges_max_goal, weight_attr):  
    e_delete = G.new_edge_property("bool", True) 
    G_reduced = SRS2(G, e_delete, weight_attr, edges_max_goal)
    return G_reduced
 
def get_stats(G_reduced, weight_attr, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, time_spent):
    edge_weight = G_reduced.edge_properties[weight_attr]
    t_weight = sum(edge_weight[edge] for edge in G_reduced.edges())

    total_weight.append(t_weight) 
    in_degree.append(get_in_degree(G_reduced))
    out_degree.append(get_out_degree(G_reduced))
    running_time.append(time_spent)
    #average_clustering.append(nx.average_clustering(G_reduced.to_undirected(as_view=True)))

    nn.append(G_reduced.num_vertices())
    ne.append(G_reduced.num_edges())
    wcc.append(len(label_components(G_reduced, directed=False)[1])) 

    return total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time
 
def SRS2_test(filename, G, edge_cuts, weight_attr): 
    total_weight = [] 
    in_degree = []
    out_degree = []
    average_clustering = []
    nn = []
    ne = []
    wcc = [] 
    running_time = [] 

    for edge_cut in edge_cuts:  
        edges_max_goal = G.num_edges() * edge_cut 
        
        current_time = time.time() 
        G_reduced = run_SRS2(G, edges_max_goal, weight_attr) 
        time_spent = time.time()-current_time

        total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time = get_stats(G_reduced, weight_attr, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, time_spent)
 
        G_reduced.clear_filters()

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time

def SRS2_test_with_graphs(filename, G, edge_cuts, weight_attr='transferred'):   
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
        edges_max_goal = G.num_edges() * edge_cut 
        
        current_time = time.time() 
        G_reduced = run_SRS2(G, edges_max_goal, weight_attr) 
        time_spent = time.time()-current_time

        total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time = get_stats(G_reduced, weight_attr, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, time_spent)
 
        graphs.append(Graph(G_reduced, prune=True))
        G_reduced.clear_filters()

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, graphs

 
def SRS2_graphs(filename, G, edge_cuts, weight_attr='transferred'):  
    graphs = []

    for edge_cut in edge_cuts:  
        edges_max_goal = G.num_edges() * edge_cut  
        G_reduced = run_SRS2(G, edges_max_goal, weight_attr) 
        graphs.append(Graph(G_reduced, prune=True))
        G_reduced.clear_filters()

    return graphs

 