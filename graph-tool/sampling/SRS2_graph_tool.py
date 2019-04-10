from graph_tool.all import *  
import networkx as nx
import numpy as np
import json
import time
import datetime
import random
import bisect
import itertools
 
def weakly_connected_components(G): 
    seen = set()
    for v in G.vertices():
        if v not in seen:
            c = set(_plain_bfs(G, v))
            yield c
            seen.update(c)


def number_weakly_connected_components(G): 
    return sum(1 for wcc in weakly_connected_components(G))

def _plain_bfs(G, source):
    """A fast BFS node generator 
    The direction of the edge between nodes is ignored. 
    For directed graphs only. 
    """ 
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                yield v
                seen.add(v)
                nextlevel.update(v.in_neighbors())
                nextlevel.update(v.out_neighbors())

def SRS2(G_reduced, e_delete, weight_attr, cut_size): 
    edge_weight = G_reduced.edge_properties[weight_attr]
    edges_by_weight = [(edge, edge_weight[edge]) for edge in G_reduced.edges()]
    sorted_edges_by_weight = sorted(edges_by_weight,reverse=True, key=lambda x: x[1])
     
    edges_to_remove = list(sorted_edges_by_weight)[int(cut_size):] 

    for edge in edges_to_remove:
        e_delete[edge] = False 

    G_reduced.set_edge_filter(e_delete) 
    return G_reduced

def get_in_degree(G): 
    return (sum(G.get_in_degrees(G.get_vertices()) )/float(G.num_vertices()))

def get_out_degree(G): 
    return (sum(G.get_out_degrees(G.get_vertices())/float(G.num_vertices())))

def run_SRS2_test(G, edges_max_goal, weight_attr):  
    e_delete = G.new_edge_property("bool", True) 
    G_reduced = SRS2(G, e_delete, weight_attr, edges_max_goal)
    return G_reduced
 
def get_stats(G_reduced, weight_attr, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc):
    edge_weight = G_reduced.edge_properties[weight_attr]
    t_weight = sum(edge_weight[edge] for edge in G_reduced.edges())

    total_weight.append(t_weight) 
    in_degree.append(get_in_degree(G_reduced))
    out_degree.append(get_out_degree(G_reduced))
    #running_time.append(time_spent)
    #average_clustering.append(nx.average_clustering(G_reduced.to_undirected(as_view=True)))

    nn.append(G_reduced.num_vertices())
    ne.append(G_reduced.num_edges())
    wcc.append(number_weakly_connected_components(G_reduced))

    return total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc

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
        edges_max_goal = G.number_of_edges() * edge_cut 
        
        current_time = time.time() 
        G_reduced = run_SRS2_test(G, edges_max_goal, weight_attr) 
        time_spent = time.time()-current_time

        total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc = get_stats(G_reduced, weight_attr, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc)
 
        G_reduced.clear_filters()

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc

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
        edges_max_goal = G.number_of_edges() * edge_cut 
        
        current_time = time.time() 
        G_reduced = run_SRS2_test(G, edges_max_goal, weight_attr) 
        time_spent = time.time()-current_time

        total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc = get_stats(G_reduced, weight_attr, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc)
 
        graphs.append(Graph(G_reduced, prune=True))
        G_reduced.clear_filters()

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, graphs

 