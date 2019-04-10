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

    for size in sizes:
        print("size", size) 
        indexes = np.random.choice(node_indexes, size, replace=True, p=node_prob) 
        nodes_to_keep = nodes[indexes]
        nodes_to_keep = list(set(nodes_to_keep))
        print("nodes_to_keep size: ", len(nodes_to_keep))
        G_reduced = Graph(graph)
        nodes_to_remove = [x for x in G_reduced.get_vertices() if x not in nodes_to_keep]

        for v in reversed(sorted(nodes_to_remove)):
            G_reduced.remove_vertex(v, fast=True)
 
        actual_edge_cut = 1 - (graph.num_edges() - G_reduced.num_edges()) / graph.num_edges()

        edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc = get_stats(G_reduced, actual_edge_cut, weight_attr, edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc)


    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc

def get_stats(G_reduced, actual_edge_cut, weight_attr, edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc):
    edge_weight = G_reduced.edge_properties[weight_attr]
    t_weight = sum(edge_weight[edge] for edge in G_reduced.edges())

    edge_cuts_percentage.append(actual_edge_cut) 
    total_weight.append(t_weight) 
    in_degree.append(get_in_degree(G_reduced))
    out_degree.append(get_out_degree(G_reduced)) 
    #average_clustering.append(nx.average_clustering(G_reduced.to_undirected(as_view=True)))

    nn.append(G_reduced.num_vertices())
    ne.append(G_reduced.num_edges())
    wcc.append(number_weakly_connected_components(G_reduced))

    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc


def get_in_degree(G): 
    return (sum(G.get_in_degrees(G.get_vertices()) )/float(G.num_vertices()))

def get_out_degree(G): 
    return (sum(G.get_out_degrees(G.get_vertices())/float(G.num_vertices())))
    
def WIS_test(G, edge_percentages, weight_attr='transferred'): 
    edge_cuts= [int(G.num_vertices() * x) for x in edge_percentages] 
    G_r = G.copy()
    edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc = WIS_reduce_test(G_r, edge_cuts, weight_attr=weight_attr)
    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc


def WIS_reduce_test_with_graphs(graph, sizes, weight_attr):   # weighted    
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
    graphs = []

    for size in sizes:
        print("size", size) 
        indexes = np.random.choice(node_indexes, size, replace=True, p=node_prob) 
        nodes_to_keep = nodes[indexes]
        nodes_to_keep = list(set(nodes_to_keep))
        print("nodes_to_keep size: ", len(nodes_to_keep))
        G_reduced = Graph(graph)
        nodes_to_remove = [x for x in G_reduced.get_vertices() if x not in nodes_to_keep]

        for v in reversed(sorted(nodes_to_remove)):
            G_reduced.remove_vertex(v, fast=True)
 
        actual_edge_cut = 1 - (graph.num_edges() - G_reduced.num_edges()) / graph.num_edges()

        edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc = get_stats(G_reduced, actual_edge_cut, weight_attr, edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc)
        graphs.append(Graph(G_reduced, prune=True))
        G_reduced.clear_filters()
        
    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, graphs


def WIS_test_with_graph(G, edge_percentages, weight_attr='transferred'): 
    edge_cuts= [int(G.number_of_nodes() * x) for x in edge_percentages]  
    edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, graphs = WIS_reduce_test_with_graphs(G, edge_cuts, weight_attr=weight_attr)
    return edge_cuts_percentage, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, graphs
