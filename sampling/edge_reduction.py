import networkx as nx
import matplotlib.pyplot as plt
import json
import time
import datetime
import numpy as np
from collections import OrderedDict
from math import log10
  
def remove_edges(G_reduced, items, edges_max_goal): 
    current_time = time.time()
    removed_edges = []

    # lowest to highest
    sorted_edges = sorted(items,reverse=False, key=lambda x: items[x])  
    sorted_bet_cent_edges = [(x, items[x]) for x in sorted_edges] 

   # print("sorted_bet_cent_edges2 :", ordered2)
    print("sorting took:", time.time()-current_time)

    current_time = time.time()
    for edge, bet_cent in sorted_bet_cent_edges:  
        if (G_reduced.number_of_edges() <= edges_max_goal):
            #print("done :", G_reduced.number_of_edges())
            break
        if G_reduced.degree(edge[0]) > 2 and G_reduced.degree(edge[1]) > 2:
            G_reduced.remove_edge(*edge)  
            #if (bet_cent != 0.0):
            #    print(edge, bet_cent) 
            removed_edges.append((edge, bet_cent))
        
    time_spent = time.time()-current_time
    print("for loop took : ", time_spent)

    return G_reduced, removed_edges
  
def postprocess(G_reduced, items):
    current_time = time.time()
    if nx.number_weakly_connected_components(G_reduced) == 1:
        #print("****** already 1 component ")
        return G_reduced
    #print("items", items)
    _components = (G_reduced.subgraph(c) for c in nx.weakly_connected_components(G_reduced))
    print("number of disconnected components before postprocessing:", len(_components))
  
    #for x in items:
    #    if (x[1] != 0.0):
    #        print(x, x[0], x[1]) 

    #  highest to lowest
    removed_sorted_edges = sorted(items,reverse=False, key=lambda x: x[1])  
     
    for edge, bc in removed_sorted_edges: 
        #print(edge, bc)
        if len(_components) == 1: 
            break
            
        for c in _components:
            if c.has_node(edge[0]) and c.has_node(edge[1]):
                # edge is within one component
                break
            elif c.has_node(edge[0]) or c.has_node(edge[1]): 
                # edge is within one component
                G_reduced.add_edge(*edge)
                _components = (G_reduced.subgraph(c) for c in nx.weakly_connected_components(G_reduced))
                break
     
    time_spent = time.time()-current_time
    print("remove_edges took : ", time_spent)
    return G_reduced

def edge_reduce(G, edges_max_goal, weight_attr='transferred'):
    bet_cent_edges = nx.edge_betweenness_centrality(G, weight=weight_attr)
 
    G_reduced, removed_edges = remove_edges(G, bet_cent_edges, edges_max_goal) 
    G_reduced = postprocess(G_reduced, removed_edges)

def edge_reduce_test(G, edge_cuts, weight_attr='transferred'):
    bet_cent_edges = nx.edge_betweenness_centrality(G, weight=weight_attr)
    
    total_weight = []

    for edge_cut in edge_cuts:
        edges_max_goal = G.number_of_edges() * edge_cut
        G_reduced, removed_edges = remove_edges(G.copy(), bet_cent_edges, edges_max_goal) 
        G_reduced = postprocess(G_reduced, removed_edges)
        
        total_weight.append(G_reduced.size(weight=weight_attr)) 

    return edge_cuts, total_weight

def edge_reduce_approximate(G, edges_max_goal, weight_attr='transferred'): 
    c = 10
    take_count = int(c * log10(nx.number_of_nodes(G)))
    print("take_count",take_count)

    bet_cent_edges = nx.edge_betweenness_centrality(G, k=take_count, weight=weight_attr) 
  
    G_reduced, removed_edges = remove_edges(G, bet_cent_edges, edges_max_goal) 
    G_reduced = postprocess(G_reduced, removed_edges)
    

def get_in_degree(G):
    return (sum(d for n, d in G.in_degree())/float(len(G)))

def get_out_degree(G):
    return (sum(d for n, d in G.out_degree())/float(len(G)))

def edge_reduce_approximate_test(G, edge_cuts, weight_attr='transferred'):
    c = 10
    take_count = int(c * log10(nx.number_of_nodes(G)))
    #print("take_count",take_count)

    bet_cent_edges = nx.edge_betweenness_centrality(G, k=take_count, weight=weight_attr) 
 
    #print("bet_cent_edges: ", bet_cent_edges)
 
    total_weight = []
    in_degree = []
    out_degree = []
    running_time = []
    average_clustering = []
    nn = []
    ne = []
    wcc = []

    for edge_cut in edge_cuts:  
        current_time = time.time()
        edges_max_goal = G.number_of_edges() * edge_cut
        print("edge_cut:", edge_cut)
        print("edges_max_goal:", edges_max_goal)
        G_reduced, removed_edges = remove_edges(G.copy(), bet_cent_edges, edges_max_goal)
        print("weight: ", G_reduced.size())
        print("weight: ", G_reduced.size(weight=weight_attr)) 
        G_reduced = postprocess(G_reduced, removed_edges)
        
        time_spent = time.time()-current_time
        
        total_weight.append(G_reduced.size(weight=weight_attr)) 
        in_degree.append(get_in_degree(G_reduced))
        out_degree.append(get_out_degree(G_reduced))
        running_time.append(time_spent)
        #average_clustering.append(nx.average_clustering(G_reduced.to_undirected()))

        nn.append(G_reduced.number_of_nodes())
        ne.append(G_reduced.number_of_edges())
        wcc.append(nx.number_weakly_connected_components(G_reduced))

        print("weight: ", G_reduced.size())
        print("weight: ", G_reduced.size(weight=weight_attr)) 

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc


def edge_reduce_approximate_test_with_graph(G, edge_cuts, weight_attr='transferred'):
    c = 10
    take_count = int(c * log10(nx.number_of_nodes(G)))
    #print("take_count",take_count)

    bet_cent_edges = nx.edge_betweenness_centrality(G, k=take_count, weight=weight_attr) 
 
    #print("bet_cent_edges: ", bet_cent_edges)
 
    total_weight = []
    in_degree = []
    out_degree = []
    running_time = []
    average_clustering = []
    nn = []
    ne = []
    wcc = []
    graphs = []

    for edge_cut in edge_cuts:  
        current_time = time.time()
        edges_max_goal = G.number_of_edges() * edge_cut
        G_reduced, removed_edges = remove_edges(G.copy(), bet_cent_edges, edges_max_goal)
        print("weight: ", G_reduced.size())
        print("weight: ", G_reduced.size(weight=weight_attr)) 
        G_reduced = postprocess(G_reduced, removed_edges)
        
        time_spent = time.time()-current_time
        
        total_weight.append(G_reduced.size(weight=weight_attr)) 
        in_degree.append(get_in_degree(G_reduced))
        out_degree.append(get_out_degree(G_reduced))
        running_time.append(time_spent)
        #average_clustering.append(nx.average_clustering(G_reduced.to_undirected()))

        nn.append(G_reduced.number_of_nodes())
        ne.append(G_reduced.number_of_edges())
        wcc.append(nx.number_weakly_connected_components(G_reduced))
        graphs.append(G_reduced)
        print("weight: ", G_reduced.size())
        print("weight: ", G_reduced.size(weight=weight_attr)) 

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, graphs

