import networkx as nx
#import matplotlib.pyplot as plt
import json
import time
import datetime
import numpy as np
from collections import OrderedDict
from math import log10
  
def remove_edges(G_reduced, items, edges_max_goal):
    current_time = time.time()
    removed_edges = []
    sorted_bet_cent_edges = sorted(items,reverse=False, key=lambda x: x[1])
    
    #print("count :", len(list(sorted_bet_cent_edges)))
    print("sorting old took:", time.time()-current_time)
    to_remove = G_reduced.number_of_edges() - edges_max_goal
    for bet_cent in sorted_bet_cent_edges:  
        if (len(removed_edges) >= to_remove):
            break
        if G_reduced.degree(bet_cent[0]) > 2 and G_reduced.degree(bet_cent[1]) > 2:
            G_reduced.remove_edge(bet_cent[0], bet_cent[1]) 
            removed_edges.append(bet_cent)

    time_spent = time.time()-current_time
    print("for loop took : ", time_spent)
    G_reduced.remove_edges_from(removed_edges)

    return G_reduced, removed_edges
 
def postprocess(G_reduced, items):
    number_wcc = nx.number_weakly_connected_components(G_reduced) 
    if number_wcc == 1:
        #print("****** already 1 component ")
        return G_reduced

    current_time = time.time()
    #print("items", items)
    _components = [c for c in nx.weakly_connected_components(G_reduced)]
    #print( _components )
   # _components = list(G_reduced.subgraph(c) for c in nx.weakly_connected_components(G_reduced))
    print("number of disconnected components before postprocessing:", number_wcc)
   
    #print("items: ", items)
    #print("reversed(items): ", reversed(items))
    for edge in reversed(items):
        #print(edge)
        if number_wcc == 1: 
            break
            
        for c in _components:
            if edge[0] in c and edge[1] in c :
                # edge is within one component
                break
            elif edge[0] in c or edge[1] in c : 
                # edge is within one component
                G_reduced.add_edge(*edge)
                _components = [c for c in nx.weakly_connected_components(G_reduced)]
                number_wcc = len(_components)
                # try _components = nx.weakly_connected_components(G_reduced)
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

def edge_reduce_approximate_test(filename, G, edge_cuts, weight_attr='transferred'):
    c = 10
    take_count = int(c * log10(nx.number_of_nodes(G)))
    print("edge_betweenness_centrality")

    bet_cent_edges = nx.edge_betweenness_centrality(G, k=take_count, weight=weight_attr) 
 
    print("edge_betweenness_centrality done")
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
        print("copying:")
        graph = nx.DiGraph(G)
        print("original:", graph.number_of_edges())
        G_reduced, removed_edges = remove_edges(graph, bet_cent_edges, edges_max_goal)
        #graph.clear() 
        G_reduced = postprocess(G_reduced, removed_edges)
        
        time_spent = time.time()-current_time
        
        total_weight.append(G_reduced.size(weight=weight_attr)) 
        in_degree.append(get_in_degree(G_reduced))
        out_degree.append(get_out_degree(G_reduced))
        running_time.append(time_spent)
        #average_clustering.append(nx.average_clustering(G_reduced.to_undirected(as_view=True)))

        nn.append(G_reduced.number_of_nodes())
        ne.append(G_reduced.number_of_edges())
        wcc.append(nx.number_weakly_connected_components(G_reduced))

        print("weight: ", G_reduced.size())
        print("weight: ", G_reduced.size(weight=weight_attr)) 

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc

 

def read_json(filename):
    with open(filename) as f:
        js_graph = json.load(f) #, default={'sender': 'source'})
        _attrs = dict(source='sender', target='receiver', name='guid',
              key='guid', link='links')
    #return nx.readwrite.node_link_graph(js_graph, {'link': 'links', 'source': 'sender', 'target': 'receiver', 'key': 'guid'})
    return nx.readwrite.node_link_graph(js_graph, directed=True, multigraph=False, attrs={'link': 'links', 'source': 'sender', 'target': 'receiver', 'key': 'guid', 'name': 'guid'} )
 
def read_json_file(filename):
    graph = read_json(filename)
    return graph.subgraph(max(nx.weakly_connected_components(graph), key=len))  

