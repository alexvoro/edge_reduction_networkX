import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import datetime
import random
import bisect
import itertools

def selectRoot(G, weight_attr):
    cum_weights = [0]*G.number_of_nodes()
    nodes = [0]*G.number_of_nodes()
    for i,v in enumerate(G.nodes):  
        cum_weights[i] = (v, G.degree(v, weight=weight_attr))
        nodes[i] = v 
    
    items = sorted(cum_weights,reverse=True, key=lambda x: x[1])
    root = items[:1] 
    return root[0][0]

def FBF_recursive(G, G_ud, tree, weight_attr, neighbours = None, lastAdded = None):  
    # pick a neightbour Vn+1 with a highest degree
    if neighbours == None:
        neighbours = []  
    neighbours.extend(G.degree(G.neighbors(lastAdded), weight=weight_attr)) 
          
    neighbours = [x for x in neighbours if x[0] not in tree.nodes]  
    neighbours = list(set(neighbours))  
     
    top = sorted(neighbours,reverse=True, key=lambda x: x[1])[:1][0]  
    edges = [e for e in G_ud.edges(top[0]) if e[1] in tree.nodes or e[0] in tree.nodes]  
    vn_weight = []
    for e in edges:
        if e[0] != top:
            node = e[0]
        else:
            node = e[1] 
        vn_weight.extend([x for x in G.degree(G_ud.neighbors(node), weight=weight_attr) if x[0] in tree.nodes]) 
     
    vn_weight = list(set(vn_weight))  
        
    other_end = sorted(vn_weight, reverse=True, key=lambda x: x[1])[:1][0]
     
    directed_edges = [e for e in G.edges(top[0], data=weight_attr) if e[1] in tree.nodes]
    directed_edges = directed_edges + [e for e in G.edges(other_end[0], data=weight_attr) if e not in tree.edges.data()]
 
    edge = [e for e in directed_edges if e[1] == top[0] and e[0] == other_end[0] or e[0] == top[0] and e[1] == other_end[0]][0]
     
    return edge, neighbours, top[0]

def dense_component_extraction(G, tree, threshold, weight_attr):
    # for each edge not in tree
    skipped_edges = set(G.edges.data(weight_attr, default=0)) - set(tree.edges.data(weight_attr, default=0))  
    
    # compute shortest path, keep adding the ones with the longest path
    items = []
    for edge in skipped_edges:
        length = nx.shortest_path_length(G, source=edge[0], target=edge[1], weight = weight_attr)
        items.append((edge[0], edge[1], edge[2], length))
    
    items = sorted(items,reverse=True, key=lambda x: x[3]) 
    for item in items: 
        if tree.number_of_edges() >= threshold:
            continue
        tree.add_edge(item[0], item[1], weight=item[2]) 
    
    return tree

def FBF(G, weight_attr, root, threshold): 
    tree = nx.DiGraph() 
    tree.add_node(root)  
    
    G_ud = G.to_undirected()
    edge, neighbours, top = FBF_recursive(G, G_ud, tree, weight_attr, None, root) 
    tree.add_edge(edge[0], edge[1], weight=edge[2])
    
    while tree.number_of_nodes() != G.number_of_nodes():
        edge, neighbours, top = FBF_recursive(G, G_ud, tree, weight_attr, neighbours, top)
        tree.add_edge(edge[0], edge[1], weight=edge[2])
        
    print("before dense_component_extraction")
    print(nx.info(tree))
    tree = dense_component_extraction(G, tree, threshold, weight_attr)
    
    print("after dense_component_extraction")
    print(nx.info(tree))
    
    return tree

def get_in_degree(G):
    return (sum(d for n, d in G.in_degree())/float(len(G)))

def get_out_degree(G):
    return (sum(d for n, d in G.out_degree())/float(len(G)))

def run_focus_test(G, edge_cuts, weight_attr='transferred'):
    G_r = G.copy()
    total_weight = [] 
    in_degree = []
    out_degree = []
    average_clustering = []
    nn = []
    ne = []
    wcc = []

    root = selectRoot(G_r, weight_attr)

    print("root: ", root)
    for edge_cut in edge_cuts: 
        threshold = G_r.number_of_edges() * edge_cut

        root = selectRoot(G_r, weight_attr)

        G_reduced = FBF(G_r, weight_attr, root, threshold)
        print("weight: ", G_reduced.size())
        print("weight: ", G_reduced.size(weight=weight_attr)) 

        total_weight.append(G_reduced.size(weight=weight_attr))
        print("weight: ", G_reduced.size())
        print("weight: ", G_reduced.size(weight=weight_attr))
        in_degree.append(get_in_degree(G_reduced))
        out_degree.append(get_out_degree(G_reduced))
        average_clustering.append(nx.average_clustering(G_reduced.to_undirected()))
        #print("postprocessing took:", time.time()-current_time)  
        #print(nx.info(G_reduced))

        nn.append(G_reduced.number_of_nodes())
        ne.append(G_reduced.number_of_edges())
        wcc.append(len(list(nx.weakly_connected_component_subgraphs(G_reduced))))

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc


def run_focus_test_one(G, edge_cuts, weight_attr='transferred'):
    G_r = G.copy()
    total_weight = []
    in_degree = []
    out_degree = []

    edge_cut = edge_cuts[0]
    threshold = G_r.number_of_edges() * edge_cut

    root = selectRoot(G_r, weight_attr)
    print("root: ", root)
     
    G_reduced = FBF(G_r, weight_attr, root, threshold)
    print("weight: ", G_reduced.size())
    print("weight: ", G_reduced.size(weight=weight_attr)) 

    total_weight.append(G_reduced.size(weight=weight_attr))
    print("weight: ", G_reduced.size())
    print("weight: ", G_reduced.size(weight=weight_attr))

    in_degree.append(get_in_degree(G_reduced))
    out_degree.append(get_out_degree(G_reduced))

    #print("postprocessing took:", time.time()-current_time)  
    #print(nx.info(G_reduced))

    return edge_cuts, total_weight, in_degree, out_degree
    

def run_focus(G, weight, threshold): 
    root = selectRoot(G, weight)
    print("root: ", root)
    
    current_time = time.time()
    tree = FBF(G, weight, root, threshold)
    print("FBF took", time.time()-current_time)
    return tree