import networkx as nx
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

    neighbours.extend(G.degree(G_ud.neighbors(lastAdded), weight=weight_attr)) 
           
    # neighbours of nodes in tree
    neighbours = [x for x in neighbours if x[0] not in tree.nodes]  
    neighbours = list(set(neighbours))  
     
    # neighbours with the highest degree
    top = sorted(neighbours, reverse=True, key=lambda x: x[1])[:1][0]
    
    # edges between Vn+1 and tree nodes 
    edges = [e for e in G_ud.edges(top[0], data=weight_attr) if e[1] in tree.nodes or e[0] in tree.nodes] 
  
    nodes = []
    for e in edges:
        nodes.append(e[0])
        nodes.append(e[1])
        
    nodes = set(nodes) - set(top) 
    
    vn_weight = (G.degree(nodes, weight=weight_attr)) 
    vn_weight = list(set(vn_weight)) 
    other_end = sorted(vn_weight, reverse=True, key=lambda x: x[1])[:1][0] 
    
    edge = [e for e in edges if e[0] == other_end[0] or e[1] == other_end[0]][0]
    #print(edge)
    return edge, neighbours, top[0] 

def dense_component_extraction(G, tree, threshold, weight_attr):
    if tree.number_of_edges() >= threshold:
        return tree 
    # for each edge not in tree
    skipped_edges = set(G.edges.data(weight_attr, default=0)) - set(tree.edges.data(weight_attr, default=0))  
    tree_ud = tree.to_undirected()
    
    # compute shortest path, keep adding the ones with the longest path
    items = []
    lengths = dict(nx.all_pairs_shortest_path_length(tree_ud)) 
    for edge in skipped_edges:
        #length = nx.shortest_path_length(tree, source=edge[0], target=edge[1], weight = weight_attr)
        length = lengths[edge[1]][edge[0]]
        items.append((edge[0], edge[1], edge[2], length)) 
    
    items = sorted(items, reverse=True, key=lambda x: x[3])  
    #print("items", items)
    
    items = [(item[0], item[1], item[2]) for item in items]
    number_to_add = int(threshold - tree.number_of_edges()) 
    tree.add_weighted_edges_from(items[:number_to_add])
    return tree 

def FBF(G, weight_attr, root, threshold): 
    tree = nx.DiGraph() 
    tree.add_node(root)  
    
    G_ud = G.to_undirected()
    edge, neighbours, top = FBF_recursive(G, G_ud, tree, weight_attr, None, root) 
    #print("weight=edge[2]", edge[2])
    #print(edge)
    #tree.add_edges_from([edge])
    tree.add_edge(edge[0], edge[1], weight=edge[2])
    #print(tree.edges(edge[0], data="weight"))

    while tree.number_of_nodes() != G.number_of_nodes():
        edge, neighbours, top = FBF_recursive(G, G_ud, tree, weight_attr, neighbours, top)
        #print("weight=edge[2]", edge[2])
        
        #tree.add_edges_from([edge])
        tree.add_edge(edge[0], edge[1], weight_attr=edge[2])
        
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
        print("edge_cut:", edge_cut)
        print("threshold:", threshold)
        #root = selectRoot(G_r, weight_attr)

        print("threshold", threshold)
        G_reduced = FBF(G_r, weight_attr, root, threshold)
        print("weight: ", G_reduced.size())
        #print("weight_attr", weight_attr)
        
        #print("weight: ", G_reduced.size(weight=weight_attr)) 
        print("weight: ", G_reduced.size(weight="weight")) 
        
        print(nx.info(G_reduced))
        total_weight.append(G_reduced.size(weight="weight")) 
        #print("weight: ", G_reduced.size())
        #print("weight: ", G_reduced.size(weight=weight_attr))
        in_degree.append(get_in_degree(G_reduced))
        out_degree.append(get_out_degree(G_reduced))
        #average_clustering.append(nx.average_clustering(G_reduced.to_undirected()))
        #print("postprocessing took:", time.time()-current_time)  
        #print(nx.info(G_reduced))

        nn.append(G_reduced.number_of_nodes())
        ne.append(G_reduced.number_of_edges())
        wcc.append(nx.number_weakly_connected_components(G_reduced))

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
     
    G_reduced = FBF(G.copy(), weight_attr, root, threshold)
    #print("weight: ", G_reduced.size())
    #print("weight: ", G_reduced.size(weight=weight_attr)) 

    total_weight.append(G_reduced.size(weight=weight_attr))
    #print("weight: ", G_reduced.size())
    #print("weight: ", G_reduced.size(weight=weight_attr))

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

def run_focus_test_with_graphs(G, edge_cuts, weight_attr='transferred'):
    G_r = G.copy()
    total_weight = [] 
    in_degree = []
    out_degree = []
    average_clustering = []
    nn = []
    ne = []
    wcc = []
    graphs = []

    root = selectRoot(G_r, weight_attr)

    print("root: ", root)
    for edge_cut in edge_cuts: 
        threshold = G_r.number_of_edges() * edge_cut

        root = selectRoot(G_r, weight_attr)

        print("threshold", threshold)
        G_reduced = FBF(G_r, weight_attr, root, threshold) 
        
        print(nx.info(G_reduced))
        total_weight.append(G_reduced.size(weight="weight"))  
        in_degree.append(get_in_degree(G_reduced))
        out_degree.append(get_out_degree(G_reduced))
        #average_clustering.append(nx.average_clustering(G_reduced.to_undirected())) 

        nn.append(G_reduced.number_of_nodes())
        ne.append(G_reduced.number_of_edges())
        wcc.append(len(list(nx.weakly_connected_component_subgraphs(G_reduced))))
        graphs.append(G_reduced)

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, graphs

