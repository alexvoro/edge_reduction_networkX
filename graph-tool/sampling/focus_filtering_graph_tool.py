from graph_tool.all import *  
import numpy as np
import json
import time
import datetime
import random
import bisect
import itertools
import operator


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

def selectRoot(graph, weight_attr):
    nodes = graph.get_vertices()
    edge_weight = graph.edge_properties[weight_attr]

    in_degrees =  np.array(graph.get_in_degrees(graph.get_vertices(), edge_weight))
    out_degrees =  np.array(graph.get_out_degrees(graph.get_vertices(), edge_weight)) 
    cum_weights = np.add(in_degrees, out_degrees)    
  
    nodes_degrees = np.column_stack((nodes,cum_weights))
    print(nodes_degrees)
    items = sorted(nodes_degrees,reverse=True, key=lambda x: x[1])
    root = items[:1] 

    print("root", root)
    root2 = nodes_degrees[np.argmax( np.array([x[1] for x in nodes_degrees]))]

    print("root2", root2)
    #todo replace root with root2

    return root[0][0], nodes_degrees

def FBF_recursive(G, tree_nodes, weight_attr, node_degrees, neighbours = None, lastAdded = None):  
    # pick a neightbour Vn+1 with a highest degree
    if neighbours == None:
        neighbours = []  
    v = G.vertex(lastAdded) 
    n = v.all_neighbors()
    new_neighbours = [(vert, vert.in_degree(weight=weight_attr)+vert.out_degree(weight=weight_attr)) for vert in v.all_neighbors()]
    neighbours.extend(new_neighbours)  
            
    # neighbours of nodes in tree
    neighbours = [x for x in neighbours if x[0] not in tree_nodes]  
    neighbours = list(set(neighbours))  
     
    # neighbours with the highest degree 
    top = max(neighbours, key=lambda x: x[1])
    #print("neighbours.len 2:", len(neighbours)) 
    #print("neighbours.len 2:", len(neighbours))
    #print(G.edges)

    # edges between Vn+1 and tree nodes    
    edges = [e for e in top[0].all_edges() if e.source() in tree_nodes or e.target() in tree_nodes] 

    edges_vert = np.array([[e.source(), e.target()] for e in edges]).flatten()
    nodes = set(list(edges_vert)) - set([top[0]])  
    #vn_weight = [node_degrees[np.where(node_degrees[:, 0] == vert)][0] for vert in nodes]
    
    vn_weight = [(vert, vert.in_degree(weight=weight_attr)+vert.out_degree(weight=weight_attr)) for vert in nodes]
    other_end = max(vn_weight, key=lambda x: x[1])
  
    edge = [e for e in edges if e.source() == other_end[0] or e.target() == other_end[0]][0] 
    return edge, neighbours, top[0] 

def dense_component_extraction(G, tree, v_delete, e_delete, threshold, weight_attr):
    if tree.num_edges() >= threshold:
        return tree 

    # for each edge not in tree
    #skipped_edges = set(G.edges.data(weight_attr, default=0)) - set(tree.edges.data(weight_attr, default=0))  
    filtered_edges = set(G.edges()) - set(tree.edges()) 

    edge_weight = G.edge_properties[weight_attr]
    #t_weight = sum(edge_weight[edge] for edge in G_reduced.edges())
    
    #filtered_edges = set(G.edges()) - set(tree.edges())  
     
    #current_time = time.time()
    #tree_ud = tree.to_undirected(as_view=True) 
      
    # compute shortest path, keep adding the ones with the longest path
    items = []  
    #lengths = dict(nx.all_pairs_shortest_path_length(tree_ud))
    #lengths = dict(nx.all_pairs_shortest_path_length(tree_ud)) 

    for edge in filtered_edges:
        #length = nx.shortest_path_length(tree_ud, source=edge[0], target=edge[1], weight = weight_attr)
        length = graph_tool.topology.shortest_distance(G, edge.source(), edge.target(), weights=edge_weight)
        #items.append((edge[0], edge[1], edge[2], lengths[edge[1]][edge[0]])) 
        #items.append((edge[0], edge[1], edge[2], length)) 
        items.append((edge, length)) 
    
    items = sorted(items, reverse=True, key=lambda x: x[1])  
    #print("items", items) 
    
    #items = [(item[0], item[1], item[2]) for item in items]
    number_to_add = int(threshold - tree.num_edges()) 
    edges_to_add = items[:number_to_add]
    print("tree.num_edges(): ", tree.num_edges())
  
    for edge in edges_to_add:
        e_delete[edge[0]] = True  
 
    tree.set_edge_filter(e_delete)
    print("tree.num_edges(): ", tree.num_edges())

    #tree.add_weighted_edges_from(items[:number_to_add])
    return tree 

def FBF(G, weight_attr, root, threshold, node_degrees):  
    G_num_vertices = G.num_vertices()
    tree = Graph(G) 

    v_delete = tree.new_vertex_property("bool", False)  
    e_delete = tree.new_edge_property("bool", False)
      
    # set filters: remove all edges and vertices except for root
    v_delete[root] = True 
    tree.set_vertex_filter(v_delete)
    tree.set_edge_filter(e_delete)
     
    tree_nodes = [root]
    tree_edges = []
    neighbours = list()
    edge_weight = G.edge_properties[weight_attr]
 
    top = root
    while len(tree_nodes) != G_num_vertices:
        edge, neighbours, top = FBF_recursive(G, tree_nodes, edge_weight, node_degrees, neighbours, top)

        tree_nodes.append(top)
        tree_edges.append(edge)
        
    #tree.add_nodes_from(tree_nodes)
    #tree.add_weighted_edges_from(tree_edges)
    for node in tree_nodes:
        v_delete[node] = True 
    
    for edge in tree_edges:
        e_delete[edge] = True  

    tree.set_vertex_filter(v_delete)
    tree.set_edge_filter(e_delete)

    print("before dense_component_extraction")
    print("num edges: ", tree.num_vertices(), tree.num_edges()) 
    tree = dense_component_extraction(G,  tree, v_delete, e_delete, threshold, weight_attr) 
    print("after dense_component_extraction")
    print("num edges: ", tree.num_vertices(), tree.num_edges()) 
    return tree

def get_in_degree(G): 
    return (sum(G.get_in_degrees(G.get_vertices()) )/float(G.num_vertices()))

def get_out_degree(G): 
    return (sum(G.get_out_degrees(G.get_vertices())/float(G.num_vertices())))

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

def run_focus_test(G, edge_cuts, weight_attr='transferred'): 
    total_weight = [] 
    in_degree = []
    out_degree = []
    average_clustering = []
    nn = []
    ne = []
    wcc = []

    root, node_degrees = selectRoot(G, weight_attr)
    
    print("root: ", root)
    for edge_cut in edge_cuts: 
        print("G.num_edges():", G.num_edges())
        threshold = G.num_edges() * edge_cut
        print("edge_cut:", edge_cut)
        print("threshold:", threshold) 

        print("original:", G.num_edges())

        G_reduced = FBF(G, weight_attr, root, threshold, node_degrees) 
        total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc = get_stats(G_reduced, weight_attr, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc)

        G_reduced.clear_filters()
 
    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc


def run_focus_test_with_graphs(G, edge_cuts, weight_attr='transferred'): 
    G_r = G
    total_weight = [] 
    in_degree = []
    out_degree = []
    average_clustering = []
    nn = []
    ne = []
    wcc = []
    graphs = []

    root, nodes_degrees = selectRoot(G_r, weight_attr)

    print("root: ", root)
    for edge_cut in edge_cuts: 
        print("G.num_edges():", G.num_edges())
        threshold = G.num_edges() * edge_cut
        print("edge_cut:", edge_cut)
        print("threshold:", threshold) 

        print("original:", G.num_edges())

        G_reduced = FBF(G, weight_attr, root, threshold, nodes_degrees) 
        total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc = get_stats(G_reduced, weight_attr, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc)

        graphs.append(G_reduced)
        G_reduced.clear_filters()

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, graphs

