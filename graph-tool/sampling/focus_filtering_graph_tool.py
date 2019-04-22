from graph_tool.all import *  
import numpy as np
import json
import time
import datetime
import random
import bisect
import itertools
import operator 

def selectRoot(graph, weight_attr):
    nodes = graph.get_vertices()
    edge_weight = graph.edge_properties[weight_attr]

    in_degrees =  np.array(graph.get_in_degrees(graph.get_vertices(), edge_weight))
    out_degrees =  np.array(graph.get_out_degrees(graph.get_vertices(), edge_weight)) 
    cum_weights = np.add(in_degrees, out_degrees)    
  
    nodes_degrees = np.column_stack((nodes,cum_weights))
    items = sorted(nodes_degrees, reverse=True, key=lambda x: x[1])
    root = items[:1] 

    print("root", root)
    root2 = nodes_degrees[np.argmax( np.array([x[1] for x in nodes_degrees]))]

    print("root2", root2)
    #todo replace root with root2
    v_degrees = np.column_stack(([v for v in graph.vertices()], cum_weights))

    return root[0][0], v_degrees
   
def FBF_recursive(G, tree_nodes, weight_attr, node_degrees, neighbours, lastAdded = None, node_int_degrees = None):  
    # pick a neightbour Vn+1 with a highest degree
 
    n_v = np.append(G.get_in_neighbors(lastAdded), G.get_out_neighbors(lastAdded)) 
    if(len(neighbours) != 0): 
        neighbours = neighbours[~np.in1d(neighbours[:,0].astype(int), tree_nodes)]
        #neighbours = np.array([x for x in neighbours if x[0] not in tree_nodes])
        n_v = [x for x in n_v if x not in np.array(neighbours)[:,0]  and x not in tree_nodes]  
    else: 
        n_v = [x for x in n_v if x not in tree_nodes] 
   
    n_w = node_int_degrees[n_v]
    new_neighbours = np.column_stack([n_v,n_w])
    if len(neighbours ) == 0 :
        neighbours = new_neighbours
    else:
        neighbours = np.append(neighbours, new_neighbours, axis=0)   
        
    # neighbours with the highest degree 
    top = max(neighbours, key=lambda x: x[1])

    # edges between Vn+1 and tree nodes     
    v_edges = np.vstack((G.get_in_edges(top[0]), G.get_out_edges(top[0])))
    edges_vert = np.intersect1d(np.unique([[e[0], e[1]] for e in v_edges]), tree_nodes)
  
    vn_weight = np.column_stack([edges_vert.astype(int),node_int_degrees[edges_vert.astype(int)]])
    other_end = max(vn_weight, key=lambda x: x[1])
    edge = G.edge(other_end[0], top[0])
    
    if edge is None:
        edge = G.edge(top[0], other_end[0]) 
    return edge, neighbours, top[0]
   
def dense_component_extraction(G, tree, v_delete, e_delete, threshold, weight_attr): 
    if tree.num_edges() >= threshold:
        return tree  
    
    all_edges = G.get_edges()[:,:2]
    tree_edges = tree.get_edges()[:,:2] 

    dims = np.maximum(all_edges.max(0),tree_edges.max(0))+1
    filtered_edges = all_edges[~np.in1d(np.ravel_multi_index(all_edges.T,dims),np.ravel_multi_index(tree_edges.T,dims))]
    
    # compute shortest path, keep adding the ones with the longest path 
    dist = graph_tool.topology.shortest_distance(tree, directed=False) 
    items = [(edge, dist[edge[0]].a[int(edge[1])] ) for edge in filtered_edges]   
     
    items = sorted(items, reverse=True, key=lambda x: x[1])   
    number_to_add = int(threshold - tree.num_edges()) 
    edges_to_add = items[:number_to_add]

    for edge in edges_to_add:
        e_delete[G.edge(edge[0][0], edge[0][1])] = True  
 
    tree.set_edge_filter(e_delete) 
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
 
    current_time = time.time() 

    node_int_degrees = np.add(G.get_in_degrees(G.get_vertices(), eweight=edge_weight), G.get_out_degrees(G.get_vertices(), eweight=edge_weight))
    #neighbours = np.array([])
    while len(tree_nodes) != G_num_vertices:
        edge, neighbours, top = FBF_recursive(G, tree_nodes, edge_weight, node_degrees, neighbours, top, node_int_degrees)

        tree_nodes.append(top)
        tree_edges.append(edge)
 
    print("time_spent: ", time.time()-current_time)
    for node in tree_nodes:
        v_delete[G.vertex(node)] = True
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

def run_focus_test(G, edge_cuts, weight_attr='transferred'): 
    total_weight = [] 
    in_degree = []
    out_degree = []
    average_clustering = []
    nn = []
    ne = []
    wcc = []
    running_time = []
 
    root, node_degrees = selectRoot(G, weight_attr)
    
    print("root: ", root)
    for edge_cut in edge_cuts: 
        print("G.num_edges():", G.num_edges())
        threshold = G.num_edges() * edge_cut
        print("edge_cut:", edge_cut)
        print("threshold:", threshold) 

        print("original:", G.num_edges())

        current_time = time.time() 
        G_reduced = FBF(G, weight_attr, root, threshold, node_degrees) 
        time_spent = time.time()-current_time
        print("time_spent:", time_spent)
        total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time = get_stats(G_reduced, weight_attr, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, time_spent)

        G_reduced.clear_filters()
 
    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time


def run_focus_test_with_graphs(G, edge_cuts, weight_attr='transferred'): 
    G_r = G
    total_weight = [] 
    in_degree = []
    out_degree = []
    average_clustering = []
    nn = []
    ne = []
    wcc = []
    running_time = []
    graphs = []

    root, nodes_degrees = selectRoot(G_r, weight_attr)

    print("root: ", root)
    for edge_cut in edge_cuts: 
        print("G.num_edges():", G.num_edges())
        threshold = G.num_edges() * edge_cut
        print("edge_cut:", edge_cut)
        print("threshold:", threshold) 

        print("original:", G.num_edges())

        current_time = time.time() 
        G_reduced = FBF(G, weight_attr, root, threshold, nodes_degrees) 
        time_spent = time.time()-current_time
        total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time = get_stats(G_reduced, weight_attr, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, time_spent)

        graphs.append(G_reduced)
        G_reduced.clear_filters()

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc, running_time, graphs

