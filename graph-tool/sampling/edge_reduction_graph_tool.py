import json
import time
import datetime
from math import log10
from graph_tool.all import *  
import numpy as np

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

def remove_edges(G_reduced, items, edges_max_goal):
    current_time = time.time()
    removed_edges = []
    #sorted_bet_cent_edges = sorted(items,reverse=False) 
    sorted_bet_cent_edges_ind = np.argsort(items.a, axis=None)
    sorted_bet_cent_edges_ind = np.unravel_index(np.argsort(items.a, axis=None), items.a.shape)
    sorted_bet_cent_edges = G_reduced.get_edges()[sorted_bet_cent_edges_ind]
  
    time_spent = time.time()-current_time 
 
    #print("count :", len(list(sorted_bet_cent_edges)))
    print("sorting took:", time.time()-current_time)
 
    for bet_cent in sorted_bet_cent_edges:  
        if (G_reduced.num_edges() <= edges_max_goal):
            print("done :", G_reduced.num_edges())
            break
        #ttt = G_reduced.has_node(bet_cent[0])
        v0 = G_reduced.vertex(bet_cent[0])
        v1 = G_reduced.vertex(bet_cent[1])
        if (v0.in_degree() + v0.out_degree()) > 2 and (v1.in_degree() + v1.out_degree()) > 2:
            G_reduced.remove_edge(G_reduced.edge(bet_cent[0],bet_cent[1]))
            removed_edges.append(bet_cent) 

    time_spent = time.time()-current_time
    print("for loop took : ", time_spent)
    current_time = time.time()

    return G_reduced, removed_edges
 

def edge_reduce(G, edges_max_goal, weight_attr): 
    cent = graph_tool.centrality.betweenness(G, pivots=G.get_vertices()[nodes_rand], vprop=None, eprop=None, weight=None, norm=True)
    bet_cent_edges = cent[1]
    G_reduced, removed_edges = remove_edges(graph, bet_cent_edges, edges_max_goal)
    G_reduced = postprocess(G_reduced, removed_edges)
    return G_reduced 
 
def edge_reduce_approximate(G, edges_max_goal, weight_attr): 
    c = 10
    take_count = int(c * log10(G.num_vertices())) 
    nodes_rand = np.random.choice(G.num_vertices(), take_count)
    edge_weight = G.edge_properties[weight_attr]

    cent = graph_tool.centrality.betweenness(G, pivots=G.get_vertices()[nodes_rand], vprop=None, eprop=None, weight=edge_weight, norm=True)
    bet_cent_edges = cent[1]
    G_reduced, removed_edges = remove_edges(graph, bet_cent_edges, edges_max_goal)
    G_reduced = postprocess(G_reduced, removed_edges)
    return G_reduced

def get_in_degree(G): 
    return (sum(G.get_in_degrees(G.get_vertices()) )/float(G.num_vertices()))

def get_out_degree(G): 
    return (sum(G.get_out_degrees(G.get_vertices())/float(G.num_vertices())))

def postprocess(G_reduced, items):
    wcc = weakly_connected_components(G_reduced) 
    number_wcc = sum(1 for c in wcc)
    if number_wcc == 1:
        #print("****** already 1 component ")
        return G_reduced

    current_time = time.time()
    #print("items", items)
    _components = [c for c in wcc]
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
                G_reduced.add_edge(G_reduced.edge(edge[0],edge[1]))
                _components = weakly_connected_components(G_reduced) 
                number_wcc = len(_components)
                # try _components = nx.weakly_connected_components(G_reduced)
                break
     
    time_spent = time.time()-current_time
    print("postprocessing took : ", time_spent)
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

def edge_reduce_approximate_test(G, edge_cuts, weight_attr='transferred'):
    c = 10
    take_count = int(c * log10(G.num_vertices())) 
    nodes_rand = np.random.choice(G.num_vertices(), take_count)
    edge_weight = G.edge_properties[weight_attr]

    cent = graph_tool.centrality.betweenness(G, pivots=G.get_vertices()[nodes_rand], vprop=None, eprop=None, weight=edge_weight, norm=True)
    bet_cent_edges = cent[1]  
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
        edges_max_goal = G.num_edges() * edge_cut
        graph = Graph(G)
        print("original:", G.num_edges())
        G_reduced, removed_edges = remove_edges(graph, bet_cent_edges, edges_max_goal)
        G_reduced = postprocess(G_reduced, removed_edges)
 
        time_spent = time.time()-current_time 
        total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc = get_stats(G_reduced, weight_attr, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc)

        print("num edges: ", G_reduced.num_edges())  
        #print("weight: ", G_reduced.size())
        #print("weight: ", G_reduced.size(weight=weight_attr)) 

    return edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc

 