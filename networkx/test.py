import networkx as nx 
import numpy as np
import json
import time
import os
import sampling
from math import log10
from sampling import WIS, extensions, edge_reduction, focus_filtering, SRS2

def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.load(f) #, default={'sender': 'source'})
        _attrs = dict(source='sender', target='receiver', name='guid',
              key='guid', link='links')
    #return nx.readwrite.node_link_graph(js_graph, {'link': 'links', 'source': 'sender', 'target': 'receiver', 'key': 'guid'})
    return nx.readwrite.node_link_graph(js_graph, directed=True, multigraph=False, attrs={'link': 'links', 'source': 'sender', 'target': 'receiver', 'key': 'guid', 'name': 'guid'} )

def write_json_file(G, filename):
    _attrs = dict(source='sender', target='receiver', name='guid',
                key='guid', link='links')
    s2 = nx.readwrite.node_link_data(G, attrs={'link': 'links', 'source': 'sender', 'target': 'receiver', 'key': 'guid', 'name': 'guid'})
 
    with open(filename, 'w') as outfile:
        json.dump(s2, outfile)
 
def read_files_in_folder(d = 'network_data'):
    #subfolders = [f.path for f in os.scandir(d) if f.is_dir() ] 
  
    for file in os.listdir(d):    
        if file.endswith(".json"):  
            G = read_json_file(os.path.join(d, file))
            print_graph_data(G)

def run_tests_for_files_in_folder(d, weight_attr):
    #subfolders = [f.path for f in os.scandir(d) if f.is_dir() ] 
    
    data = {}
    for file in os.listdir(d):  
        print(file) 
        print(os.path.isfile(file))
        if file.endswith(".json"):  
            G = read_json_file(os.path.join(d, file)) 
            print_graph_data(G)
            data = run_tests(G, file, data, weight_attr)

    save_json(data)
 
def run_tests_for_file(file, weight_attr):
    #subfolders = [f.path for f in os.scandir(d) if f.is_dir() ] 
    
    data = {} 
    if file.endswith(".json"): 
        print(file)
        G = read_json_file(file) 
        print_graph_data(G)
        data = run_tests(G, file, data, weight_attr)

    save_json(data)

def print_graph_data(G): 
    print(nx.info(G))  
    print("is_strongly_connected", nx.is_strongly_connected(G))
    print("number_connected_components", nx.number_weakly_connected_components(G))
    print()
    print("____________________________________________")

def save_json(data): 
    with open('tests_output_nx.json', 'w') as outfile:
        json.dump(data, outfile)
 
def run_tests(graph, file_name, data, weight_attr): 
    #edge_percentages = [1, 0.8, 0.6, 0.4, 0.2, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002 ]  
    #edge_percentages = [0.06, 0.05, 0.04, 0.03, 0.02, 0.01] 
    #edge_percentages = [ 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01] 
    edge_percentages = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    #edge_percentages = [1, 0.3, 0.1, 0.08, 0.06, 0.03, 0.01]   
    #edge_percentages = [1, 0.7, 0.4]

    graph = graph.subgraph(max(nx.weakly_connected_components(graph), key=len))  
    print(nx.info(graph)) 

    print("BC")
    edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2  = sampling.edge_reduction_old.edge_reduce_approximate_test(file_name, graph.copy(), edge_percentages, weight_attr)
    #edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2 = [], [], [], [], [], [] , [], []
     
    print("edge_cuts_BC", edge_cuts_2)
    print("total_weight_BC", total_weight_2) 
    print("wcc_BC", wcc2)    
    print("in_degree_BC", in_degree2)
    print("out_degree_BC", out_degree2)
    print("average_clustering_BC", average_clustering2)
    print("nn_BC", nn2)
    print("ne_BC", ne2)

    print("WIS")
    edge_cuts_1, total_weight_1, in_degree1, out_degree1, average_clustering1, nn1, ne1, wcc1 = sampling.WIS.WIS_test(file_name, graph.copy(), edge_percentages, weight_attr)
    print("edge_cuts_WIS", edge_cuts_1)
    print("total_weight_WIS", total_weight_1) 
    print("wcc_WIS", wcc1)    
    print("in_degree_WIS", in_degree1)
    print("out_degree_WIS", out_degree1)
    print("average_clustering1", average_clustering1)
    print("nn1", nn1)
    print("ne1", ne1)

    print("FF")
    edge_cuts_3, total_weight_3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3  = sampling.focus_filtering.run_focus_test(graph, edge_percentages, weight_attr)
 
    print("edge_cuts_FF", edge_cuts_3)
    print("total_weight_FF", total_weight_3) 
    print("wcc_FF", wcc3)   

    print("SRS2")
    edge_cuts_4, total_weight_4, in_degree4, out_degree4, average_clustering4, nn4, ne4, wcc4 = sampling.SRS2.SRS2_test(file_name, graph.copy(), edge_percentages, weight_attr)
    print("edge_cuts_SRS2", edge_cuts_4)
    print("total_weight_SRS2", total_weight_4) 
    print("wcc_SRS2", wcc4)   
    print("in_degree_SRS2", in_degree4)
    print("out_degree_SRS2", out_degree4)
    print("average_clustering_SRS2", average_clustering4)
    print("nn_SRS2", nn4)
    print("ne_SRS2", ne4)

    graph = {
        'number_of_nodes': graph.number_of_nodes(),
        'number_of_edges': graph.number_of_edges(),
        'number_of_selfloops': graph.number_of_selfloops(),
        'number_of_wcc': nx.number_weakly_connected_components(graph),
    }

    tests= {
        'edge_cuts_WIS': edge_cuts_1,
        'total_weight_WIS': total_weight_1,
        'in_degree_WIS': in_degree1,
        'out_degree_WIS': out_degree1,
        'nn_WIS': nn1,
        'ne_WIS': ne1,
        'average_clustering_WIS': average_clustering1,
        'wcc_WIS': wcc1,
        'edge_cuts_BC': edge_cuts_2,
        'total_weight_BC': total_weight_2,
        'in_degree_BC': in_degree2,
        'out_degree_BC': out_degree2, 
        'nn_BC': nn2,
        'ne_BC': ne2,
        'average_clustering_BC': average_clustering2,
        'wcc_BC': wcc2,
        'edge_cuts_FF': edge_cuts_3,
        'total_weight_FF': total_weight_3,
        'in_degree_FF': in_degree3,
        'out_degree_FF': out_degree3,
        'nn_FF': nn3,
        'ne_FF': ne3,
        'average_clustering_FF': average_clustering3,
        'wcc_FF': wcc3,
        'edge_cuts_SRS2': edge_cuts_4,
        'total_weight_SRS2': total_weight_4,
        'in_degree_SRS2': in_degree4,
        'out_degree_SRS2': out_degree4,
        'nn_SRS2': nn4,
        'ne_SRS2': ne4,
        'average_clustering_SRS2': average_clustering4,
        'wcc_SRS2': wcc4
    }

    data[file_name] = {
        'graph': graph,
        'tests': tests
    } 

    return data
 
def run_test_for_file(graph, file_name, data, weight_attr): 
    edge_percentages = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]  
     
    print("WIS")
    edge_cuts_1, total_weight_1, in_degree1, out_degree1, average_clustering1, nn1, ne1, wcc1 = sampling.WIS.WIS_test(file_name, graph.copy(), edge_percentages, weight_attr)
    print("edge_cuts_WIS", edge_cuts_1)
    print("total_weight_WIS", total_weight_1) 
    print("wcc_WIS", wcc1)    
    print("in_degree_WIS", in_degree1)
    print("out_degree_WIS", out_degree1)
    print("average_clustering1", average_clustering1)
    print("nn1", nn1)
    print("ne1", ne1)

def save_graph(original_file_name, graphs, edge_percentages, alg_name):
    for x in range(0, len(edge_percentages)):
        p = str(edge_percentages[x])
        p = p.replace(".", "-")
        file_name = original_file_name.replace('.json', '_')
        file_name = file_name + p + "_" + alg_name + '.json'
        print(file_name)
        write_json_file(graphs[x], file_name)

def run_test_for_file_save_graph(graph, file_name, data, weight_attr): 
    edge_percentages = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]  
      
    print(nx.info(graph))
    G_ud = graph.to_undirected(as_view=True)
    print("is_connected", nx.is_connected(G_ud)) 
    
    graph = graph.subgraph(max(nx.weakly_connected_components(graph), key=len))
    print(nx.info(graph))
  
  
    edge_cuts, total_weight, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc, graphs  = sampling.edge_reduction.edge_reduce_approximate_test_with_graph(graph.copy(), edge_percentages, weight_attr)
    edge_cuts, total_weight, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc, graphs_WIS  = sampling.WIS.WIS_test_with_graph(graph.copy(), edge_percentages, weight_attr)
    edge_cuts3, total_weight3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3, graphs_SRS2 = sampling.SRS2.SRS2_test_with_graphs(graph.copy(), edge_percentages, weight_attr)
    edge_cuts4, total_weight4, in_degree4, out_degree4, average_clustering4, nn4, ne4, wcc4, graphs_FF = sampling.focus_filtering.run_focus_test_with_graphs(graph.copy(), edge_percentages, weight_attr)
      
    save_graph(file_name, graphs_BC, edge_percentages, "BC")
    save_graph(file_name, graphs_WIS, edge_percentages, "WIS")
    save_graph(file_name, graphs_SRS2, edge_percentages, "SRS2")
    save_graph(file_name, graphs_FF, edge_percentages, "FF")

def convert_file_json_to_graphml(d, file):
    G = read_json_file(os.path.join(d, file)) 
    print_graph_data(G)
    file_name = file.replace("json", "graphml")
    nx.write_graphml_lxml(G, os.path.join(d, file_name))
 
def convert_files_json_to_graphml(d):
    #subfolders = [f.path for f in os.scandir(d) if f.is_dir() ] 
    
    data = {} 
    for file in os.listdir(d):  
        # print(file) 
        print(os.path.isfile(file))
        if file.endswith(".json"):  
            convert_file_json_to_graphml(d, file)

def convert_file_graphml_to_json(d, file): 
    G = nx.read_graphml(os.path.join(d, file)) 
    print_graph_data(G)
    file_name = file.replace("graphml","json") 
    
    write_json_file(G, os.path.join(d, file_name))

# convert_file_graphml_to_json("test_data_json","9001-a5226a35-269d64c4_1_gt__with_id_0-005_gt_BC.graphml")

def convert_files_graphml_to_json(d):
    #subfolders = [f.path for f in os.scandir(d) if f.is_dir() ] 
    
    data = {} 
    files = os.listdir(d) 

    pairs = []
    for file in files:
        if file.endswith(".graphml"):  

            # Use join to get full file path.
            location = os.path.join(d, file)

            # Get size and add to list of tuples.
            size = os.path.getsize(location)
            pairs.append((file, size))

    pairs.sort(key=lambda s: s[0])
    for file, size in pairs:  
        print(file) 
        print(os.path.isfile(file))
        if file.endswith(".graphml"):  
            convert_file_graphml_to_json(d, file) 
 

def add_edge_ids(G, d, filename):
    ints = list(range(0, G.number_of_edges()))
    ids = 0
    for edge in G.edges:
        G[edge[0]][edge[1]]['id'] = ids
        ids = ids + 1 

    print(len(ints))
    print( G.number_of_edges()) 
    ids = list((nx.get_edge_attributes(G, 'id')).values())
    
    print("write_json_file")
    filename = filename.replace(".json","_with_id.json") 
    write_json_file(G, os.path.join(d, filename))  

def add_ids_to_files_in_folder(d ): 
    betw = []
    shortest_path = []
    files = []

    for file in os.listdir(d):    
        print(file) 
        if file.endswith(".json"):  
            G = read_json_file(os.path.join(d, file))
            print(nx.info(G))   
            print("g number_of_edges():", G.number_of_edges()) 
            #graph = G.subgraph(max(nx.weakly_connected_components(G), key=len))  
            # print("graph number_of_edges():", graph.number_of_edges()) 
            
            add_edge_ids(G, d, file)


def ids_to_json(G, d, filename): 
    ids = list((nx.get_edge_attributes(G, 'id')).values())
    
    print("write_json_file")
    filename = filename.replace(".graphml","_with_id.json") 
    write_json_file(G, os.path.join(d, filename)) 

    filename = filename.replace(".json","reduced.json") 
    with open(os.path.join(d, filename), 'w') as outfile:
        json.dump(ids, outfile)

def convert_to_json_reduced_files_in_folder(d ):
    betw = []
    shortest_path = []
    files = []

    for file in os.listdir(d):    
        print(file) 
        if file.endswith(".graphml"):  
            G = nx.read_graphml(os.path.join(d, file))
            print(nx.info(G))   
            print("g number_of_edges():", G.number_of_edges()) 
             
            ids_to_json(G, d, file)

# add ids to edges
#d = "test_data_real"
#file = "9031-a023an12.json"
#G = read_json_file(os.path.join(d, file))
#G = G.subgraph(max(nx.weakly_connected_components(G), key=len))  
            
#add_edge_ids(G, d, file)
#convert_file_json_to_graphml(d, "9031-a023an12_with_id.json") 
  
# make a file with ids only from the full reduced graph
# d = "test_data_real"
# file = "9031-a023an12_with_id_0-02_gt_BC.graphml"

# G = nx.read_graphml(os.path.join(d, file))
# ids_to_json(G, d, file)

def run_assortativity_for_files_in_folder(d):
    assortativity = [] 

    files = os.listdir(d) 

    pairs = []
    for file in files:
        if file.endswith(".json"):
            print(file)
            # Use join to get full file path.
            location = os.path.join(d, file)

            # Get size and add to list of tuples.
            size = os.path.getsize(location)
            pairs.append((file, size))

    pairs.sort(key=lambda s: s[0])
    for file, size in pairs:   
        if file.endswith(".json"):  
            G = read_json_file(os.path.join(d, file))
            ass = nx.attribute_assortativity_coefficient(G,'site')  
            assortativity.append(ass)   
            # print(file, ass)
    print(assortativity)
 
def measure_files_in_folder(d, weight_attr):
    #subfolders = [f.path for f in os.scandir(d) if f.is_dir() ] 
    
    betw = []
    shortest_path = []
    files = []

    for file in os.listdir(d):    
        if file.endswith(".json"):  
            G = read_json_file(os.path.join(d, file))
            print("number_connected_components", nx.number_weakly_connected_components(G))
            G = G.subgraph(max(nx.weakly_connected_components(G), key=len))  
            print(file)
            print(nx.info(G))   
            files.append(file) 

            bet = measure_betweenness(G, weight_attr)
            print(bet)
            betw.append(bet)
            shortest = measure_shortest(G) 
            print(shortest)
            shortest_path.append(shortest)

    print(files)
    print(betw)
    print(shortest_path)

def measure_betweenness(G, weight_attr):
    c = 10
    take_count = int(c * log10(nx.number_of_nodes(G)))  

    current_time = time.time()
    bet_cent_edges = nx.edge_betweenness_centrality(G, k=take_count, weight=weight_attr) 
    time_spent = time.time()-current_time 

    return time_spent

def measure_shortest(G):
    current_time = time.time()
    tree_ud = G.to_undirected(as_view=True) 
      
    lengths = dict(nx.all_pairs_shortest_path_length(tree_ud))  
    time_spent = time.time()-current_time 

    return time_spent
 