
#import matplotlib.pyplot as plt
import networkx as nx 
import numpy as np
import json
import time
import os
import sampling
from sampling import WIS, edge_reduction, edge_reduction_old, focus_filtering, SRS2

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

def get_graph(filename): 
    G = read_json_file(filename)
    print(nx.info(G))

    G_ud = G.to_undirected()
    print("is_connected", nx.is_connected(G_ud))
    print("is_strongly_connected", nx.is_strongly_connected(G))
    print("number_connected_components", nx.number_connected_components(G_ud))

    G_mc = list(G_ud.subgraph(c).copy() for c in nx.connected_components(G_ud))
    # Connected components are sorted in descending order of their size
    G_components = list(nx.connected_component_subgraphs(G_ud))
    _components = (G_ud.subgraph(c) for c in nx.connected_components(G_ud))

    # todo: do for-loop to run algorithm for all components
    #G_mc = G_components[0]
    G_mc = G_ud

    print("is_connected", nx.is_connected(G_mc)) 
    print("number_connected_components", nx.number_connected_components(G_mc))
    print("dencity", nx.density(G_mc)) 

    G = nx.Graph(G)
        
    print(nx.info(G))

def read_files_in_folder(d = 'network_data'):
    #subfolders = [f.path for f in os.scandir(d) if f.is_dir() ] 
  
    for file in os.listdir(d):    
        if os.path.isfile(file) and file.endswith(".json"):  
            G = read_json_file(file) 
            print(nx.info(G))
            G_ud = G.to_undirected()
            print("is_connected", nx.is_connected(G_ud))
            print("is_strongly_connected", nx.is_strongly_connected(G))
            print("number_connected_components", nx.number_connected_components(G_ud))
            print()
            print("____________________________________________")


def run_tests_for_files_in_folder(d, weight_attr):
    #subfolders = [f.path for f in os.scandir(d) if f.is_dir() ] 
    
    data = {}
    for file in os.listdir(d):  
        print(file)
        print(file.endswith(".json"))
        print(os.path.isfile(file))
        if file.endswith(".json"): 
            print(file)
            G = read_json_file(os.path.join(d, file)) 
            print(nx.info(G))
            G_ud = G.to_undirected()
            print("is_connected", nx.is_connected(G_ud))
            print("is_strongly_connected", nx.is_strongly_connected(G))
            print("number_connected_components", nx.number_connected_components(G_ud))
            print()
            print("____________________________________________")
            data = run_tests(G, file, data, weight_attr)

    save_json(data)
 
def run_tests_for_file(file, weight_attr):
    #subfolders = [f.path for f in os.scandir(d) if f.is_dir() ] 
    
    data = {} 
    if file.endswith(".json"): 
        print(file)
        G = read_json_file(file) 
        print(nx.info(G))  

        print("is_strongly_connected", nx.is_strongly_connected(G))
        print("number_connected_components", nx.number_weakly_connected_components(G))
        print()
        print("____________________________________________")
        data = run_tests(G, file, data, weight_attr)

    save_json(data)

def save_json(data): 
    with open('tests_output_9101-1383f38c.json', 'w') as outfile:
        json.dump(data, outfile)
 
def run_tests(graph, file_name, data, weight_attr): 
    #edge_percentages = [1, 0.8, 0.6, 0.4, 0.2, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002 ]  
    edge_percentages = [  0.06, 0.05, 0.04, 0.03, 0.02, 0.01] 
    #edge_percentages = [ 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01] 
    #edge_percentages = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    #edge_percentages = [1, 0.8, 0.5, 0.3, 0.1, 0.08, 0.06, 0.03, 0.01]   
    #graph = read_json_file("real_data_small.json")
    
    #G_ud = graph.to_undirected()
    #print("is_connected", nx.is_connected(G_ud))
    #graph1 = list(nx.weakly_connected_component_subgraphs(graph))[0]

    graph = graph.subgraph(max(nx.weakly_connected_components(graph), key=len))  
    print(nx.info(graph))
    #weight_attr = weight_attr
    #weight_attr = 'lastTs'

    #print("BC")
    #edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2  = sampling.edge_reduction_old.edge_reduce_approximate_test(graph.copy(), edge_percentages, weight_attr)

    #print("edge_cuts_BC", edge_cuts_2)
    #print("total_weight_BC", total_weight_2) 
    #print("wcc_BC", wcc2)    
    #print("in_degree_BC", in_degree2)
    #print("out_degree_BC", out_degree2)
    #print("average_clustering2", average_clustering2)
    #print("nn2", nn2)
    #print("ne2", ne2)
    edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2 = [], [], [], [], [], [] , [], []
    print("FF")
    edge_cuts_3, total_weight_3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3  = sampling.focus_filtering.run_focus_test(graph.copy(), edge_percentages, weight_attr)
    # edge_cuts_3, total_weight_3 = sampling.edge_reduction.edge_reduce_test(graph.copy(), edge_cuts_1, 'weight')
 
    print("edge_cuts_FF", edge_cuts_3)
    print("total_weight_FF", total_weight_3) 
    print("wcc_FF", wcc3)   

    print("WIS")
    edge_cuts_1, total_weight_1, in_degree1, out_degree1, average_clustering1, nn1, ne1, wcc1 = sampling.WIS.WIS_test(graph.copy(), edge_percentages, weight_attr)
    print("edge_cuts_WIS", edge_cuts_1)
    print("total_weight_WIS", total_weight_1) 
    print("wcc_WIS", wcc1)    
    print("in_degree_WIS", in_degree1)
    print("out_degree_WIS", out_degree1)
    print("average_clustering1", average_clustering1)
    print("nn1", nn1)
    print("ne1", ne1)

    print("SRS2")
    edge_cuts_4, total_weight_4, in_degree4, out_degree4, average_clustering4, nn4, ne4, wcc4 = sampling.SRS2.SRS2_test(graph.copy(), edge_percentages, weight_attr)
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
    #edge_percentages = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]  
    #edge_percentages = [1, 0.8, 0.5, 0.3, 0.1, 0.08, 0.06, 0.03, 0.01]   
    edge_percentages = [0.03, 0.02, 0.01] 
    #edge_percentages = [1, 0.8, 0.6, 0.4, 0.2, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002 ]  
    
    #edge_percentages = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01] 
 
    #graph = read_json_file("real_data_small.json")
    print(nx.info(graph)) 
    graph = list(nx.weakly_connected_component_subgraphs(graph))[0]
    print(nx.info(graph))
 
    #weight_attr = 'lastTs'
  
    #print("BC")
    #edge_cuts, total_weight, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc  = sampling.edge_reduction.edge_reduce_approximate_test(graph.copy(), edge_percentages, weight_attr)

    #edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc  = sampling.edge_reduction_old.edge_reduce_approximate_test(graph.copy(), edge_percentages, weight_attr)

    print("FF")
    edge_cuts, total_weight, in_degree, out_degree, average_clustering, nn, ne, wcc  = sampling.focus_filtering.run_focus_test(graph.copy(), edge_percentages, weight_attr)
    # edge_cuts_3, total_weight_3 = sampling.edge_reduction.edge_reduce_test(graph.copy(), edge_cuts_1, 'weight')
 
    print("edge_cuts_FF", edge_cuts)
    print("total_weight_FF", total_weight) 
    print("in_degree", in_degree) 
    print("out_degree", out_degree) 
    print("average_clustering", average_clustering) 
    print("nn", nn) 
    print("ne", ne) 
    print("wcc_FF", wcc)   

def save_graph(original_file_name, graphs, edge_percentages, alg_name):
    for x in range(0, len(edge_percentages)):
        p = str(edge_percentages[x])
        p = p.replace(".", "-")
        file_name = original_file_name.replace('.json', '_')
        file_name = file_name + p + "_" + alg_name + '.json'
        print(file_name)
        write_json_file(graphs[x], file_name)

def run_test_for_file_save_graph(graph, file_name, data, weight_attr): 
    #edge_percentages = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]  
    #edge_percentages = [1, 0.8, 0.5, 0.3, 0.1, 0.08, 0.06, 0.03, 0.01]   
    edge_percentages = [1, 0.8, 0.5] 
    #graph = read_json_file("real_data_small.json")
    print(nx.info(graph))
    G_ud = graph.to_undirected()
    print("is_connected", nx.is_connected(G_ud))
    graph = list(nx.weakly_connected_component_subgraphs(graph))[0]
    print(nx.info(graph))
 
    #weight_attr = 'lastTs'
  
    print("BC")
    #edge_cuts, total_weight, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc  = sampling.edge_reduction.edge_reduce_approximate_test(graph.copy(), edge_percentages, weight_attr)

    #edge_cuts, total_weight, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc, graphs  = sampling.edge_reduction.edge_reduce_approximate_test_with_graph(graph.copy(), edge_percentages, weight_attr)
    edge_cuts, total_weight, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc, graphs_WIS  = sampling.WIS.WIS_test_with_graph(graph.copy(), edge_percentages, weight_attr)
    edge_cuts3, total_weight3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3, graphs_SRS2 = sampling.SRS2.SRS2_test_with_graphs(graph.copy(), edge_percentages, weight_attr)
    edge_cuts4, total_weight4, in_degree4, out_degree4, average_clustering4, nn4, ne4, wcc4, graphs_FF = sampling.focus_filtering.run_focus_test_with_graphs(graph.copy(), edge_percentages, weight_attr)
    
    #graph = sampling.edge_reduction.edge_reduce_approximate(graph.copy(), 0.5, weight_attr)
    #print("FF")
    #edge_cuts_3, total_weight_3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3  = sampling.focus_filtering.run_focus_test(graph.copy(), edge_percentages, weight_attr)
    # edge_cuts_3, total_weight_3 = sampling.edge_reduction.edge_reduce_test(graph.copy(), edge_cuts_1, 'weight')
 
    #print("edge_cuts_FF", edge_cuts)
    #print("total_weight_FF", total_weight) 
    #print("wcc_FF", wcc)   

    save_graph(file_name, graphs_WIS, edge_percentages, "WIS")
    save_graph(file_name, graphs_SRS2, edge_percentages, "SRS2")
    save_graph(file_name, graphs_FF, edge_percentages, "FF")

#run_tests_for_files_in_folder('test_data', 'lastTs')
#G = read_json_file("test_data/test_caveman_8_50.json") 
#run_tests_for_file("test_data/test_caveman_8_50.json", "lastTs")

G = read_json_file("9101-1383f38c.json") 
run_tests_for_file("9101-1383f38c.json", "lastTs")


#G = read_json_file("real_data_small.json") 
#run_test_for_file(G, "real_data_small.json", {}, "lastTs") 

#G = read_json_file("test_data/test_caveman_8_50.json") 
#run_test_for_file(G, "test_caveman_8_50.json", {}, "lastTs") 

#G = read_json_file("real_data_small.json") 
#run_test_for_file(G, "real_data_small.json", {}, "lastTs") 

#G = read_json_file("test_data/test_caveman_2_5.json") 
#run_test_for_file_save_graph(G, "test_caveman_2_5.json", {}, "lastTs") 

#G = read_json_file("test_data/test_caveman_2_5.json") 
#run_test_for_file(G, "test_caveman_2_5.json", {}, "lastTs") 