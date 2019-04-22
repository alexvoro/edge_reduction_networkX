# We will need some things from several places
from __future__ import division, absolute_import, print_function
import sys
import os
# We need to import the graph_tool module itself 
#sys.path.append('/usr/local/Cellar/graph-tool/2.27_6/lib/python3.7/site-packages/')
from graph_tool.all import * 
from matplotlib.pyplot import *   
from numpy.random import *  
import json
import time
seed(42)
import sampling
from sampling import WIS_graph_tool, edge_reduction_graph_tool, focus_filtering_graph_tool, SRS2_graph_tool
   
def load_g(file_name): 
    graph = load_graph(file_name)
    print("num_vertices", graph.num_vertices())
    print("num_edges", graph.num_edges())
    print("weakly_connected_components", len(label_components(graph, directed=False)[1])) 
      
    if len(label_components(graph, directed=False)[1]) != 1:
        c = label_largest_component(graph) 
        largest_component = GraphView(graph, vfilt=c) 
        g_largest_component = Graph(largest_component, prune=True)
        print("num_vertices", g_largest_component.num_vertices())
        print("num_edges", g_largest_component.num_edges()) 

        return g_largest_component
    else:
        return graph

def save_json(data): 
    print("data", data)
    print("saving json")
    with open('tests_gt__all_output.json', 'w') as outfile:
        json.dump(data, outfile)

def save_json_file(data, file_name): 
    print("data", data)
    print("saving json")
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)

def save_graph(original_file_name, graphs, edge_percentages, alg_name):
    for x in range(0, len(edge_percentages)):
        p = str(edge_percentages[x])
        p = p.replace(".", "-")
        file_name = original_file_name.replace('.graphml', '_')
        file_name = file_name + p + "_gt_" + alg_name+ '.graphml'
        print(file_name)
        graphs[x].save(file_name, fmt='graphml') 

def run_tests_save_graph(graph, file_name, data, weight_attr):
    edge_percentages = [1, 0.8, 0.5] 
    edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2, running_time2, graphs_BC = sampling.edge_reduction_graph_tool.edge_reduce_approximate_test_with_graph(graph, edge_percentages, weight_attr)
    edge_cuts_3, total_weight_3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3, running_time3, graphs_FF = sampling.focus_filtering_graph_tool.run_focus_test_with_graphs(graph, edge_percentages, weight_attr)
    edge_cuts_1, total_weight_1, in_degree1, out_degree1, average_clustering1, nn1, ne1, wcc1, running_time1, graphs_WIS = sampling.WIS_graph_tool.WIS_test_with_graph(graph, edge_percentages, weight_attr)
    edge_cuts_4, total_weight_4, in_degree4, out_degree4, average_clustering4, nn4, ne4, wcc4, running_time4, graphs_SRS2 = sampling.SRS2_graph_tool.SRS2_test_with_graphs(file_name, graph.copy(), edge_percentages, weight_attr)
     
    save_graph(file_name, graphs_WIS, edge_percentages, "WIS")
    save_graph(file_name, graphs_SRS2, edge_percentages, "SRS2")
    save_graph(file_name, graphs_FF, edge_percentages, "FF")
    save_graph(file_name, graphs_BC, edge_percentages, "BC")

def run_tests(graph, file_name, data, weight_attr): 
    #edge_percentages = [1, 0.7, 0.4] 
    #edge_percentages = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01] 
    #edge_percentages = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] 
    edge_percentages = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01] 
    #edge_percentages = [ 0.5] 

    edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2, running_time2 = run_BC(edge_percentages, graph, file_name, {}, weight_attr)
    edge_cuts_3, total_weight_3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3, running_time3 = run_FF(edge_percentages, graph, file_name, {}, weight_attr)
    edge_cuts_1, total_weight_1, in_degree1, out_degree1, average_clustering1, nn1, ne1, wcc1, running_time1 = run_WIS(edge_percentages, graph, file_name, {}, weight_attr)
    edge_cuts_4, total_weight_4, in_degree4, out_degree4, average_clustering4, nn4, ne4, wcc4, running_time4 = run_SRS2(edge_percentages, graph, file_name, {}, weight_attr)
    
    data = write_json_output(file_name, data, graph,
        edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2, running_time2,
        edge_cuts_1, total_weight_1, in_degree1, out_degree1, average_clustering1, nn1, ne1, wcc1, running_time1,
        edge_cuts_3, total_weight_3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3, running_time3,
        edge_cuts_4, total_weight_4, in_degree4, out_degree4, average_clustering4, nn4, ne4, wcc4, running_time4)
    
    file_name = file_name.replace(".graphml", ".json")
    file_name = "test_gt_output_" + file_name

    save_json_file(data, file_name)

    return data
  
def run_tests_for_files_in_folder(d, weight_attr):
    #subfolders = [f.path for f in os.scandir(d) if f.is_dir() ] 
    
    data = {}
    for file in os.listdir(d):  
        print(file) 
        print(os.path.isfile(file))
        if file.endswith(".graphml"):  
            G = load_g(os.path.join(d, file))  
            data = run_tests(G, file, data, weight_attr)

    save_json(data)

def run_tests_for_files_in_folder_save_graphs(d, weight_attr):
    #subfolders = [f.path for f in os.scandir(d) if f.is_dir() ] 
    
    data = {}
    for file in os.listdir(d):  
        print(file) 
        print(os.path.isfile(file))
        if file.endswith(".graphml"):  
            G = load_g(os.path.join(d, file))  
            run_tests_save_graph(G, file, data, weight_attr) 

def run_FF_test(graph, file_name, data, weight_attr): 
    #edge_percentages = [0.7]
    #edge_percentages = [0.3, 0.1, 0.08, 0.06]  
    edge_percentages = [1, 0.3, 0.1, 0.08, 0.06, 0.03, 0.01] 
    current_time = time.time()
    edge_cuts_3, total_weight_3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3, running_time  = sampling.focus_filtering_graph_tool.run_focus_test(graph, edge_percentages, weight_attr, True)
    print("new  took:", time.time()-current_time)  
    print(total_weight_3)

    current_time = time.time()
    edge_cuts_3, total_weight_3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3, running_time  = sampling.focus_filtering_graph_tool.run_focus_test(graph, edge_percentages, weight_attr)
    print("old  took:", time.time()-current_time)  
    print(total_weight_3)
 
    #save_json(data)
 
def run_WIS_test(graph, file_name, data, weight_attr): 
    #edge_percentages = [1, 0.7, 0.4]
    edge_percentages = [1, 0.3, 0.1, 0.08, 0.06, 0.03, 0.01] 
    current_time = time.time()
    edge_cuts_1, total_weight_1, in_degree1, out_degree1, average_clustering1, nn1, ne1, wcc1 = sampling.WIS_graph_tool.WIS_test(graph, edge_percentages, weight_attr)
    print("edge_cuts_WIS", edge_cuts_1)
    print("total_weight_WIS", total_weight_1) 
    print("wcc_WIS", wcc1)    
    print("in_degree_WIS", in_degree1)
    print("out_degree_WIS", out_degree1)
    print("average_clustering1", average_clustering1)
    print("nn1", nn1)
    print("ne1", ne1)

def run_BC(edge_percentages, graph, file_name, data, weight_attr):
    print("BC")
    edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2, running_time2 = sampling.edge_reduction_graph_tool.edge_reduce_approximate_test(graph, edge_percentages, weight_attr)
    #edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2 = [], [], [], [], [], [] , [], []
     
    print("edge_cuts_BC", edge_cuts_2)
    print("total_weight_BC", total_weight_2) 
    print("wcc_BC", wcc2)    
    print("in_degree_BC", in_degree2)
    print("out_degree_BC", out_degree2)
    print("average_clustering2", average_clustering2) 
    print("nn2", nn2)
    print("ne2", ne2)
    print("running_time_BC", running_time2)

    return  edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2, running_time2
 
def run_FF(edge_percentages, graph, file_name, data, weight_attr):
    print("FF")
    edge_cuts_3, total_weight_3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3, running_time3  = sampling.focus_filtering_graph_tool.run_focus_test(graph, edge_percentages, weight_attr)
    # edge_cuts_3, total_weight_3 = sampling.edge_reduction.edge_reduce_test(graph.copy(), edge_cuts_1, 'weight')
 
    print("edge_cuts_FF", edge_cuts_3)
    print("total_weight_FF", total_weight_3) 
    print("wcc_FF", wcc3)   
    print("running_time_FF", running_time3)
    return edge_cuts_3, total_weight_3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3, running_time3

def run_SRS2(edge_percentages, graph, file_name, data, weight_attr):
    print("SRS2")
    edge_cuts_4, total_weight_4, in_degree4, out_degree4, average_clustering4, nn4, ne4, wcc4, running_time4 = sampling.SRS2_graph_tool.SRS2_test(file_name, graph.copy(), edge_percentages, weight_attr)
    print("edge_cuts_SRS2", edge_cuts_4)
    print("total_weight_SRS2", total_weight_4) 
    print("wcc_SRS2", wcc4)   
    print("in_degree_SRS2", in_degree4)
    print("out_degree_SRS2", out_degree4)
    print("average_clustering_SRS2", average_clustering4)
    print("nn_SRS2", nn4)
    print("ne_SRS2", ne4)
    print("running_time_SRS2", running_time4)

    return edge_cuts_4, total_weight_4, in_degree4, out_degree4, average_clustering4, nn4, ne4, wcc4, running_time4 

def run_WIS(edge_percentages, graph, file_name, data, weight_attr):
    edge_cuts_1, total_weight_1, in_degree1, out_degree1, average_clustering1, nn1, ne1, wcc1, running_time1 = sampling.WIS_graph_tool.WIS_test(graph, edge_percentages, weight_attr)
    print("edge_cuts_WIS", edge_cuts_1)
    print("total_weight_WIS", total_weight_1) 
    print("wcc_WIS", wcc1)    
    print("in_degree_WIS", in_degree1)
    print("out_degree_WIS", out_degree1)
    print("average_clustering1", average_clustering1)
    print("nn1", nn1)
    print("ne1", ne1)
    print("running_time_WIS", running_time1)

    return edge_cuts_1, total_weight_1, in_degree1, out_degree1, average_clustering1, nn1, ne1, wcc1, running_time1

def write_json_output(file_name, data, graph,
    edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2, running_time2,
    edge_cuts_1, total_weight_1, in_degree1, out_degree1, average_clustering1, nn1, ne1, wcc1, running_time1,
    edge_cuts_3, total_weight_3, in_degree3, out_degree3, average_clustering3, nn3, ne3, wcc3, running_time3,
    edge_cuts_4, total_weight_4, in_degree4, out_degree4, average_clustering4, nn4, ne4, wcc4, running_time4):
    graph = {
        'number_of_nodes': graph.num_vertices(),
        'number_of_edges': graph.num_edges(), 
        'number_of_wcc': len(label_components(graph, directed=False)[1]),
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
        'running_time_WIS': running_time1,
        'edge_cuts_BC': edge_cuts_2,
        'total_weight_BC': total_weight_2,
        'in_degree_BC': in_degree2,
        'out_degree_BC': out_degree2, 
        'nn_BC': nn2,
        'ne_BC': ne2,
        'average_clustering_BC': average_clustering2,
        'wcc_BC': wcc2,
        'running_time_BC': running_time2,
        'edge_cuts_FF': edge_cuts_3,
        'total_weight_FF': total_weight_3,
        'in_degree_FF': in_degree3,
        'out_degree_FF': out_degree3,
        'nn_FF': nn3,
        'ne_FF': ne3,
        'average_clustering_FF': average_clustering3,
        'wcc_FF': wcc3,
        'running_time_FF': running_time3,
        'edge_cuts_SRS2': edge_cuts_4,
        'total_weight_SRS2': total_weight_4,
        'in_degree_SRS2': in_degree4,
        'out_degree_SRS2': out_degree4,
        'nn_SRS2': nn4,
        'ne_SRS2': ne4,
        'average_clustering_SRS2': average_clustering4,
        'wcc_SRS2': wcc4,
        'running_time_SRS2': running_time4
    }

    data[file_name] = {
        'graph': graph,
        'tests': tests
    } 

    return data

weight_attr = "lastTs"
 
#g = load_g("test_caveman_2_5.graphml")
#run_tests_for_files_in_folder("test_data_2", weight_attr)
run_tests_for_files_in_folder_save_graphs("test_data", weight_attr)
#run_tests(g, "test_caveman_2_5.graphml", {}, weight_attr)
#run_FF_test(g, "test_caveman_8_50.graphml", {}, weight_attr)
#run_WIS_test(g, "test_caveman_8_50.graphml", {}, weight_attr)
#run_tests_for_files_in_folder("test_data", weight_attr)
 
#print(g.get_edges()) 

#g = load_g("9037-12bbf821.graphml")
#run_tests(g, "9037-12bbf821.graphml", {}, weight_attr)

#g = load_g("9101-12bbf821.graphml")
#run_tests(g, "9101-12bbf821.graphml", {}, weight_attr)

#g = load_g("small.graphml")
#run_tests(g, "small.graphml", {}, weight_attr)
 
#g = load_g("huge.graphml")
#run_tests(g, "hug.graphml", {}, weight_attr)