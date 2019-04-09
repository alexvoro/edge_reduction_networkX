# We will need some things from several places
from __future__ import division, absolute_import, print_function
import sys
import os
# We need to import the graph_tool module itself 
#sys.path.append('/usr/local/Cellar/graph-tool/2.27_6/lib/python3.7/site-packages/')
from graph_tool.all import * 
from matplotlib.pyplot import *   
from numpy.random import *  
seed(42)
import sampling
from sampling import WIS_graph_tool, edge_reduction_graph_tool, extensions
 
# let's construct a Price network (the one that existed before Barabasi). It is
# a directed network, with preferential attachment. The algorithm below is
# very naive, and a bit slow, but quite simple.

# We start with an empty, directed graph
#graph_tool.set_fast_edge_removal(fast=True)
g = Graph()
#g2 = load_graph("9101-12bbf821.graphml")
#print("num_vertices",   g2.num_vertices())
#print("num_edges",   g2.num_edges())

def load_g(file_name): 
    graph = load_graph(file_name)
    print("num_vertices", graph.num_vertices())
    print("num_edges", graph.num_edges())
    print("weakly_connected_components", extensions.number_weakly_connected_components(graph))
    max_component = max(extensions.weakly_connected_components(graph), key=len)

    graph_largest_component = GraphView(graph, vfilt=lambda v: v in max_component)
    print("num_vertices", graph_largest_component.num_vertices())
    print("num_edges", graph_largest_component.num_edges())
    #tt = graph_tool.topology.label_largest_component(graph)

    #u = graph_tool.GraphView(graph, vfilt=tt)   # extract the largest component as a graph
    #print(u.num_vertices())

    #g = graph_tool.GraphView(graph, vfilt=graph_tool.topology.label_largest_component(graph))
    #print("num_vertices", g.num_vertices())
    #print("num_edges", g.num_edges())
    return graph_largest_component

def run_tests(graph, file_name, data, weight_attr): 
    edge_percentages = [1, 0.7, 0.4]
    run_BC(edge_percentages, graph, file_name, {}, weight_attr)

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
    edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2  = sampling.edge_reduction_graph_tool.edge_reduce_approximate_test(graph, edge_percentages, weight_attr)
    #edge_cuts_2, total_weight_2, in_degree2, out_degree2, average_clustering2, nn2, ne2, wcc2 = [], [], [], [], [], [] , [], []
     
    print("edge_cuts_BC", edge_cuts_2)
    print("total_weight_BC", total_weight_2) 
    print("wcc_BC", wcc2)    
    print("in_degree_BC", in_degree2)
    print("out_degree_BC", out_degree2)
    print("average_clustering2", average_clustering2)
    print("nn2", nn2)
    print("ne2", ne2)

weight_attr = "lastTs" 
#g = load_g("test_caveman_8_50.graphml")
#run_tests(g, "test_caveman_8_50.graphml", {}, weight_attr)
#print(g.get_edges())

#g = load_g("9101-1383f38c.graphml")
#run_tests(g, "9101-1383f38c.graphml", {}, weight_attr)

#g = load_g("9101-12bbf821.graphml")
#run_tests(g, "9101-12bbf821.graphml", {}, weight_attr)

g = load_g("small.graphml")
run_tests(g, "small.graphml", {}, weight_attr)
g = load_g("huge.graphml")
run_tests(g, "hug.graphml", {}, weight_attr)

def load_test_graph(): 
    # We want also to keep the age information for each vertex and edge. For that
    # let's create some property maps
    v_age = g.new_vertex_property("int")
    e_age = g.new_edge_property("int")

    # The final size of the network
    N = 100000

    # We have to start with one vertex
    v = g.add_vertex()
    v_age[v] = 0

    # we will keep a list of the vertices. The number of times a vertex is in this
    # list will give the probability of it being selected.
    vlist = [v]

    # let's now add the new edges and vertices
    for i in range(1, N):
        # create our new vertex
        v = g.add_vertex()
        v_age[v] = i

        # we need to sample a new vertex to be the target, based on its in-degree +
        # 1. For that, we simply randomly sample it from vlist.
        i = randint(0, len(vlist))
        target = vlist[i]

        # add edge
        e = g.add_edge(v, target)
        e_age[e] = i

        # put v and target in the list
        vlist.append(target)
        vlist.append(v)

    # now we have a graph!

    # let's do a random walk on the graph and print the age of the vertices we find,
    # just for fun.

    v = g.vertex(randint(0, g.num_vertices()))
    while True:
        print("vertex:", int(v), "in-degree:", v.in_degree(), "out-degree:",
            v.out_degree(), "age:", v_age[v])

        if v.out_degree() == 0:
            print("Nowhere else to go... We found the main hub!")
            break

        n_list = []
        for w in v.out_neighbors():
            n_list.append(w)
        v = n_list[randint(0, len(n_list))]

    # let's save our graph for posterity. We want to save the age properties as
    # well... To do this, they must become "internal" properties:

    g.vertex_properties["age"] = v_age
    g.edge_properties["age"] = e_age

    # now we can save it
    g.save("price.xml.gz")


    # Let's plot its in-degree distribution
    in_hist = graph_tool.stats.vertex_hist(g, "in")

    y = in_hist[0]
    err = np.sqrt(in_hist[0])
    err[err >= y] = y[err >= y] - 1e-2

    figure(figsize=(6,4))
    errorbar(in_hist[1][:-1], in_hist[0], fmt="o", yerr=err,
            label="in")
    gca().set_yscale("log")
    gca().set_xscale("log")
    gca().set_ylim(1e-1, 1e5)
    gca().set_xlim(0.8, 1e3)
    subplots_adjust(left=0.2, bottom=0.2)
    xlabel("$k_{in}$")
    ylabel("$NP(k_{in})$")
    tight_layout()
    #savefig("price-deg-dist.pdf")
    #savefig("price-deg-dist.svg")