import copy
import os
from datetime import datetime
import pyximport
pyximport.install(language_level=3)
from networkx.algorithms.isomorphism import categorical_node_match, DiGraphMatcher, categorical_edge_match
from tqdm import tqdm

from config import graph_clustering_dir, trainimage_freqgraph_csv_path
from sims.conceptnet.places import Conceptnet
from sims.graph_clustering import compute_image_freqgraph_count_mat
from sims.graph_utils import json_to_nx, print_graph_picture



from sims.graph_mining import read_freqgraphs, print_graphs, load_and_print_fgraphs, prepare_graphs_with_PRS
from sims.sims_config import SImS_config
import pandas as pd
import numpy as np
import graph_tool as gt
from graph_tool.all import subgraph_isomorphism as gt_subgraph_isomorphism, graph_draw

# def jsongraph_to_graphtool(json_graph):
#     gtgraph = gt.Graph()
#
#     vlabels = gtgraph.new_vertex_property("string")
#
#     for n in json_graph['nodes']:
#         nlabel = n['label']
#         v = gtgraph.add_vertex(n['id'])
#         vlabels[n['id']] = nlabel
#
#     edge_list = [(l['s'], l['r'], l['pos']) for l in json_graph['links']]
#     elabels = gtgraph.new_edge_property("string")
#     gtgraph.add_edge_list(edge_list, eprops=[elabels])
#     return gtgraph, vlabels, elabels

# def subgraph_isomorphism2(subgraph, graph):
#     """
#     Check whether subgraph is a subgraph of graph (subgraph-isomorphism)
#     :param subgraph: json_graph considered as sub-graph
#     :param graph: json_graph considered as graph
#     :return: list of arrays, one for each match. Each array maps subgraphNodeId:graphNodeId
#     """
#
#
#     # gt_subgraph, vlabels_s, elabels_s = jsongraph_to_graphtool(subgraph)
#     # gt_graph, vlabels_g, elabels_g = jsongraph_to_graphtool(graph)
#     # if len(subgraph['nodes'])>1:
#     #     res = gt_subgraph_isomorphism(gt_subgraph, gt_graph, edge_label=(elabels_s, elabels_g),
#     #                                                 vertex_label=(vlabels_s, vlabels_g), max_n=0, induced=False)
#     #     if len(res)>0:
#     #         return [r.a for r in res]
#     # else:
#     #     # If the subgraph has only 1 node, find if the graph has the same node label
#     #     node_label = subgraph['nodes'][0]['label']
#     #     for g in graph['nodes']:
#     #         if node_label==g['label']:
#     #             return [{subgraph['nodes'][0]['id']: g['id']}]
#
#     return []











def get_maximal_itemsets(freq_graphs):
    """

    :param freq_graphs:
    :return:
    """

    # Extract unique itemsets from frequent graphs
    itemset2graph = {}  # key=itemset, value=list of graphs with this itemset
    itemset_list = []   # list of itemsets
    sup_list = []       # list of supports associated to itemset_list
    for g in freq_graphs:
        itemset = {n['label'] for n in g['g']['nodes']}
        itemset_t = tuple(sorted({n['label'] for n in g['g']['nodes']}))   # All node classes (itemset)
        sup = g['sup']

        # The same itemset may appear multiple times (same nodes, different edges)
        if (itemset_t not in itemset2graph.keys()):
            # Add new itemset
            itemset_list.append(itemset)
            sup_list.append(sup)
            itemset2graph[itemset_t] = [g]
        else:
            itemset2graph[itemset_t].append(g)
    # Create Series, index=sorted ascending support, value=itemset
    itemset_series = pd.Series(itemset_list, index=sup_list)
    itemset_series.sort_index(inplace=True)

    maximal_itemsets = []
    # Find maximal itemsets (no superset that is frequent)
    for sup, itemset in itemset_series.items():
        # Find a superset among itemsets with equal or lower freq (apriori principle)
        candidates = itemset_series[itemset_series.index<=sup].values
        maximal = True
        for c in candidates:
            if len(itemset)<len(c) and itemset.issubset(c):
                maximal = False
                break
        if maximal:
            maximal_itemsets.append(tuple(sorted(itemset)))

    maximal_graphs = []
    maximal_itemsets = set(maximal_itemsets)
    # Add to result only maximal itemsets
    for itemset, graphs in itemset2graph.items():
        if itemset in maximal_itemsets:
            if len(graphs)>1: # between more graphs choose the one with highest support
                max_graph = None
                for g in graphs:
                    if max_graph is None or g['sup']>max_graph['sup']:
                        max_graph=g
                maximal_graphs.append(max_graph)
            else:
                maximal_graphs.append(graphs[0])
    return maximal_graphs



if __name__ == "__main__":
    ### Choose methods to be run ###
    class RUN_CONFIG:
        print_maximal = False # Print maximal-itemset graphs
        associate_places = False
        compute_image_freqgraph_count_mat = True  # Associate training COCO images to frequent graphs (7 minutes)
        compute_freqgraph_place_count_mat = False  # Associate frequent graphs to places
        compute_image_place_count_mat = False  # Associate training COCO images to places
        associate_to_freq_graphs = False

    start_time = datetime.now()
    # Load configuration for COCO and default frequent graphs
    config = SImS_config('COCO')

    if RUN_CONFIG.print_maximal:
        # Read frequent graphs, given configuration
        fgraphs = read_freqgraphs(config)
        maximal_fgraphs = get_maximal_itemsets(fgraphs)
        out_path = os.path.join(config.SGS_dir,'charts/maximal')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # Print maximal-itemset graphs
        print_graphs(maximal_fgraphs, out_path)

    if RUN_CONFIG.associate_places:
        # Read frequent graphs, given configuration
        fgraphs = read_freqgraphs(config)
        maximal_fgraphs = get_maximal_itemsets(fgraphs)
        conceptnet = Conceptnet()

        for i, g in enumerate(maximal_fgraphs):
            rank = conceptnet.rank_related_places(g['g'], antonyms=False)
            if len(rank) > 1:
                print_graph_picture(f"{config.SGS_dir}/charts/places/{rank[0][0]}_{rank[1][0]}_{i}.png", json_to_nx(g['g']))
            elif len(rank) > 0:
                print_graph_picture(f"{config.SGS_dir}/charts/places/{rank[0][0]}_{i}.png",
                                        json_to_nx(g['g']))
            else:
                print_graph_picture(f"{config.SGS_dir}/charts/places/g_{i}.png",
                                    json_to_nx(g['g']))

    if RUN_CONFIG.compute_image_freqgraph_count_mat:
        # Check subgraph isomorphism of each frequent graph with COCO training images
        start_time = datetime.now()
        compute_image_freqgraph_count_mat(config, trainimage_freqgraph_csv_path)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))


    # if RUN_CONFIG.associate_to_freq_graphs:
    #     with open(os.path.join(COCO_SGS_dir, 'train_freqGraph_kbfilter_prune_gspan_005.json')) as f:
    #         freq_graphs = json.load(f)
    #     conceptnet = Conceptnet()
    #
    #     def associate_graph(g, i):
    #         rank = conceptnet.rank_related_places(g['g'])
    #         if len(rank) > 0:
    #             print_graph_picture(f"{COCO_SGS_dir}/clusters/{rank[0][0]}_{i}.png", json_to_nx(g['g']))
    #
    #     for i, g in enumerate(freq_graphs):
    #         associate_graph(g, i)
    #

    # if RUN_CONFIG.compute_freqgraph_place_count_mat:
    #     print(f"Selected experiment: {experiments[selected_experiment]}")
    #     start_time = datetime.now()
    #     compute_freqgraph_place_count_mat(experiment)
    #     end_time = datetime.now()
    #     print('Duration: ' + str(end_time - start_time))
    #
    # if RUN_CONFIG.compute_image_place_count_mat:
    #     print(f"Selected experiment: {experiments[selected_experiment]}")
    #     start_time = datetime.now()
    #     compute_image_place_count_mat()
    #     end_time = datetime.now()
    #     print('Duration: ' + str(end_time - start_time))
    # print('Done')