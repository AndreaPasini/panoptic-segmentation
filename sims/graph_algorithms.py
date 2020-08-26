import pyximport
pyximport.install(language_level=3)

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from networkx.algorithms.isomorphism import categorical_node_match, categorical_edge_match, DiGraphMatcher
from sims.graph_utils import json_to_nx
from sims.sgs import prepare_graphs_with_PRS, read_freqgraphs

def subgraph_isomorphism(subgraph, graph, induced=False):
    """
    Check whether subgraph is a subgraph of graph (subgraph-isomorphism)
    :param subgraph: json_graph considered as sub-graph
    :param graph: json_graph considered as graph
    :return: list of dictionaries, one for each match. Each dictionary maps graphNodeId:subgraphNodeId
    """

    g_nodes = {n['label'] for n in graph['nodes']}
    s_nodes = {n['label'] for n in subgraph['nodes']}
    if not induced:
        g_nodes_map = {n['id'] : n['label'] for n in graph['nodes']}
        s_nodes_map = {n['id'] : n['label'] for n in subgraph['nodes']}

    if len(s_nodes - g_nodes) == 0:
        if not induced:
            # Copy the old graph (shallow copy)
            graph = copy.copy(graph)

            # NetworkX search for induced subgraphs.
            # These operations aim at reducing noise edges and simulates a non-induced isomorphism
            # 1. find useless connections
            # Connections of the subgraph
            good_edges = {(s_nodes_map[e['s']], e['pos'], s_nodes_map[e['r']]) for e in subgraph['links']}
            # Remove connections of graph if not contained in the subgraph
            pruned_edges = []
            for e in graph['links']:
                e_str = (g_nodes_map[e['s']], e['pos'], g_nodes_map[e['r']])
                if e_str in good_edges:
                    pruned_edges.append(e)
            graph['links'] = pruned_edges

            # 2. remove useless nodes (with label not included in subgraph)
            useless = g_nodes - s_nodes
            pruned_nodes = []
            for n in graph['nodes']:
                if n['label'] not in useless:
                    pruned_nodes.append(n)
            graph['nodes']=pruned_nodes

        nmatch = categorical_node_match('label','')
        ematch = categorical_edge_match('pos','')
        matcher = DiGraphMatcher(json_to_nx(graph), json_to_nx(subgraph),
                                 node_match=nmatch, edge_match=ematch)
        res_list = []
        i = 0
        # Necessary because sometimes crashes (too many matches)
        for res in matcher.subgraph_isomorphisms_iter():
            res_list.append(res)
            if i>=0:
                break
            i+=1

        return res_list
    else:
        return []


def get_isomorphism_count_vect(graph, freq_graphs):
    """
    Associate a json graph to the frequent graphs (sub-graph isomorphism).
    Obtain a count-vector with the number of found instances for each frequent graph
    :param graph: json graph with
    :param freq_graphs: frequent graphs (obtained with gspan or subdue)
    :return: the count-vector (Numpy) with the number of found instances for each frequent graph
    """
    cvector = np.zeros(len(freq_graphs))
    g_nodes = {n['label'] for n in graph['nodes']}
    for i, fgraph in enumerate(freq_graphs):
            # Sub-graph isomorphism
            fg_nodes = {n['label'] for n in fgraph['g']['nodes']}
            if len(fg_nodes - g_nodes) == 0:
                match_list = subgraph_isomorphism(fgraph['g'], graph)
                cvector[i] += len(match_list)
    return cvector


def compute_coverage_mat(config):
    """
    Given a graph mining experiment configuration, associate training COCO images to frequent graphs.
    Important: in the result considers only frequent graphs with >=2 nodes.
    Result is a count matrix saved to a csv file (1 row for each image, 1 column for each freq graph)
    :param config: experiment configuration (SImS_config class)
    """
    experiment_name = config.getSGS_experiment_name()
    output_file = os.path.join(config.SGS_dir, "coverage_mat_"+experiment_name+".csv")

    # Read training graphs
    train_graphs_filtered = prepare_graphs_with_PRS(config)
    # Read frequent graphs
    freq_graphs = read_freqgraphs(config)
    pbar = tqdm(total=len(train_graphs_filtered))
    cmatrix = []
    # Subgraph isomorphism to match frequent graphs with COCO train graphs
    g_ids = []
    freq_graphs2 = []
    for i, g in enumerate(freq_graphs):
        if len(g['g']['nodes']) >= 2:
            g_ids.append(i)
            freq_graphs2.append(g)
    for g in train_graphs_filtered:
            # Important: at least "2 nodes" to be considered.
            cmatrix.append(get_isomorphism_count_vect(g, freq_graphs2))
            pbar.update()
    pbar.close()
    cmatrix = pd.DataFrame(cmatrix, columns=g_ids)
    cmatrix.to_csv(output_file, sep=",", index=False)