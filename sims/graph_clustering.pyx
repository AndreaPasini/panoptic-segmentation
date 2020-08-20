import copy
import json
import os

import numpy as np
import pandas as pd
from networkx.algorithms.isomorphism import categorical_node_match, categorical_edge_match, DiGraphMatcher
from tqdm import tqdm

from config import trainimage_freqgraph_csv_path, freqgraph_place_csv_path, trainimage_place_csv_path
from sims.conceptnet.places import Conceptnet
from sims.graph_mining import prepare_graphs_with_PRS, get_exp_name, read_freqgraphs
from sims.graph_utils import json_to_nx

import graph_tools as gt
from graph_tool.all import subgraph_isomorphism as gt_subgraph_isomorphism


# def subgraph_isomorphism(subgraph, graph):
#     """
#     Check whether subgraph is a subgraph of graph (subgraph-isomorphism)
#     :param subgraph: json_graph considered as sub-graph
#     :param graph: json_graph considered as graph
#     :return: list of dictionaries, one for each match. Each dictinary maps graphNodeId:subgraphNodeId
#     """
#     nmatch = categorical_node_match('label','')
#     ematch = categorical_edge_match('pos','')
#     matcher = DiGraphMatcher(json_to_nx(graph), json_to_nx(subgraph),
#                              node_match=nmatch, edge_match=ematch)
#     return list(matcher.subgraph_isomorphisms_iter())

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
        return list(matcher.subgraph_isomorphisms_iter())
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

def compute_image_freqgraph_count_mat(experiment, output_file):
    """
    Given a graph mining experiment configuration, associate training COCO images to frequent graphs.
    Result is a count matrix saved to a csv file (1 row for each image, 1 column for each freq graph)
    :param experiment: experiment configuration (dictionary)
    """
    # Read training graphs
    train_graphs_filtered = prepare_graphs_with_PRS(experiment)
    # Read frequent graphs
    freq_graphs = read_freqgraphs(experiment)
    pbar = tqdm(total=len(train_graphs_filtered))
    cmatrix = []
    # Subgraph isomorphism to match frequent graphs with COCO train graphs
    for g in train_graphs_filtered:
        cmatrix.append(get_isomorphism_count_vect(g, freq_graphs))
        pbar.update()
    pbar.close()
    cmatrix = pd.DataFrame(cmatrix)
    cmatrix.to_csv(output_file, sep=",", index=False)

def compute_freqgraph_place_count_mat(experiment):
    """
    Given a graph mining experiment configuration, associate frequent graphs to places.
    Result is a count matrix saved to a csv file (1 row for each freq graph, 1 column for each conceptnet place)
    :param experiment: experiment configuration (dictionary)
    """
    # Read frequent graphs
    freq_graphs = read_freqgraphs(experiment)
    # Read conceptnet
    conceptnet = Conceptnet()
    cmatrix = []
    places_map = {place:i for i,place in enumerate(conceptnet.places)}
    # For each frequent graph
    for fgraph in freq_graphs:
        cvector = np.zeros(len(conceptnet.places))
        # Get places related to this frequent graph
        rank = conceptnet.rank_related_places(fgraph['g'])
        for place, w in rank:
            cvector[places_map[place]]=w
        cmatrix.append(cvector)
    cmatrix= pd.DataFrame(cmatrix, columns=conceptnet.places)
    cmatrix.to_csv(freqgraph_place_csv_path, sep=",", index=False)

def compute_image_place_count_mat():
    """
    Associate COCO training images to places.
    Requires running compute_image_freqgraph_count_mat() and compute_freqgraph_place_count_mat() first
    Result is a count matrix saved to a csv file (1 row for each image, 1 column for each conceptnet place)
    """
    # Read count matrices
    freqg_place = pd.read_csv(freqgraph_place_csv_path)
    img_freqg = pd.read_csv(trainimage_freqgraph_csv_path)
    cmatrix = []
    pbar = tqdm(total=len(img_freqg))
    for img_row in img_freqg.iterrows():
        cvector = np.zeros(freqg_place.shape[1])
        for i, fgraph in enumerate(img_row[1]):
            if fgraph>0:
                cvector += freqg_place.iloc[i].to_numpy()
        cmatrix.append(cvector)
        pbar.update()
    pbar.close()
    cmatrix = pd.DataFrame(cmatrix, columns=freqg_place.columns)
    cmatrix.to_csv(trainimage_place_csv_path, sep=",", index=False)