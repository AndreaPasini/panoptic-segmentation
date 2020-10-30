import pyximport
pyximport.install(language_level=3)

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from networkx.algorithms.isomorphism import categorical_node_match, categorical_edge_match, DiGraphMatcher
from sims.graph_utils import json_to_nx


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


def get_isomorphism_count_vect(graph, sgs_graphs, min_nodes=2):
    """
    Return a count-vector with shape [len(sgs_graphs)].
    For each SGS graph it performs sub-graph isomorphism with respect to the input graph.
    Each element in the vector counts the number of occurrences of an SGS graph inside the input graph.
    :param graph: json graph
    :param sgs_graphs: list of graphs in the SGS
    :param min_nodes: minimum number of nodes for a graph to be considered.
                If the condition is not satisfied, the SGS graph has 0 coverage for all the
                collection graphs (e.g. column with all 0).
    :return: the count-vector (Numpy) with the number of found instances for each SGS graph
    """
    cvector = np.zeros(len(sgs_graphs))
    g_nodes = {n['label'] for n in graph['nodes']}
    for i, fgraph in enumerate(sgs_graphs):
        if len(fgraph['g']['nodes']) >= min_nodes: # No coverage if min nodes not satisfied
            # Sub-graph isomorphism
            fg_nodes = {n['label'] for n in fgraph['g']['nodes']}
            if len(fg_nodes - g_nodes) == 0:
                match_list = subgraph_isomorphism(fgraph['g'], graph)
                cvector[i] += len(match_list)
    return cvector

def compute_coverage_matrix(collection_graphs, sgs_graphs, min_nodes=2):
    """
    Compute the coverage matrix for a given summary (SGS) with respect to the input collection.
    Each row is associated to one of the collection graphs (e.g., g_i).
    Every element in the row counts the number of occurrences of each SGS graph inside g_i.

    :param collection_graphs: list of graphs in the initial collection
    :param sgs_graphs: list of summary graphs
    :param min_nodes: minimum number of nodes for a graph to be considered.
                    Default 2 (i.e., empty graphs and single-node graphs are not considered
                    since they are too generic).
                    If the condition is not satisfied, the SGS graph has 0 coverage for all the
                    collection graphs (e.g. column with all 0).
    :return: coverage matrix (pandas DataFrame)
    """
    g_ids = [g['g']['graph']['name'] for g in sgs_graphs]
    cmatrix = []
    pbar = tqdm(total=len(collection_graphs))

    for g in collection_graphs:
        cmatrix.append(get_isomorphism_count_vect(g, sgs_graphs, min_nodes))
        pbar.update()
    pbar.close()
    cmatrix = pd.DataFrame(cmatrix, columns=g_ids)
    return cmatrix


def compute_diversity(sgs_graphs):
    """
    Given json graphs, compute diversity score.
    :param sgs_graphs: list of the SGS json graphs
    :return: diversity score (float)
    """

    if len(sgs_graphs)==1:
        if len(sgs_graphs[0]['g']['nodes'])>0:
            return 1   # 1 graph, with at least 1 node. Diversity=1
        else:
            return 0   # 1 empty graph. Diversity=0

    node_sets = []
    for g in sgs_graphs:
        node_sets.append({n['label'] for n in g['g']['nodes']})

    distances = 0
    n = 0
    for i in range(len(node_sets)):
        for j in range(i+1, len(node_sets)):
            ni = node_sets[i]
            nj = node_sets[j]
            n+=1 # Count this pair

            # Distance between a void node and a non-void node is 0
            if len(ni)==0 or len(nj)==0:
                continue

            intersect = ni & nj
            union = ni | nj
            distances += 1-len(intersect)/len(union)

    return distances/n