import pyximport

from sims.graph_algorithms import compute_diversity

pyximport.install(language_level=3)
from panopticapi.utils import read_train_img_captions

from scipy.stats import entropy
import json
from config import COCO_PRS_json_path, COCO_train_graphs_json_path, COCO_train_graphs_subset_json_path, \
    COCO_train_graphs_subset2_json_path, COCO_SGS_dir
from sims.prs import filter_PRS_histograms
from sims.sgs import load_sgs
import numpy as np
import os
import pandas as pd

def evaluate_SGS(simsConf, topk=None):
    """
    Evaluate SImS SGS.
    Compute statistics (avg. nodes, distinct classes, ...) on the extracted frequent subgraphs.
    :param simsConf: experimental configuration class
    :param topk: number of top-k frequent graphs to be considered for the evaluation
    :return: dictionary with statistics
    """
    # Read frequent graphs
    sgs_graphs = load_sgs(simsConf)
    # Read coverage matrix
    coverage_mat = pd.read_csv(os.path.join(simsConf.SGS_dir,
                               "coverage_mat_" + simsConf.getSGS_experiment_name() + ".csv"),
                               index_col=None)
    res =  evaluate_summary_graphs(sgs_graphs, coverage_mat, topk)
    config = simsConf.SGS_params
    if 'minsup' not in config:
        config['minsup'] = None
    res["Minsup"] = config['minsup']
    res["Edge pruning"] = 'Y' if config['edge_pruning'] else 'N'
    res["Node pruning"] = 'Y' if config['node_pruning'] else 'N'
    return res


def evaluate_summary_graphs(summary_graphs, coverage_mat, topk=None):
    """
    Evaluate a graph summary (SImS or Competitors)
    :param summary_graphs: list of summary graphs
    :param coverage_mat: coverage matrix of summary graphs to whole dataset
    :return: dictionary with statistics
    """

    for i, g in enumerate(summary_graphs):
        g['id'] = i

    # Pick graphs with>2 nodes
    summary_graphs = [g for g in summary_graphs if len(g['g']['nodes']) >= 2]

    # Pick top-k graphs
    if topk is not None:
        # Sort graphs by support
        summary_graphs = sorted(summary_graphs, key=lambda g: -g['sup'])[:topk]

    dist_classes = {}
    dist_sets = {}
    tot_nodes = 0       # Number of nodes
    tot_dist_nodes = 0  # Number of distinct classes in each graph
    max_dist_nodes = 0  # Max n. distinct classes
    nodes_nodes_dist = 0    # Ratio distinct nodes / nodes
    std_nodes = []
    for g in summary_graphs:
        nodes = [n['label'] for n in g['g']['nodes']]   # All node classes
        tot_nodes += len(nodes)                         # Number of nodes
        std_nodes.append(len(nodes))
        nodes_set = set(nodes)
        tot_dist_nodes += len(nodes_set)                # Number of distinct classes
        max_dist_nodes = max(max_dist_nodes, len(nodes_set)) # Max n. distinct classes
        nodes_nodes_dist += len(nodes_set)/len(nodes)
        # Add
        for n in nodes_set:                             # Track distinct classes
            if n in dist_classes:
                dist_classes[n] += 1
            else:
                dist_classes[n] = 1
        nodes_tuple = tuple(sorted(nodes_set))          # Track distinct class sets
        if nodes_tuple in dist_sets:
            dist_sets[nodes_tuple] += 1
        else:
            dist_sets[nodes_tuple] = 1


    # Compute coverage
    if topk:    # Select only topk graphs
        ids = [str(g['id']) for g in summary_graphs]
        coverage_mat = coverage_mat[ids]
    covered = (coverage_mat.sum(axis=1) > 0).sum()
    coverage = covered / len(coverage_mat)

    # Compute diversity
    diversity = compute_diversity(summary_graphs)

    res_dict = {"N. graphs": len(summary_graphs),
                "N. distinct class sets": len(dist_sets),
                "Distinct Set Ratio": round(len(dist_sets)/len(summary_graphs), 3),
                "Avg. nodes": round(tot_nodes / len(summary_graphs), 2),
                "Avg. distinct classes": round(tot_dist_nodes / len(summary_graphs),2), "Max. distinct classes": max_dist_nodes,
                "Distinct Node Ratio":round(nodes_nodes_dist/len(summary_graphs),2),
                "Std. nodes": round(np.std(std_nodes),2),
                "Coverage" : round(coverage,2),
                "Diversity" : round(diversity, 2)}
    return res_dict


def create_COCO_images_subset():
    """
    Create a subset of COCO images, designed for comparing SImS against competitors
    Select images that contain either of: {'river'}, {'surfboard'}, {'clock'}, {'book'}
    """
    if os.path.exists(COCO_train_graphs_subset_json_path):
        return

    # Remove old data (avoids conflicts with old instances of this dataset)
    for file in os.listdir(COCO_SGS_dir + "/subset/"):
        if file.startswith('preparedGraphs') or file.startswith("sgs_"):
            os.remove(COCO_SGS_dir + "/subset/" + file)

    print("Selecting COCO subset for article...")
    # Read COCO Train graphs
    with open(COCO_train_graphs_json_path, 'r') as f:
        train_graphs = json.load(f)

    categories = [{'river'}, {'surfboard'}, {'clock'}, {'book'}]
    graphs = [[] for el in categories]
    for g in train_graphs:
        labels = {node['label'] for node in g['nodes']}
        for i, cat in enumerate(categories):
            if cat.issubset(labels):
                graphs[i].append(g)
                break

    all_graphs = []
    for el in graphs:
        all_graphs.extend(el)
    with open(COCO_train_graphs_subset_json_path, 'w') as f:
        json.dump(all_graphs, f)
    print("Done.")

def select_COCO_images_by_caption(img_captions, search_query):
    """
    Given image captions retrieve only those
    that match the given textual query
    :param img_captions: image captions (use read_train_img_captions)
    :param search_query: list of strings that must be contained in the selected images
    :return: same structure as img_captions, filtered with query
    """
    res = {}
    for img, captions in img_captions.items():
        match = []
        for c in captions:
            for q in search_query:
                if q in c:
                    match.append(c)
        if len(match)>0:
            res[img]=match
    return res

def create_COCO_images_subset2():
    """
    Create a subset of COCO images, designed for comparing SImS against competitors
    Select images whose caption contains either "skiing" or "driving".
    """
    if os.path.exists(COCO_train_graphs_subset2_json_path):
        return

    # Remove old data (avoids conflicts with old instances of this dataset)
    for file in os.listdir(COCO_SGS_dir + "/subset2/"):
        if file.startswith('preparedGraphs') or file.startswith("sgs_"):
            os.remove(COCO_SGS_dir + "/subset2/" + file)

    print("Selecting COCO subset for article...")
    img_captions = read_train_img_captions('train')
    sel_img_captions = select_COCO_images_by_caption(img_captions, ['skiing', 'driving'])

    # Read COCO Train graphs
    with open(COCO_train_graphs_json_path, 'r') as f:
        train_graphs = json.load(f)

    all_graphs = []
    for g in train_graphs:
        if g['graph']['name'] in sel_img_captions:
            all_graphs.append(g)

    with open(COCO_train_graphs_subset2_json_path, 'w') as f:
        json.dump(all_graphs, f)
    print("Done.")
