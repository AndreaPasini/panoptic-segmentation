import pyximport

from sims.graph_algorithms import compute_diversity, compute_coverage_matrix

pyximport.install(language_level=3)
from panopticapi.utils import read_train_img_captions

from scipy.stats import entropy
import json
from config import COCO_PRS_json_path, COCO_train_graphs_json_path, COCO_train_graphs_subset_json_path, \
    COCO_train_graphs_subset2_json_path, COCO_SGS_dir
from sims.prs import filter_PRS_histograms, load_PRS, edge_pruning, node_pruning
from sims.sgs import load_SGS, SGS_to_represented_imgs, prepare_graphs_with_PRS, SGS_to_represented_img_graphs
import numpy as np
import os
import pandas as pd


def compute_coverage_mat_sims(config, method):
    """
    Use this function to evaluate SImS coverage.

    Compute the coverage matrix for a given summary (SGS) with respect to the input collection.
    Each row is associated to one of the collection graphs (e.g., g_i).
    Every element in the row counts the number of occurrences of each summary graph inside g_i.

    Important: with method="std" the result matrix only includes SGS graphs with >=2 nodes.

    :param config: experiment configuration (SImS_config class)
    :param method: "std" - standard evaluation: use frequent graphs in the SGS for subgraph isomorphism
                   "img_min"/"img_max"/"img_avg" - evaluation using image pairing.
                            in this case it pairs an image to each SGS graph with the method min/max/avg
                            then filters the scene graph with edge+node pruning
                            finally computes coverage and diversity on the filtered graph
    """
    experiment_name = config.getSGS_experiment_name()
    if method=='std':
        suffix = ""
    else:
        suffix = '_'+method
    output_file = os.path.join(config.SGS_dir, f"coverage_mat_{experiment_name}{suffix}.csv")

    # Read input collection
    tmp = config.SGS_params
    #config.SGS_params['edge_pruning']=True
    #config.SGS_params['node_pruning']=True # This speeds up process.
    input_graphs_filtered = prepare_graphs_with_PRS(config)
    config.SGS_params=tmp

    if method!='std':
        represented_imgs, sgs = SGS_to_represented_imgs(config, method)
        summary_graphs = SGS_to_represented_img_graphs(config, represented_imgs, sgs)
    else:
        summary_graphs = load_SGS(config, min_nodes=2) # Read frequent graphs, only those with >=2 nodes

    print("Computing coverage matrix...")
    cmatrix = compute_coverage_matrix(input_graphs_filtered, summary_graphs)
    cmatrix.to_csv(output_file, sep=",", index=False)

def evaluate_SGS(simsConf, topk=None, method='std'):
    """
    Evaluate SImS SGS.
    Compute statistics (avg. nodes, distinct classes, ...) on the extracted frequent subgraphs.
    :param simsConf: experimental configuration class
    :param topk: number of top-k frequent graphs to be considered for the evaluation
    :param method: "std" - standard evaluation: use frequent graphs in the SGS
                   "img_min"/"img_max"/"img_avg" - image evaluation.
                            in this case it pairs an image to each SGS graph with the method min/max/avg
                            then filters the scene graph with edge+node pruning
                            finally computes coverage and diversity on the filtered graph
    :return: dictionary with statistics
    """

    if method!='std':
        suffix = f"_{method}"
    else:
        suffix = ""

    # Read SGS summary. Only graphs with >=2 nodes
    sgs = load_SGS(simsConf, min_nodes=2)
    # Read coverage matrix
    coverage_mat = pd.read_csv(os.path.join(simsConf.SGS_dir,
                               f"coverage_mat_{simsConf.getSGS_experiment_name()}{suffix}.csv"),
                               index_col=None)

    # Read frequent graphs (consider only graphs with at least 2 nodes)
    if method!='std':
        represented_imgs = [(int(img.split('_')[1])) for img in coverage_mat.columns]
        summary_graphs = SGS_to_represented_img_graphs(simsConf, represented_imgs, sgs)
    else:
        summary_graphs = sgs


    res =  evaluate_summary_graphs(summary_graphs, coverage_mat, topk)
    config = simsConf.SGS_params
    if 'minsup' not in config:
        config['minsup'] = None
    res["Minsup"] = config['minsup']
    res["Edge pruning"] = 'Y' if config['edge_pruning'] else 'N'
    res["Node pruning"] = 'Y' if config['node_pruning'] else 'N'
    return res

def evaluate_SGS_df(simsConf, topk=None):
    """
    Evaluate SImS SGS. Return result into printable dataframe.
    Compute statistics (avg. nodes, distinct classes, ...) on the extracted frequent subgraphs.
    :param simsConf: experimental configuration class
    :param topk: number of top-k frequent graphs to be considered for the evaluation
    :return: printable dataframe with statistics.
    """
    res = evaluate_SGS(simsConf, topk)
    res_df = pd.DataFrame([res], columns=["Minsup", "Edge pruning", "Node pruning", "N. graphs",
                                            "Avg. nodes", "Std. nodes",
                                            "Coverage",
                                            "Diversity"])
    return res_df

def evaluate_summary_graphs(summary_graphs, coverage_mat, topk=None):
    """
    Evaluate a graph summary (SImS or Competitors)
    :param summary_graphs: list of summary graphs
    :param coverage_mat: coverage matrix of summary graphs to whole dataset
    :return: dictionary with statistics
    """

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
        if len(nodes)>0:
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
        ids = [str(g['g']['graph']['name']) for g in summary_graphs]
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


