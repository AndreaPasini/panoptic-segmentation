import pyximport
pyximport.install(language_level=3)

import json
import os
from shutil import copyfile
from sims.prs import get_sup_ent_lists, filter_PRS_histograms, edge_pruning, node_pruning
from sims.gspan_mining.mining import prepare_gspan_graph_data, run_gspan_mining
from sims.subdue_mining.mining import prepare_subdue_graph_data, run_subdue_mining
from sims.visualization import print_graphs


def prepare_graphs_with_PRS(simsConf):
    """
    Given experiment configuration, return graphs, pruned and filtered according to KB
    Experiment may refer to either COCO or VG dataset.
    :param simsConf: experimental configuration class
    :return graphs (json format)
    """
    config = simsConf.SGS_params
    edge_pruning_str ="_eprune" if config['edge_pruning'] else ""
    node_pruning_str = "_nprune" if config['node_pruning'] else ""
    output_file = os.path.join(simsConf.SGS_dir, f"preparedGraphs{edge_pruning_str}{node_pruning_str}.json")
    # Check whether graph data has already been created
    if not os.path.exists(output_file):
        # Read PRS
        with open(simsConf.PRS_json_path, 'r') as f:
            prs = json.load(f)
        # Get support and entropy
        sup, entr = get_sup_ent_lists(prs)
        # Get minsup and maxentr
        min_sup, max_entropy = simsConf.get_PRS_filters(sup, entr)

        # Filter PRS
        if config['edge_pruning']:
            prs_filtered = filter_PRS_histograms(prs, min_sup, max_entropy)
        else:
            prs_filtered = filter_PRS_histograms(prs, min_sup, 100)  # No filter for entropy
        # Read COCO Train graphs
        with open(simsConf.scene_graphs_json_path, 'r') as f:
            input_scene_graphs = json.load(f)
        # Edge pruning: filter graphs with KB
        print("Filtering graphs with PRS...")
        input_scene_graphs_filtered = edge_pruning(prs_filtered, input_scene_graphs)
        # Node pruning (prune equivalent nodes to reduce redundancies and to reduce mining time)
        if config['node_pruning']:
            print("Pruning nodes...")
            input_scene_graphs_filtered = node_pruning(input_scene_graphs_filtered)
        with open(output_file, "w") as f:
            json.dump(input_scene_graphs_filtered, f)
        return input_scene_graphs_filtered
    else:
        with open(output_file, "r") as f:
            return json.load(f)


def build_SGS(simsConf):
    """
    Run graph mining experiment
    :param simsConf: experimental configuration class
    """
    config = simsConf.SGS_params
    edge_pruning ="_eprune" if config['edge_pruning'] else ""
    node_pruning = "_nprune" if config['node_pruning'] else ""

    # Load dataset categories
    obj_categories, rel_categories = simsConf.load_categories()

    input_graphs_data_path = os.path.join(simsConf.SGS_dir, f"preparedGraphs{edge_pruning}{node_pruning}_{config['alg']}.data")
    exp_name = simsConf.getSGS_experiment_name()
    sgs_graphs_path = os.path.join(simsConf.SGS_dir, exp_name + '.json')

    if not os.path.exists(simsConf.SGS_dir):
        os.makedirs(simsConf.SGS_dir)

    # Check whether graph data has already been converted for
    if not os.path.exists(input_graphs_data_path):
        print(f"Preparing graphs for {config['alg']}...")
        train_graphs_filtered = prepare_graphs_with_PRS(simsConf)

        if config['alg']=='gspan':
            # Convert json graphs to the correct format for gspan mining.
            prepare_gspan_graph_data(input_graphs_data_path, train_graphs_filtered, obj_categories, rel_categories)
        elif config['alg']=='subdue':
            prepare_subdue_graph_data(input_graphs_data_path, train_graphs_filtered, obj_categories, rel_categories)

    # Mining of frequent graphs
    if config['alg'] == 'gspan':
        # Necessary because gspan program outputs with the same name of the input file
        tmp_input = os.path.join(simsConf.SGS_dir, exp_name + ".data")
        copyfile(input_graphs_data_path, tmp_input)
        run_gspan_mining(tmp_input, config['minsup'], sgs_graphs_path, obj_categories, rel_categories)
        os.remove(tmp_input)
    elif config['alg'] == 'subdue':
        run_subdue_mining(input_graphs_data_path, config['nsubs'], sgs_graphs_path, obj_categories, rel_categories)

def read_freqgraphs(simsConf):
    """
    Read Json frequent graphs generated with run_graph_mining()
    :param simsConf: experimental configuration class
    :return: loaded json frequent graphs
    """
    exp_name = simsConf.getSGS_experiment_name()

    # Read frequent graphs
    freq_graphs_path = os.path.join(simsConf.SGS_dir, exp_name + '.json')
    with open(freq_graphs_path, 'r') as f:
        freq_graphs = json.load(f)
    return freq_graphs


def load_and_print_SGS(simsConf, subsample=True, pdfformat=True, alternate_colors=True, clean_class_names=True):
    """
    Load SGS frequent graphs and print to files
    :param simsConf: experimental configuration class
    :param subsample: subsample graphs if >500
    :param pdfformat: True to print pdf, False to print png
    :param alternate_colors: True if you want to alternate different colors for nodes
    :param clean_class_names: True if you want to print cleaned COCO classes (e.g. remove "-merged")
    """
    graphs = read_freqgraphs(simsConf)
    out_path = os.path.join(simsConf.SGS_dir, f"charts/{simsConf.getSGS_experiment_name()}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Print graphs
    print_graphs(graphs, out_path, subsample, pdfformat, alternate_colors, clean_class_names)
