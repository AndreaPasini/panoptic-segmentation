import pyximport

from sims.scene_graphs.image_processing import getImageName

pyximport.install(language_level=3)

import json
import os
import pandas as pd
from shutil import copyfile
import copy
from sims.prs import get_sup_ent_lists, filter_PRS_histograms, edge_pruning, node_pruning, load_PRS
from sims.gspan_mining.mining import prepare_gspan_graph_data, run_gspan_mining
from sims.subdue_mining.mining import prepare_subdue_graph_data, run_subdue_mining
from sims.visualization import print_graphs


def prepare_graphs_with_PRS(simsConf, overwrite_cache=False):
    """
    Given experiment configuration, return graphs, pruned and filtered according to KB
    Experiment may refer to either COCO or VG dataset.
    :param simsConf: experimental configuration class
    :param overwrite_cache: True to force overwrite cached data from previous executions
                             (e.g., use it when you change the configuration of PRS parameters)
    :return graphs (json format)
    """
    config = simsConf.SGS_params
    edge_pruning_str ="_eprune" if config['edge_pruning'] else ""
    node_pruning_str = "_nprune" if config['node_pruning'] else ""
    output_file = os.path.join(simsConf.SGS_dir, f"preparedGraphs{edge_pruning_str}{node_pruning_str}.json")
    # Check whether graph data has already been created
    if (not os.path.exists(output_file)) or overwrite_cache:
        # Read PRS
        prs = load_PRS(simsConf, False)
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


def build_SGS(simsConf, overwrite_PRS_cache=False):
    """
    Build the SGS (Scene Graph Summary) with frequent subgraph mining
    :param simsConf: experimental configuration class
    :param overwrite_PRS_cache: True to force overwrite of cached PRS data from previous executions
                               (e.g., use True when you change the configuration of the PRS.
                               Not necessary when changing the configuration of the SGS.)
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
    if not os.path.exists(input_graphs_data_path) or overwrite_PRS_cache:
        print(f"Preparing graphs for {config['alg']}...")
        train_graphs_filtered = prepare_graphs_with_PRS(simsConf, overwrite_PRS_cache)

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

def load_SGS(simsConf, min_nodes=2):
    """
    Read Json graphs in the SGS, generated with build_SGS()
    :param simsConf: experimental configuration class
    :param min_nodes: pick graphs with at least min_nodes
    :return: loaded json frequent graphs (SGS)
    """
    exp_name = simsConf.getSGS_experiment_name()

    # Read frequent graphs
    sgs_path = os.path.join(simsConf.SGS_dir, exp_name + '.json')
    with open(sgs_path, 'r') as f:
        sgs_graphs = json.load(f)

    # Pick graphs with more than min_nodes
    sgs_graphs = [g for g in sgs_graphs if len(g['g']['nodes']) >= min_nodes]

    return sgs_graphs


def load_and_print_SGS(simsConf, subsample=True, pdfformat=True, alternate_colors=True, clean_class_names=True):
    """
    Load SGS frequent graphs and print to files
    :param simsConf: experimental configuration class
    :param subsample: subsample graphs if >500
    :param pdfformat: True to print pdf, False to print png
    :param alternate_colors: True if you want to alternate different colors for nodes
    :param clean_class_names: True if you want to print cleaned COCO classes (e.g. remove "-merged")
    """
    graphs = load_SGS(simsConf)
    out_path = os.path.join(simsConf.SGS_dir, f"charts/{simsConf.getSGS_experiment_name()}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Print graphs
    print_graphs(graphs, out_path, subsample, pdfformat, alternate_colors, clean_class_names)


def SGS_to_represented_imgs(simsConf, method='img_avg'):
    """
    Pair to each SGS graph the identifier of a represented image in the input collection.
    :param simsConf: SImS_config SImS configuration class
    :param method: "img_min"/"img_max"/"img_avg" : given the represented images
                    of an SGS graph, it pairs the one that has the min/max/avg number of nodes.
    :return (list with the id of the represented image for each sgs graph, SGS graphs)
    """
    sgs = load_SGS(simsConf)
    # Get number of nodes for each input scene graph
    with open(simsConf.scene_graphs_json_path) as f:
        input_graphs = json.load(f)
    # Read coverage matrix
    coverage_mat = pd.read_csv(os.path.join(simsConf.SGS_dir,
                                "coverage_mat_" + simsConf.getSGS_experiment_name() + ".csv"),
                               index_col=None)

    # Get number of nodes for each input scene graph
    coverage_mat['nNodes'] = 0
    for i, g in enumerate(input_graphs):
        coverage_mat.loc[i,'nNodes'] = len(g['nodes'])

    # For each SGS graph
    paired_images = []
    print(f"Finding represented image for each SGS graph ({method})...")
    for g in sgs:
        gid = g['g']['graph']['name']
        if str(gid) in coverage_mat.columns:
            col = coverage_mat[[str(gid),'nNodes']]
            col = col[col.iloc[:,0]>0]

            if method=='img_min':
                selectedImgIndex = col['nNodes'].idxmin()
            elif method=='img_max':
                selectedImgIndex = col['nNodes'].idxmax()
            elif method=='img_avg':
                col['nNodes'] = (col['nNodes']-col['nNodes'].mean()).abs()
                selectedImgIndex = col['nNodes'].idxmin()
            else:
                print("Not a valid value for method parameter.")

            selectedImg = input_graphs[selectedImgIndex]['graph']['name']
            paired_images.append(selectedImg)
    return paired_images, sgs


def SGS_to_represented_img_graphs(simsConf, represented_imgs, sgs):
    """
    Pair to each SGS graph the graph of a represented image in the input collection.
    The graphs of the represented images are preprocessed with edge and node pruning.
    :param simsConf: SImS_config object with SImS configuration
    :param represented_imgs: output ids of SGS_to_represented_imgs()
    :param sgs: SGS graphs
    :return: a list with the same format of the SGS [{'sup': ,'g': }, {'sup':...]
            the supports are the original support values of SGS graphs
    """
    # Load associated graphs
    with open(simsConf.scene_graphs_json_path) as f:
        input_graphs = {g['graph']['name']: g for g in json.load(f)}
    paired_img_graphs = [copy.deepcopy(input_graphs[img]) for img in represented_imgs]

    prs = load_PRS(simsConf, filtering=True)
    paired_img_graphs = edge_pruning(prs, paired_img_graphs)
    paired_img_graphs = node_pruning(paired_img_graphs)

    res = []
    for img_g, sgs_g in zip(paired_img_graphs, sgs):
        img_g['graph']['name'] = f"{sgs_g['g']['graph']['name']}_{img_g['graph']['name']}"
        res.append({'sup':sgs_g['sup'],'g':img_g})
    return res


def load_and_print_SGS_images(simsConf, method='img_avg'):
    """
    Plot images associated to SGS graphs
    :param simsConf: experimental configuration class
    :param method: same options as specified for associate_img_to_sgs() in this file
    """
    out_path = os.path.join(simsConf.SGS_dir, f"charts/{simsConf.getSGS_experiment_name()}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Pair images:
    paired_images, sgs = SGS_to_represented_imgs(simsConf, method)
    # Copy jpeg files:
    for selectedImg, g in zip(paired_images, sgs):
        imgName = getImageName(selectedImg, extension='jpg')
        copyfile(os.path.join(simsConf.img_dir, imgName), os.path.join(out_path, f"s_{g['sup']}_g{g['g']['graph']['name']}_{selectedImg}.jpg"))
