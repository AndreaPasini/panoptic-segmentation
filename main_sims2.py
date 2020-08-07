import os
from datetime import datetime
import pyximport

from sims.conceptnet.places import Conceptnet
from sims.graph_utils import json_to_nx, print_graph_picture

pyximport.install(language_level=3)
from sims.graph_mining import read_freqgraphs, print_graphs, load_and_print_fgraphs
from sims.sims_config import SImS_config
import pandas as pd



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
        compute_image_freqgraph_count_mat = False  # Associate training COCO images to frequent graphs (7 minutes)
        compute_freqgraph_place_count_mat = False  # Associate frequent graphs to places
        compute_image_place_count_mat = False  # Associate training COCO images to places
        associate_to_freq_graphs = False

    start_time = datetime.now()
    # Load configuration for COCO and default frequent graphs
    config = SImS_config('COCO')
    # Read frequent graphs, given configuration
    fgraphs = read_freqgraphs(config)
    maximal_fgraphs = get_maximal_itemsets(fgraphs)

    if RUN_CONFIG.print_maximal:
        out_path = os.path.join(config.SGS_dir,'charts/maximal')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # Print maximal-itemset graphs
        print_graphs(maximal_fgraphs, out_path)

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
    # if RUN_CONFIG.compute_image_freqgraph_count_mat:
    #     print(f"Selected experiment: {experiments[selected_experiment]}")
    #     start_time = datetime.now()
    #     compute_image_freqgraph_count_mat(experiment)
    #     end_time = datetime.now()
    #     print('Duration: ' + str(end_time - start_time))
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