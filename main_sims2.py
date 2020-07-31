import os
from datetime import datetime
import pyximport
pyximport.install(language_level=3)
from sims.graph_mining import read_freqgraphs, print_graphs
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
    for itemset, graphs in itemset2graph.items():
        if itemset in maximal_itemsets:
            maximal_graphs.extend(graphs)
    return maximal_graphs


if __name__ == "__main__":
    start_time = datetime.now()
    # Load configuration for COCO and default frequent graphs
    config = SImS_config('COCO')
    # Read frequent graphs, given configuration
    fgraphs = read_freqgraphs(config)
    maximal = get_maximal_itemsets(fgraphs)
    out_path = os.path.join(config.SGS_dir,'maximal')
    os.makedirs(out_path)
    print_graphs(maximal, out_path)
    print('Done')