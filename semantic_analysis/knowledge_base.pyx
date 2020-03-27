"""
This file contains functions related to the knowledge base computation and analysis:
- create_kb_histograms() get image training graphs and compute KB (triplets and histograms)
"""

import json
from multiprocessing.pool import Pool
import pyximport
from scipy.stats import entropy
from tqdm import tqdm
pyximport.install(language_level=3)


def compute_hist_from_graph(graph):
    """
    Compute histogram from a graph (=image)
    :return: histogram
    """
    # Create map with node labels
    node_labels = {n['id'] : n['label'] for n in graph['nodes']}
    hist = {}
    for l in graph['links']:
        s = node_labels[l['s']]
        r = node_labels[l['r']]
        pos = l['pos']
        if (s,r) in hist:
            h_sr = hist[s, r]
            if pos in h_sr:
                h_sr[pos]+=1
            else:
                h_sr[pos]=1
        else:
            hist[s, r] = {}
            hist[s, r][pos]=1
    return hist

def create_kb_histograms(graphs_json_path, out_json_path):
    """
    Analyze training graphs generated by create_kb_graphs()
    Generate knowledge base histograms
    :param graphs_json_path: json file with graphs
    """
    with open(graphs_json_path, "r") as f:
        graphs = json.load(f)

    # Init progress bar
    pbar = tqdm(total=len(graphs))

    def update(x):
        pbar.update()

    print("Number of graphs: %d" % len(graphs))
    print("Scheduling tasks...")
    pool = Pool(10)
    results = []

    # Analyze all graphs
    for graph in graphs:
        results.append(pool.apply_async(compute_hist_from_graph, args=(graph,), callback=update))
    pool.close()
    pool.join()
    pbar.close()

    print("Collecting results...")
    # Collect histogram results
    histograms = {}
    for res_getter in results:
        h_pairs = res_getter.get()
        if h_pairs is not None:
            for pair, hist in h_pairs.items():
                if pair not in histograms:
                    # add histogram as it is if pair is not existing
                    histograms[pair] = hist
                else:
                    total_hist = histograms[pair]
                    # update histograms if pair already existing
                    for key in hist:
                        if key in total_hist:
                            total_hist[key] += hist[key]
                        else:
                            total_hist[key] = hist[key]

    n_pairs = 0
    # Compute histogram supports and entropy
    for hist in histograms.values():
        sup = sum(hist.values())  # support: sum of all occurrences in the histogram
        ent = []
        for pos, count in hist.items():
            perc = count / sup
            hist[pos] = perc
            ent.append(perc)
        hist['sup'] = sup
        n_pairs += sup
        # Important: the dictionary inside hist may not contain all the different position labels
        # E.g. {'side':0.5, 'side-up':0.5}
        # There are missing zeros like ... 'above':0, 'on':0,...
        # However these 0 terms does not influence entropy (because 0*log(0)=0)
        hist['entropy'] = entropy(ent, base=2)

    with open(out_json_path, "w") as f:
        json.dump({str(k): v for k, v in histograms.items()}, f)

    print(f"Number of analyzed object pairs: {n_pairs}")
    print(f"Number of generated histograms: {len(histograms)}")
