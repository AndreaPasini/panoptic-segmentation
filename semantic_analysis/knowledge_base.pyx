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
        pair = f"{s},{r}"
        pos = l['pos']
        #if (s,r) in hist:
        if pair in hist:
            #h_sr = hist[s, r]
            h_sr = hist[pair]
            if pos in h_sr:
                h_sr[pos]+=1
            else:
                h_sr[pos]=1
        else:
            #hist[s, r] = {}
            #hist[s, r][pos]=1
            hist[pair] = {}
            hist[pair][pos]=1
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

def filter_kb_histograms(kb, min_sup, max_entropy):
    """
    :param kb: knowledge base (json format)
    :param min_sup: minimum support for filtering histograms
    :param max_entropy: maximum entropy for filtering histograms
    :return: filtered knowledge base, based on min_sup and max_entropy
    """
    return {pair : h for pair, h in kb.items() if h['sup']>=min_sup and h['entropy']<=max_entropy}

def get_likelihood(nodes_map, link, kb):
    """
    Retrieve from the Knowledge Base the likelihood of this link
    :param nodes_map: map NodeId:ClassLabel(COCO)
    :param link: graph link
    :param kb: knowledge base (read from json)
    :return: likelihood l of link and histogram; None if there is no histogram
    """
    sub = nodes_map[link['s']]
    ref = nodes_map[link['r']]
    pos = link['pos']
    pair = f"{sub},{ref}"

    # Check for the likelihood in the KB
    if pair in kb:
        hist = kb[f"{sub},{ref}"]
        if pos in hist:
            return hist[pos], hist
        else:
            return 0, hist
    return None, None


def get_sup_ent_lists(kb):
    """
    Get support and entropy values in two lists, from the knowledge base (json format)
    :param histograms:
    """
    sup = [h['sup'] for h in kb.values()]
    ent = [h['entropy'] for h in kb.values()]
    return sup, ent


def filter_graph_edges(kb, graphs):
    """
    Prune graph edges, when they are not present in the knowledge base
    :return: filtered graphs
    """
    stat_avg_nlinks = 0
    stat_avg_nlinks_filtered = 0
    pruned_graphs = []
    for g in graphs:
        nodes_map = {node['id'] : node['label'] for node in g['nodes']}
        links = []
        for link in g['links']:
            l,h = get_likelihood(nodes_map, link, kb)
            if l is not None:
                links.append(link)
        if len(links)!=len(g['links']):
            in_edge_ids = []
            for link in links:
                in_edge_ids.append(link['s'])
                in_edge_ids.append(link['r'])
            in_edge_ids = set(in_edge_ids)
            nodes_list = []
            for n in g['nodes']:
                if n['id'] in in_edge_ids:
                    nodes_list.append(n)
        else:
            nodes_list = g['nodes']
        pruned_graph = {'directed':g['directed'],'multigraph':g['multigraph'],'graph':g['graph'],'nodes': nodes_list, 'links':links}
        pruned_graphs.append(pruned_graph)
        # Update statistics
        stat_avg_nlinks += len(g['links'])
        stat_avg_nlinks_filtered += len(pruned_graph['links'])

    print(f"Average number of links in graphs: {stat_avg_nlinks/len(graphs)}")
    print(f"Average number of links in pruned graphs: {stat_avg_nlinks_filtered / len(graphs)}")

    return pruned_graphs