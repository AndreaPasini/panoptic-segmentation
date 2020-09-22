"""
Author: Andrea Pasini
This file provides the code for running the following experiments:
- Compute the Scene Graph Summary with graph mining and different configurations
- Visualize frequent scene graphs in the SGS
- Analyze SGS with coverage and diversity
"""
import pyximport
pyximport.install(language_level=3)

from datetime import datetime
from sims.sims_config import SImS_config
from sims.sgs_evaluation import evaluate_SGS, create_COCO_images_subset, create_COCO_images_subset2
from sims.sgs import build_SGS, load_and_print_SGS, load_and_print_SGS_images
from sims.graph_algorithms import compute_coverage_mat
import pandas as pd
import sys
import os

def main():
    ### Choose methods to be run ###
    class RUN_CONFIG:
        #1. Select an experimental configuration
        experiment = 12 # Index of the experiment configuration to be run (if not specified as command-line argument)
        #2. Choose a dataset
        #dataset = 'COCO'
        #dataset = 'COCO_subset' # Experiment with only 4 COCO scenes (for paper comparisons)
        dataset = 'COCO_subset2' # Experiment with images selected by COCO caption
        # dataset = 'VG'

        #3. Run one of the following options
        compute_SGS = False           # Compute the Scene Graph Summary
        compute_coverage_mat = False  # Associate training COCO images to SGS: coverage matrix (7 minutes for experiment 8)
        print_SGS_graphs = False      # Plot SGS scene graphs
        print_SGS_images = True       # Plot images associated to SGS graphs

        #4. Run final evaluation
        #experiment_list = [0, 3, 8, 10, 6, 11] # Experiments for whole COCO (paper)
        experiment_list = [(12, i) for i in range(2, 21)] # Experiment for COCO subset1 and subset2
        evaluate_SGS_experiments = False # Plot table with evaluation for SGS configurations in experiment_list

    # Experiment configuration
    experiments = [
                   {'alg': 'gspan', 'edge_pruning': False, 'node_pruning': False, 'minsup': 0.01},  # 0) 15h 55m
                   {'alg': 'gspan', 'edge_pruning': False, 'node_pruning': True, 'minsup': 0.01},  # 1) 12h 36m
                   {'alg':'gspan', 'edge_pruning':True, 'node_pruning':False, 'minsup':0.1},  #2) 5s
                   {'alg':'gspan', 'edge_pruning':True, 'node_pruning':False, 'minsup':0.01},  #3) 4h,30m
                   {'alg': 'subdue', 'edge_pruning': True, 'node_pruning':False, 'nsubs': 10},  #4) 12h
                   {'alg': 'subdue', 'edge_pruning': True, 'node_pruning':False, 'nsubs': 100},  #5) 12h
                   {'alg': 'subdue', 'edge_pruning': True, 'node_pruning':False, 'nsubs': 10000},  #6) 12h
                   {'alg': 'gspan', 'edge_pruning': True, 'node_pruning': True, 'minsup': 0.1},  #7) 1s
                   {'alg': 'gspan', 'edge_pruning': True, 'node_pruning': True, 'minsup': 0.01},  #8) 2s (GOLD for whole COCO)
                   {'alg': 'gspan', 'edge_pruning': True, 'node_pruning': True, 'minsup': 0.005},  #9) 3s
                   {'alg': 'gspan', 'edge_pruning': True, 'node_pruning': True, 'minsup': 0.001},  #10) 7s
                   {'alg': 'subdue', 'edge_pruning': True, 'node_pruning': True, 'nsubs': 10000},  #11) 17m

                   {'alg': 'gspan', 'edge_pruning': True, 'node_pruning': True, 'minsup': 0.05} #12        (GOLD for COCO subset)
                   ]

    ### SETUP of the experiments ###
    # Experiment selection (with commandline arguments)
    if len(sys.argv) < 2:
        experiment = experiments[RUN_CONFIG.experiment]
    else:
        experiment = experiments[int(sys.argv[1])]
    # SImS configuration
    config = SImS_config(RUN_CONFIG.dataset)
    config.setSGS_params(experiment)


    ### Execution of the different experiment phases ###

    if RUN_CONFIG.compute_SGS:
        # Compute the Scene Graph Summary
        if RUN_CONFIG.dataset == 'COCO_subset':
            create_COCO_images_subset()
        elif RUN_CONFIG.dataset == 'COCO_subset2':
            create_COCO_images_subset2()

        print(f"Selected experiment: {experiment} \non dataset {RUN_CONFIG.dataset}")
        start_time = datetime.now()
        build_SGS(config)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))

    if RUN_CONFIG.compute_coverage_mat:
        # Check subgraph isomorphism of each frequent graph with COCO training images
        start_time = datetime.now()
        compute_coverage_mat(config)
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))


    if RUN_CONFIG.print_SGS_graphs:
        print(f"Selected experiment: {experiment}")
        # Print graphs to file
        # For the 4 article images (issues of graph mining), exp=11
        #load_and_print_SGS(config, subsample = [154, 155, 784, 786])
        # For the 2 examples on node pruning
        #load_and_print_SGS(config, subsample=[1175])
        # load_and_print_SGS(config, subsample=list(range(1100, 1190)))
        load_and_print_SGS(config)

    if RUN_CONFIG.print_SGS_images:
        print(f"Selected experiment: {experiment}")
        load_and_print_SGS_images(config)

    if RUN_CONFIG.evaluate_SGS_experiments:
        # Compute evaluation metrics for all the specified experimental configurations
        results = []
        topk = None
        for selected_experiment in RUN_CONFIG.experiment_list:
            if type(selected_experiment) is tuple:
                topk = selected_experiment[1]
                selected_experiment = selected_experiment[0]
            print(f"Evaluating {experiments[selected_experiment]}...")
            config.setSGS_params(experiments[selected_experiment])
            res = evaluate_SGS(config, topk)
            results.append(res)
        print("Graph mining statistics.")

        res_df = pd.DataFrame(results, columns=["Minsup","Edge pruning","Node pruning","N. graphs",
                                                "Avg. nodes","Std. nodes",
                                                "Coverage",
                                                "Diversity"])


        # Print latex table
        print(res_df.to_latex(index=False))
        res_df.to_csv(os.path.join(config.SGS_dir, 'evaluation.csv'))

if __name__ == '__main__':
    main()


