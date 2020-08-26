"""
Author: Andrea Pasini
This file provides the code for running the following experiments:
- Compute the Scene Graph Summary with graph mining and different configurations
- Visualize frequent scene graphs in the SGS
- Analyze SGS statistics
"""
import pyximport
pyximport.install(language_level=3)

from datetime import datetime
from sims.sims_config import SImS_config
from sims.sgs_evaluation import evaluate_SGS, create_COCO_images_subset, create_COCO_images_subset2
from sims.sgs import build_SGS, load_and_print_SGS
from sims.graph_algorithms import compute_coverage_mat
import pandas as pd
import sys


def main():
    ### Choose methods to be run ###
    class RUN_CONFIG:
        #1. Select an experimental configuration
        experiment = 8 # Index of the experiment configuration to be run (if not specified as command-line argument)
        #2. Choose a dataset
        dataset = 'COCO'
        #dataset = 'COCO_subset' # Experiment with only 4 COCO scenes (for paper comparisons)
        #dataset = 'COCO_subset2' # Experiment with images selected by COCO caption
        # dataset = 'VG'

        #3. Run one of the following options
        compute_SGS = False           # Compute the Scene Graph Summary
        compute_coverage_mat = True  # Associate training COCO images to SGS: coverage matrix (7 minutes for experiment 8)
        print_SGS_graphs = False      # Plot SGS scene graphs



        ########### TODO #############
        evaluate_SGS = False         # Plot table with statistics for the different SGS configurations




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




















    if RUN_CONFIG.evaluate_SGS:

        ####!!!!!! TODO: calcolare la coverage !!!!!!!
        # Important: at least "2 nodes" to be considered.!!!!!!!!!!!1

        if RUN_CONFIG.dataset=='COCO':
            exp_list = [11, 1, 6, 8, 4, 9]    # Selected experiments for analyzing statistics
        else:
            exp_list = [11,6]

        results = []
        for selected_experiment in exp_list:
            config.setSGS_params(experiments[selected_experiment])
            res = evaluate_SGS(config)
            results.append(res)
        print("Graph mining statistics.")
        res_df = pd.DataFrame(results, columns=["Minsup","Edge pruning","Node pruning","N. graphs",
                                                "Sub-topic Coverage","Distinct Set Ratio","Avg. nodes","Std. nodes",
                                                "Distinct Node Ratio"])#,"Max. distinct classes","Avg. distinct classes"
        # Print latex table
        print(res_df.to_latex(index=False))



if __name__ == '__main__':
    main()


