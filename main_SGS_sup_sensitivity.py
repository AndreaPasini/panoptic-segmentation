"""
This file applies the SGS generation with different PRS minsup values.
It represents a sensitivity analysis for PRS parameters.
"""
import pyximport

pyximport.install(language_level=3)
import pandas as pd
from sims.prs import create_PRS
from sims.sgs import build_SGS
from sims.scene_graphs.position_classifier import create_scene_graphs
from sims.sgs_evaluation import evaluate_SGS_df, evaluate_SGS
from sims.graph_algorithms import compute_coverage_mat
from config import position_classifier_path
from sims.scene_graphs.vgenome import create_scene_graphs_vg
from sims.sims_config import SImS_config
from datetime import datetime


if __name__ == "__main__":
    start_time = datetime.now()
    # A)
    # config = SImS_config('COCO_subset2')
    # config.SGS_params['minsup'] = 0.05
    # B)
    config = SImS_config('COCO')

    eval_list = []
    for sup in range(800, 500, -100):
        config.setPRS_params(minsup=sup)
        # Build SGS (about 10 seconds for COCO)
        build_SGS(config, overwrite_PRS_cache=True)
        # Compute coverage matrix (for computing coverage and diversity)
        compute_coverage_mat(config)
        # Run evaluation
        res = evaluate_SGS(config)
        res['PRS minsup'] = sup
        eval_list.append(res)
        print(f"minsup = {sup}, coverage = {res['Coverage']}, diversity = {res['Diversity']}")

    res_df = pd.DataFrame(eval_list, columns=["PRS minsup", "Minsup", "N. graphs",
                                            "Avg. nodes","Std. nodes",
                                            "Coverage",
                                            "Diversity"])


    print(res_df.to_latex(index=False))
    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))