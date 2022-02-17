"""
This file applies the complete Semantic Image Summarization (SImS) pipeline for a given configuration.
"""
import pyximport

pyximport.install(language_level=3)
import pandas as pd
from sims.prs import create_PRS
from sims.sgs import build_SGS
from sims.scene_graphs.position_classifier import create_scene_graphs
from sims.sgs_evaluation import evaluate_SGS_df, compute_coverage_mat_sims
from config import position_classifier_path
from sims.scene_graphs.vgenome import create_scene_graphs_vg
from sims.sims_config import SImS_config
from datetime import datetime

class RUN_CONFIG:
    dataset = 'COCO' # VG
    create_graphs = False
if __name__ == "__main__":
    start_time = datetime.now()
    config = SImS_config(RUN_CONFIG.dataset)

    # Compute scene graphs (may take 4 hours for COCO)
    if RUN_CONFIG.create_graphs:
        if RUN_CONFIG.dataset == 'COCO':
            create_scene_graphs(position_classifier_path, config.ann_json_path, config.ann_dir,
                                config.scene_graphs_json_path)
        elif RUN_CONFIG.dataset == 'VG':
            config.setPRS_params(minsup=20) # Use 20, instead of median
            create_scene_graphs_vg(config.ann_json_path, config.scene_graphs_json_path)
    # Build PRS (about 1 minute for COCO)
    create_PRS(config.scene_graphs_json_path, config.PRS_dir, config.PRS_json_path)
    # Build SGS (about 10 seconds for COCO)
    build_SGS(config)
    # Compute coverage matrix (for computing coverage and diversity)
    compute_coverage_mat_sims(config)
    # Run evaluation
    print(evaluate_SGS_df(config).to_latex(index=False))
    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))