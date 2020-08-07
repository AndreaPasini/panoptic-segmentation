"""
Author: Andrea Pasini
This file provides the code for running the following experiments:
- Building scene graphs (with object positions) for COCO (train, val and panoptic predictions)
- Generating the PRS (Pairwise Relationship Summary) from training graphs (histograms and triplets)
"""
import pyximport

from sims.sims_config import SImS_config
from sims.scene_graphs.vgenome import create_scene_graphs_vg

pyximport.install(language_level=3)
from datetime import datetime
from config import position_classifier_path, COCO_panoptic_val_graphs_json_path, out_panoptic_dir, out_panoptic_json_path
from sims.scene_graphs.position_classifier import validate_classifiers_grid_search, build_final_model, create_scene_graphs
from sims.prs import create_PRS

######################


### Choose methods to be run ###
class RUN_CONFIG:
    generate_train_graphs = False   # COCO, Build graphs (object positions) for training images (118,287, may take some hours)
                                    # VisualGenome, build training graphs from annotations
    generate_val_graphs = False     # Build graphs (object positions) for CNN predictions on validation set  (5,000)
    generate_PRS = False            # Generate the PRS from training graphs: save graphs and histograms

    # Choose dataset
    #dataset = 'VG'
    dataset = 'COCO'

if __name__ == "__main__":
    start_time = datetime.now()

    # Select dataset configuration
    config = SImS_config(RUN_CONFIG.dataset)

    if RUN_CONFIG.generate_train_graphs:
        if RUN_CONFIG.dataset == 'COCO':
            # Apply position classifier to images and compute scene graphs
            create_scene_graphs(position_classifier_path, config.ann_json_path, config.ann_dir, config.scene_graphs_json_path)
        elif RUN_CONFIG.dataset == 'VG':
            create_scene_graphs_vg(config.ann_json_path, config.scene_graphs_json_path)
    elif RUN_CONFIG.generate_val_graphs and RUN_CONFIG.dataset=='COCO':
        # Create scene graphs from CNN predictions (panoptic segmentation)
        create_scene_graphs(position_classifier_path, out_panoptic_json_path, out_panoptic_dir, COCO_panoptic_val_graphs_json_path)
    elif RUN_CONFIG.generate_PRS:
        create_PRS(config.scene_graphs_json_path, config.PRS_dir, config.PRS_json_path)

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))
