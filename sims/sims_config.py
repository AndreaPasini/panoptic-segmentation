"""
This file contains the class for configuring SImS to a given dataset
and with specific hyperparameter settings.
"""
import json
from scipy.stats import entropy
import numpy as np

from config import COCO_SGS_dir, COCO_PRS_json_path, COCO_train_graphs_json_path, VG_SGS_dir, \
    VG_PRS_json_path, VG_train_graphs_json_path, position_labels_csv_path, VG_objects_json_path, \
    VG_predicates_json_path, COCO_train_graphs_subset_json_path, COCO_ann_train_dir, COCO_train_json_path, \
    VG_train_json_path, COCO_PRS_dir, VG_PRS_dir, COCO_train_graphs_subset2_json_path, \
    COCO_train_graphs_subset3_json_path, COCO_img_train_dir, COCO_train_graphs_subset2_agg_json_path, \
    COCO_train_graphs_subset3_agg_json_path, COCO_PRS_agg2_json_path, COCO_PRS_agg3_json_path
from panopticapi.utils import load_panoptic_categ_list

class SImS_config:
    # Edgemap for aggregating edges
    edgemap = {
        'above' : 'higher',
        'side-up': 'higher',
        'below' : 'lower',
        'side-down': 'lower',

        'on' : 'on',
        'hanging' : 'hanging',
        'inside' : 'inside',
        'around' : 'around',
        'side' : 'side'
    }

    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == 'COCO':
            self.ann_dir = COCO_ann_train_dir
            self.img_dir = COCO_img_train_dir
            self.ann_json_path = COCO_train_json_path
            self.scene_graphs_json_path = COCO_train_graphs_json_path
            self.PRS_dir = COCO_PRS_dir
            self.PRS_json_path = COCO_PRS_json_path
            self.SGS_dir = COCO_SGS_dir + "/all/"
        elif dataset == 'COCO_subset':
            self.ann_dir = COCO_ann_train_dir
            self.img_dir = COCO_img_train_dir
            self.ann_json_path = COCO_train_json_path
            self.scene_graphs_json_path = COCO_train_graphs_subset_json_path
            self.PRS_dir = COCO_PRS_dir
            self.PRS_json_path = COCO_PRS_json_path
            self.SGS_dir = COCO_SGS_dir + "/subset/"
        elif dataset == 'COCO_subset2':
            self.ann_dir = COCO_ann_train_dir
            self.img_dir = COCO_img_train_dir
            self.ann_json_path = COCO_train_json_path
            self.scene_graphs_json_path = COCO_train_graphs_subset2_json_path
            self.PRS_dir = COCO_PRS_dir
            self.PRS_json_path = COCO_PRS_json_path
            self.SGS_dir = COCO_SGS_dir + "/subset2/"
        elif dataset == 'COCO_subset3':
            self.ann_dir = COCO_ann_train_dir
            self.img_dir = COCO_img_train_dir
            self.ann_json_path = COCO_train_json_path
            self.scene_graphs_json_path = COCO_train_graphs_subset3_json_path
            self.PRS_dir = COCO_PRS_dir
            self.PRS_json_path = COCO_PRS_json_path
            self.SGS_dir = COCO_SGS_dir + "/subset3/"
        elif dataset == 'COCO_subset2_agg':
            # Subset 2, with aggregated edge labels
            self.ann_dir = COCO_ann_train_dir
            self.img_dir = COCO_img_train_dir
            self.ann_json_path = COCO_train_json_path
            self.scene_graphs_json_path = COCO_train_graphs_subset2_agg_json_path
            self.PRS_dir = COCO_PRS_dir
            self.PRS_json_path = COCO_PRS_agg2_json_path
            self.SGS_dir = COCO_SGS_dir + "/subset2_agg/"
        elif dataset == 'COCO_subset3_agg':
            # Subset 2, with aggregated edge labels
            self.ann_dir = COCO_ann_train_dir
            self.img_dir = COCO_img_train_dir
            self.ann_json_path = COCO_train_json_path
            self.scene_graphs_json_path = COCO_train_graphs_subset3_agg_json_path
            self.PRS_dir = COCO_PRS_dir
            self.PRS_json_path = COCO_PRS_agg3_json_path
            self.SGS_dir = COCO_SGS_dir + "/subset3_agg/"
        elif dataset == 'VG':
            self.ann_dir = None
            self.img_dir = None
            self.ann_json_path = VG_train_json_path
            self.SGS_dir = VG_SGS_dir
            self.PRS_dir = VG_PRS_dir
            self.PRS_json_path = VG_PRS_json_path
            self.scene_graphs_json_path = VG_train_graphs_json_path

        # Default PRS configuration
        self.PRS_params = {'minsup': None, 'maxentr': None}
        # Default SGS configuration
        self.SGS_params = {'alg': 'gspan', 'edge_pruning': True, 'node_pruning': True, 'minsup': 0.01}

    def setPRS_params(self, minsup=None, maxentr=None):
        """
        Overwrite default PRS configuration
        Leave to None the parameters that you don't want to change.
        :param minsup: minimum support of PRS histograms
        :param maxentr: maximum entropy of PRS histograms
        """
        if minsup is not None:
            self.PRS_params['minsup']=minsup
        if maxentr is not None:
            self.PRS_params['maxentr']=maxentr

    def setSGS_params(self, SGS_params):
        """
        Overwrite default SGS configuration
        :param SGS_params: SGS configuration (dict: alg, edge_pruning, node_pruning, min_sup)
        """
        self.SGS_params = SGS_params

    def getSGS_experiment_name(self):
        """
        Given SGS configuration (node/edge pruning, gmining algorithm, minsup),
        create a prefix string describing the experiment.
        Useful to save files where the prefix inticates the experimental configuration.
        :return: string with the prefix
        """
        config = self.SGS_params
        edge_pruning = "_eprune" if config['edge_pruning'] else ""  # Edge pruning
        node_pruning = "_nprune" if config['node_pruning'] else ""  # Node pruning
        if config['alg'] == 'gspan':
            exp_name = f"sgs{edge_pruning}{node_pruning}_{config['alg']}_{str(config['minsup'])[2:]}"
        else:
            exp_name = f"sgs{edge_pruning}{node_pruning}_{config['alg']}_{config['nsubs']}"
        return exp_name

    def load_categories(self):
        """
        :return: object categories and relationship categories for the configured dataset
        """
        if self.dataset == 'COCO' or self.dataset == 'COCO_subset' or self.dataset == 'COCO_subset2' or self.dataset == 'COCO_subset3'\
                or self.dataset == 'COCO_subset2_agg' or self.dataset == 'COCO_subset3_agg':
            obj_categories = load_panoptic_categ_list()
            with open(position_labels_csv_path) as f:
                rel_categories = tuple(s.strip() for s in f.readlines())
            if self.dataset == 'COCO_subset2_agg' or self.dataset == 'COCO_subset3_agg':
                rel_categories = tuple(set(self.edgemap[s] for s in rel_categories))
        elif self.dataset == 'VG':
            with open(VG_objects_json_path) as f:
                obj_categories = {i: l for i, l in enumerate(json.load(f))}
            with open(VG_predicates_json_path) as f:
                rel_categories = json.load(f)
        return obj_categories, rel_categories

    def get_PRS_filters(self, sup_list, entr_list):
        """
        Return min_sup and max_entr for filtering the PRS
        :param sup_list: list of PRS supports
        :param entr_list: list of PRS entropies
        :return: min_sup, max_entr
        """

        # Default minsup computation if not explicitly specified
        min_sup = self.PRS_params['minsup']
        if min_sup is None:
            med = np.median(np.log10(sup_list))
            min_sup = int(round(10 ** med))

        # Default maxentropy computation if not explicitly specified
        max_entropy = self.PRS_params['maxentr']
        if max_entropy is None:
            max_entropy = entropy([1 / 3, 1 / 3, 1 / 3])

        return min_sup, max_entropy

