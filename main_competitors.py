
import pyximport
pyximport.install(language_level=3)


from sims.prs import edge_pruning, filter_PRS_histograms, get_sup_ent_lists, node_pruning, load_PRS
from sims.scene_graphs.image_processing import getImageName
from shutil import copyfile
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from config import COCO_img_train_dir, COCO_train_graphs_subset_json_path, COCO_train_graphs_subset2_json_path, \
    COCO_train_graphs_subset3_json_path
import numpy as np
from sklearn.cluster import KMeans
from joblib import dump, load
import pandas as pd
from datetime import datetime
import json
from pyclustering.cluster.kmedoids import kmedoids
from tqdm import tqdm
from sims.graph_algorithms import get_isomorphism_count_vect, compute_coverage_matrix
from sims.sgs_evaluation import evaluate_summary_graphs
from sims.sims_config import SImS_config
from sims.visualization import print_graphs
from sims.graph_algorithms import compute_diversity

competitors_dir = '../COCO/competitors/'

def __get_SIFT(img, sift):
    """
    Return SIFT descriptors for this image (resized 200x200)
    """
    img = cv2.resize(img, (200,200))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (kps, descs) = sift.detectAndCompute(img_gray, None)
    return kps, descs

def __get_descriptors(img_names):
    """
    Return SIFT descriptors matrix (N. images x 100 x 128)
    Max 100 descriptors for each image
    """
    X = None
    sift = cv2.xfeatures2d.SIFT_create()
    for i, img_name in enumerate(img_names):
        img = cv2.imread(os.path.join(COCO_img_train_dir, img_name))
        kps, descs = __get_SIFT(img, sift)
        if descs is not None and len(descs)>0:
            if X is None:
                X = descs[:100]
            else:
                X = np.vstack([X, descs[:100]])
    return X

def get_BOW(img_names, codebook):
    """
    Compute Bag Of Words features for all the specified images
    Each image is described with a normalized histograms that counts the presence of the 500 SIFT descriptors
    :param img_names: list of image names (COCO train)
    :param codebook: Kmeans model with SIFT features codebook (500 elements/centroids)
    :return: features matrix (n_images x 500)
    """
    X = None
    sift = cv2.xfeatures2d.SIFT_create()
    for i, img_name in enumerate(img_names):    # 34 secondi con 1000 immagini
        img = cv2.imread(os.path.join(COCO_img_train_dir, img_name))
        kps, descs = __get_SIFT(img, sift)
        fvect = np.zeros(codebook.n_clusters)
        if descs is not None and len(descs)>0:
            y = codebook.predict(descs)
            unique, counts = np.unique(y, return_counts=True)
            for u, c in zip(unique, counts):
                fvect[u]=c/len(descs)

        if X is None:
            X = fvect
        else:
            X = np.vstack([X, fvect])
        if i%1000 == 0:
            print(f"Done: {i}")
    return X

def compute_BOW_descriptors():
    """
    Compute Bag Of Words features for all COCO train images
    Each image is described with a normalized histograms that counts the presence of the 500 SIFT descriptors
    """
    images = sorted(os.listdir(COCO_img_train_dir))
    selected = np.random.choice(images, 10000, replace=False)
    # Computing descriptors
    print("Computing descriptors...")
    start_time = datetime.now()
    X = __get_descriptors(selected)
    X.dump(os.path.join(competitors_dir, "sift_descr_collection.np"))
    print("Saved.")
    end_time = datetime.now()
    print('Duration: ' + str(end_time - start_time))

    print("Computing codebook with KMeans...")
    start_time = datetime.now()
    X = np.load(os.path.join(competitors_dir, "sift_descr_collection.np"),allow_pickle=True)
    print(f"Initial data: {X.shape[0]}")
    X = X[np.random.choice(X.shape[0], 100000, replace=False), :] # 100K samples
    print(f"Sampled data: {X.shape[0]}")
    codebook = KMeans(500) # Number of codes
    y = codebook.fit_transform(X)
    dump(codebook, os.path.join(competitors_dir, "sift_codebook.pkl"))
    print("Saved.")
    end_time = datetime.now()
    print('Duration: ' + str(end_time - start_time))

    print("Computing feature vectors for all images...")
    start_time = datetime.now()
    codebook = load(os.path.join(competitors_dir, "sift_codebook.pkl"))
    X = get_BOW(images, codebook)
    df = pd.DataFrame(X, index=images)
    df.to_csv(os.path.join(competitors_dir, "bow_images.pd"))
    X.dump(os.path.join(competitors_dir, "bow_images.np"))
    print("Saved.")
    end_time = datetime.now()
    print('Duration: ' + str(end_time - start_time))

def read_BOW_images(dataset='COCO_subset'):
    """
    Read features generated with compute_BOW_descriptors()
    :param dataset: 'COCO', 'COCO_subset', 'COCO_subset2' (experiments in SImS white paper)
    :return: pandas matrix with row=image, column=bow features
    """
    # Cluster images with kmedoids
    X = pd.read_csv(os.path.join(competitors_dir, "bow_images.pd"), index_col=0)
    if dataset=='COCO_subset' or dataset=='COCO_subset2':
        # Select experiment images
        if dataset == 'COCO_subset':
            input_path = COCO_train_graphs_subset_json_path
        elif dataset == 'COCO_subset2':
            input_path = COCO_train_graphs_subset2_json_path
        elif dataset == 'COCO_subset3':
            input_path = COCO_train_graphs_subset3_json_path
        else:
            print(f"Dataset {dataset} not recognized")
            exit()

        with open(input_path) as f:
            graphs = json.load(f)
        selected_names = [f"{g['graph']['name']:012d}.jpg" for g in graphs]
        X = X.loc[selected_names]
    return X

def kmedoids_summary(X, k):
    """
    Apply k-medoids to the feature vector X containin images to be summarized
    :param k: number of clusters
    :return: list of image names for the selected medoids
    """
    km = kmedoids(X.to_numpy(), np.random.randint(0,len(X), k))
    start_time = datetime.now()
    print(f"Start clustering process k={k}.")
    km.process()
    med = km.get_medoids()
    end_time = datetime.now()
    print('Done. Duration: ' + str(end_time - start_time))
    images = []
    for m in med:
        img = X.iloc[m].name
        images.append(img)
    return images, end_time - start_time


def get_kmedoids_graphs(kmedoids_result, scene_graphs):
    """
    Given kmedoids result (json generated by run_kmedoids), find associated COCO scene graphs.
    :param kmedoids_result: input map {k:[list of image names (medoids)]}
    :param scene_graphs: scene graphs of the whole image collection (from which medoids have to be extracted)
    :return: map with {k:[list of graphs]}
    """
    # Load training graphs
    graph_map = {g['graph']['name']: g for g in scene_graphs}
    # Analyze kmedoids result
    kmedoids_graphs = {}
    for k, res in kmedoids_result.items():
        images, time = res
        kmedoids_graphs_i = []
        for img_name in images:
            kmedoids_graphs_i.append(graph_map[int(img_name.split('.')[0])])
        kmedoids_graphs[k] = kmedoids_graphs_i
    return kmedoids_graphs

if __name__ == "__main__":
    class RUN_CONFIG:
        compute_BOW_descriptors = False # Map each COCO image to its BOW descriptors
        run_kmedoids = True             # Run KMedoids summary for different k values
        print_kmedoids_graphs = True   # Print scene graphs of selected kmedoids images (for each k)

        use_full_graphs = False         # True if you want to compute coverage on full graphs
                                        # False to apply node and edge pruning before computing coverage
        pairing_method = 'img_min'  # Method used to associate images to SGS graphs (see associate_img_to_sgs() in sgs.pyx)
                                     # img_min, img_max, img_avg, std

        compute_kmedoids_coverage_matrix = True # Compute graph coverage matrix for kmedoids
        compute_coverage = True  # Use coverage matrix to compute graph coverage

        #dataset = 'COCO_subset'     # Choose dataset
        #dataset = 'COCO_subset2'
        dataset = 'COCO_subset3'
        if dataset == 'COCO_subset':
            mink = 4                        # Min value of k to test kmedoids
            maxk = 20                       # Max value of k to test kmedoids
        elif dataset == 'COCO_subset2' or dataset == 'COCO_subset3':
            mink = 2
            maxk = 20

    # Paths:
    output_path = os.path.join(competitors_dir, RUN_CONFIG.dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    kmedoids_out_clusters_path = os.path.join(output_path, "centroids.json")

    # --------------------------

    # Feature extraction for each image in COCO training set
    if RUN_CONFIG.compute_BOW_descriptors:
        compute_BOW_descriptors()
    # KMedoids summary for different k values
    if RUN_CONFIG.run_kmedoids:
        X = read_BOW_images(RUN_CONFIG.dataset)
        res = {}
        avg_time = 0
        print(f"Number of images: {len(X)}")
        for k in range(RUN_CONFIG.mink, RUN_CONFIG.maxk+1):
            medoids, duration = kmedoids_summary(X, k)
            res[k] = (medoids, duration.seconds)
            avg_time += duration.seconds
            print(f"{k}: {medoids}")
            with open(os.path.join(output_path, "log.txt"),'a+') as f:
                f.write(f"{k}: {medoids}\n")
        print(str(avg_time/len(res)))
        with open(os.path.join(output_path, "avgTime.txt"), 'w') as f:
            f.write('Average time for kmedoids run on COCO subset (seconds):\n')
            f.write(str(avg_time/len(res)))
        with open(kmedoids_out_clusters_path,'w') as f:
            json.dump(res, f)
    # Print graphs associated to kmedoids
    if RUN_CONFIG.print_kmedoids_graphs:
        config = SImS_config(RUN_CONFIG.dataset)
        with open(kmedoids_out_clusters_path) as f:
            kmedoids_result = json.load(f)
        with open(config.scene_graphs_json_path, 'r') as f:
            coco_graphs = json.load(f)
        kmedoids_graphs = get_kmedoids_graphs(kmedoids_result, coco_graphs)

        for k, graphs in kmedoids_graphs.items():
            out_graphs_dir = os.path.join(output_path,'kmedoids_graphs',f'k{k}')
            if not os.path.exists(out_graphs_dir):
                os.makedirs(out_graphs_dir)
            print_graphs(graphs, out_graphs_dir)
            for i, g in enumerate(graphs):
                imgName = getImageName(g['graph']['name'], extension='jpg')
                copyfile(os.path.join(config.img_dir, imgName), os.path.join(out_graphs_dir, f"g{i}.jpg"))

    # Compute graph coverage for kmedoids
    if RUN_CONFIG.compute_kmedoids_coverage_matrix:
        config = SImS_config(RUN_CONFIG.dataset)
        with open(kmedoids_out_clusters_path) as f:
            kmedoids_result = json.load(f)
        with open(config.scene_graphs_json_path, 'r') as f:
            coco_graphs = json.load(f)
        kmedoids_graphs = get_kmedoids_graphs(kmedoids_result, coco_graphs)

        # Load pairwise relationship summary (PRS) if needed
        if RUN_CONFIG.use_full_graphs==False:
            prs = load_PRS(config, True)

        cmatrices_list = []
        for k, summary_graphs_i in kmedoids_graphs.items():
            if RUN_CONFIG.use_full_graphs == False:
                summary_graphs_i = edge_pruning(prs, summary_graphs_i)
                summary_graphs_i = node_pruning(summary_graphs_i)

            cmatrix = compute_coverage_matrix(coco_graphs, [{'g':s} for s in summary_graphs_i])
            cmatrix.columns = list(range(int(k)))
            cmatrix['k'] = k
            #cmatrix.insert(0, 'k', k)
            cmatrices_list.append(cmatrix)

        cmatrices = pd.concat(cmatrices_list, sort=True)
        cmatrices.set_index('k', inplace=True)
        cmatrices.index.name = 'k'
        if RUN_CONFIG.use_full_graphs:
            output_file = os.path.join(output_path, "coverage_mat_full.csv")
        else:
            output_file = os.path.join(output_path, "coverage_mat_pruned.csv")
        cmatrices.to_csv(output_file, sep=",")

    if RUN_CONFIG.compute_coverage:


        config = SImS_config(RUN_CONFIG.dataset)
        with open(kmedoids_out_clusters_path) as f:
            kmedoids_result = json.load(f)
        with open(config.scene_graphs_json_path, 'r') as f:
            coco_graphs = json.load(f)
        kmedoids_graphs = get_kmedoids_graphs(kmedoids_result, coco_graphs)

        if RUN_CONFIG.use_full_graphs==False:
            suffix = "_pruned"
            suffix2=f"_{RUN_CONFIG.pairing_method}"
        else:
            suffix = "_full"
            suffix2 = ""
        cmatrices = pd.read_csv(os.path.join(output_path, f"coverage_mat{suffix}.csv"), index_col='k')

        # Load pairwise relationship summary (PRS) if needed
        if RUN_CONFIG.use_full_graphs==False:
            prs = load_PRS(config, True)

        results = []
        for k, summary_graphs_i in kmedoids_graphs.items():
            if RUN_CONFIG.use_full_graphs == False:
                summary_graphs_i = edge_pruning(prs, summary_graphs_i)
                summary_graphs_i = node_pruning(summary_graphs_i)

            res = evaluate_summary_graphs([{'g':s} for s in summary_graphs_i], cmatrices.loc[int(k)].iloc[:,:int(k)])
            results.append(res)

        kmed_df = pd.DataFrame(results, columns=["N. graphs",
                                                "Avg. nodes", "Std. nodes",
                                                "Coverage",
                                                "Diversity"])

        sims_df = pd.read_csv(os.path.join(config.SGS_dir, f'evaluation{suffix2}.csv'), index_col=0)

        fig, ax = plt.subplots(1,2, figsize=[9,3])
        ax[0].plot(np.arange(RUN_CONFIG.mink, RUN_CONFIG.maxk + 1), sims_df.loc[sims_df['N. graphs'] >= RUN_CONFIG.mink]['Coverage'], label='SImS',
                   marker='o', markersize='4', color='#33a02c', markerfacecolor='#b2df8a')
        ax[0].plot(np.arange(RUN_CONFIG.mink,RUN_CONFIG.maxk + 1), kmed_df['Coverage'], label='KMedoids',
                   marker='o', markersize='4', color='#1f78b4', markerfacecolor='#a6cee3')
        ax[0].set_xlabel('# graphs (k)')
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[0].set_ylabel('coverage')
        ax[0].grid(axis='y')
        ax[1].plot(np.arange(RUN_CONFIG.mink, RUN_CONFIG.maxk + 1), sims_df.loc[sims_df['N. graphs'] >= RUN_CONFIG.mink]['Diversity'], label='SImS',
                   marker='o', markersize='4', color='#33a02c', markerfacecolor='#b2df8a')
        ax[1].plot(np.arange(RUN_CONFIG.mink,RUN_CONFIG.maxk + 1), kmed_df['Diversity'], label='KMedoids',
                   marker='o', markersize='4', color='#1f78b4', markerfacecolor='#a6cee3')

        ax[1].set_xlabel('# graphs (k)')
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].set_ylabel('diversity')
        ax[1].grid(axis='y')
        ax[1].legend(bbox_to_anchor=(1.6, 0.4), loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'evaluation{suffix2}.eps'), bbox_inches='tight')

        fig, ax = plt.subplots(1,2, figsize=[8,3], sharey=True)
        ax[0].plot(np.arange(RUN_CONFIG.mink,RUN_CONFIG.maxk + 1), kmed_df['Coverage'], label='KMedoids',
                   marker='o', markersize='4', color='#1f78b4', markerfacecolor='#a6cee3')
        ax[1].plot(np.arange(RUN_CONFIG.mink, RUN_CONFIG.maxk + 1), sims_df.loc[sims_df['N. graphs'] >= RUN_CONFIG.mink]['Coverage'], label='Coverage',
                   marker='o', markersize='4', color='#1f78b4', markerfacecolor='#a6cee3')
        ax[0].set_xlabel('# graphs (k)')
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        #ax[0].set_ylabel('coverage')
        ax[0].grid(axis='y')
        ax[0].set_title('KMedoids')
        ax[0].plot(np.arange(RUN_CONFIG.mink,RUN_CONFIG.maxk + 1), kmed_df['Diversity'], label='KMedoids',
                   marker='o', markersize='4', color='#33a02c', markerfacecolor='#b2df8a')
        ax[1].plot(np.arange(RUN_CONFIG.mink, RUN_CONFIG.maxk + 1), sims_df.loc[sims_df['N. graphs'] >= RUN_CONFIG.mink]['Diversity'], label='Diversity',
                   marker='o', markersize='4', color='#33a02c', markerfacecolor='#b2df8a')
        ax[1].set_xlabel('# graphs (k)')
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        #ax[1].set_ylabel('diversity')
        ax[1].grid(axis='y')
        ax[1].legend(bbox_to_anchor=(1.6, 0.4), loc="lower right")
        ax[1].set_title('SImS')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'evaluation2{suffix2}.eps'), bbox_inches='tight')
