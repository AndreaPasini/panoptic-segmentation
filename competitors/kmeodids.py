import json
import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.cluster import KMeans

from config import COCO_img_train_dir, COCO_train_graphs_subset_json_path, COCO_train_graphs_subset2_json_path, \
    COCO_train_graphs_subset3_json_path
from sims.sgs_evaluation import create_COCO_images_subset, create_COCO_images_subset2, create_COCO_images_subset3

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
    if dataset!='COCO':
        # Select experiment images
        if dataset == 'COCO_subset':
            input_path = COCO_train_graphs_subset_json_path
            if not os.path.exists(input_path):
                create_COCO_images_subset()
        elif dataset == 'COCO_subset2':
            input_path = COCO_train_graphs_subset2_json_path
            if not os.path.exists(input_path):
                create_COCO_images_subset2()
        elif dataset == 'COCO_subset3':
            input_path = COCO_train_graphs_subset3_json_path
            if not os.path.exists(input_path):
                create_COCO_images_subset3()
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
    from pyclustering.cluster.kmedoids import kmedoids
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