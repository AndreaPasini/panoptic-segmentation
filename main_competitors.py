import cv2
import os
from config import COCO_img_train_dir, COCO_train_graphs_subset_json_path
import numpy as np
from sklearn.cluster import KMeans
from joblib import dump, load
import pandas as pd
from datetime import datetime
import json
from pyclustering.cluster.kmedoids import kmedoids
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
    Compute Bag Of Words features for COCO train images
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

def read_BOW_images(COCO_subset=True):
    """
    Read features generated with compute_BOW_descriptors()
    :param COCO_subset True if you want features only for COCO subset (experiments in SImS white paper)
    :return: pandas matrix with row=image, column=bow features
    """
    # Cluster images with kmedoids
    X = pd.read_csv(os.path.join(competitors_dir, "bow_images.pd"), index_col=0)
    if COCO_subset:
        # Select experiment images
        with open(COCO_train_graphs_subset_json_path) as f:
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
    print("Start clustering process.")
    km.process()
    med = km.get_medoids()
    end_time = datetime.now()
    print('Done. Duration: ' + str(end_time - start_time))
    images = []
    for m in med:
        img = X.iloc[m].name
        images.append(img)
    return images, end_time - start_time



if __name__ == "__main__":
    class RUN_CONFIG:
        compute_BOW_descriptors = False # Map each image to its BOW descriptors
        run_k_medoids = True
    out_file = "centroids2.txt"

    # Feature extraction for each image in COCO training set
    if RUN_CONFIG.compute_BOW_descriptors:
        compute_BOW_descriptors()
    # KMedoids summary for different k values
    if RUN_CONFIG.run_k_medoids:
        X = read_BOW_images()
        res = {}
        avg_time = 0
        for k in range(4,21):
            medoids, duration = kmedoids_summary(X, k)
            res[k] = (medoids, duration.seconds)
            avg_time += duration.seconds
            print(f"{k}: {medoids}")
            with open(os.path.join(competitors_dir, "log.txt"),'a+') as f:
                f.write(f"{k}: {medoids}\n")
        print(str(avg_time/len(res)))
        with open(os.path.join(competitors_dir, "avgTime.txt"), 'w') as f:
            f.write(str(avg_time/len(res)))
        with open(os.path.join(competitors_dir, "centroids.json"),'w') as f:
            json.dump(res, f)