
# Relative Position Classifier with Depth (RPCD)
import json
from itertools import groupby

import cv2
import os
import pickle
from multiprocessing import Pool
from os import listdir
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyximport
pyximport.install(language_level=3)


from sims.scene_graphs.feature_extraction import image2strings_RPCD
from config import COCO_panoptic_cat_info_path, out_depth_train_dir
from sims.scene_graphs.position_classifier import get_COCO_ann_dictionaries

from panopticapi.utils import load_png_annotation

def compute_string_positions_RPCD(strings, object_ordering=None):
    """
    Compute object positions given string representation.

    ----
    :param strings: string representation
    :param object_ordering: object ids ordered if you want a specific object order the returned pairs
    :return: dictionary with object positions for each object pair

    Pseudocode: position of two objects i, j given a string
    last_object = None
    last_occurrence = -1
    for pos in [0,len(string)]
      current=string[n]
      if current==i or current==j
          if last_object != current
              if pos > last_occurrence + 1
                  obj_list = obj_list + '+' + current
              else
                  obj_list = obj_list + current
          last_object = current
          last_occurrence = pos
    look for obj_list: ij, ji, i+j, j+i, iji, jij
    """
    positions = {}

    # For each string (image column)
    for string_ids, data in strings:
        # Get unique object ids in the string
        if object_ordering:
            objects = [obj for obj in object_ordering if obj in string_ids]
        else:
            objects = sorted(list(set(string_ids)-{0}))
        # For all object pairs
        for i in range(0, len(objects)-1):
            for j in range(i+1, len(objects)):
                obj_i = objects[i]
                obj_j = objects[j]

                pair = (obj_i, obj_j)
                if not (pair in positions.keys()):
                    ij_positions = {"i on j": 0, "j on i": 0,
                                    "i above j": 0, "j above i": 0,
                                    "i support j": 0, "j support i": 0,
                                    "i around j": 0, "j around i": 0, "other": 0}
                    positions[pair] = ij_positions
                else:
                    ij_positions = positions[pair]

                obj_list = ''
                data_list = []
                last_obj = None
                last_occur = -1
                pos = 0
                for current, d_i in zip(string_ids,data):
                    if current == obj_i or current==obj_j:
                        if last_obj != current:
                            if pos > last_occur + 1 and last_occur!=-1:
                                obj_list = obj_list + ('+i' if current == obj_i else '+j')
                                data_list.append(d_i)
                            else:
                                obj_list = obj_list + ('i' if current == obj_i else 'j')
                                data_list.append(d_i)
                        last_obj = current
                        last_occur = pos
                    pos += 1

                if obj_list == 'ij':
                    # touch distance between bottom of i and top of j
                    ij_positions["ij dist"] += abs(data_list[0][2]-data_list[1][1])
                    ij_positions["i on j"] += 1
                elif obj_list == 'ji':
                    ij_positions["ji dist"] += abs(data_list[0][2]-data_list[1][1])
                    ij_positions["j on i"] += 1
                elif obj_list == 'i+j':
                    ij_positions["i above j"] += 1
                elif obj_list == 'j+i':
                    ij_positions["j above i"] += 1
                elif obj_list == 'iji' or obj_list == 'i+ji' or obj_list == 'ij+i' or obj_list == 'i+j+i':
                    ij_positions["i around j"] += 1
                    if obj_list=="iji":
                        ij_positions["ij dist"] = abs(data_list[0][2] - data_list[1][1])
                        ij_positions["ji dist"] = abs(data_list[1][2] - data_list[2][1])
                    elif obj_list=="i+ji":
                        ij_positions["ji dist"] = abs(data_list[1][2] - data_list[2][1])
                    elif obj_list=="ij+i":
                        ij_positions["ij dist"] = abs(data_list[0][2] - data_list[1][1])
                elif obj_list == 'jij' or obj_list == 'j+ij' or obj_list == 'ji+j' or obj_list == 'j+i+j':
                    ij_positions["j around i"] += 1
                else:
                    ij_positions["other"] += 1
    return positions

def image2scene_graph_RPCD(image_name, image_id, segments_info, cat_info, annot_folder, depth_folder, model):
    """
    ** Applicable to data with COCO dataset format. **
    Apply position classifier with depth (RPCD) to this image, compute scene graph
    In each relationships, the order of the pair subject-reference is chosen based on alphabetical order.
    E.g. (ceiling, floor) instead of (floor, ceiling)
    :param image_name: file name of the image
    :param image_id: identifier of the image (number extracted from image name, without leading zeros)
    :param segments_info: json with segment class information
    :param cat_info: COCO category information
    :param annot_folder: path to annotations
    :param model: relative-position classifier
    :return the scene graph
    """
    if len(segments_info)==0:
        print('Image has no segments.')
        return None

    catInf = pd.DataFrame(cat_info).T
    segInfoDf = pd.DataFrame(segments_info)

    merge = pd.concat([segInfoDf.set_index('category_id'), catInf.set_index('id')], axis=1,
                      join='inner').set_index('id')

    result = merge['name'].sort_values()
    img_ann = load_png_annotation(os.path.join(annot_folder, image_name))
    img_depth = cv2.imread(os.path.join(depth_folder, image_name), cv2.IMREAD_GRAYSCALE)
    strings = image2strings_RPCD(img_ann, img_depth)

    object_ordering = result.index.tolist()
    positions = compute_string_positions_RPCD(strings, object_ordering)
    # g = nx.DiGraph()
    # g.name = image_id
    # for id, name in result.iteritems():
    #     g.add_node(id, label=name)
    # for (s, r), pos in list(positions.items()):
    #     featuresRow = get_features(img_ann, "", s, r, positions)
    #     prediction = model.predict([np.asarray(featuresRow[3:])])[0]
    #     g.add_edge(s, r, pos=prediction)
    # return g

def create_scene_graphs_RPCD(fileModel_path, COCO_json_path, COCO_ann_dir, COCO_depth_dir,out_graphs_json_path):
    """
    ** Applicable to data with COCO dataset format. **
    Generate scene graphs from images, applying the relative position classifier
    Use depth information to compute positions
    :param fileModel_path: path to relative position model
    :param COCO_json_path: annotation file with classes for each segment (either CNN annotations or ground-truth)
    :param COCO_ann_dir: folder with png annotations (either CNN annotations or ground-truth)
    :param out_graphs_json_path: output json file with scene graphs
    """

    loaded_model = pickle.load(open(fileModel_path, 'rb'))

    # Load annotations
    id_dict, annot_dict, cat_dict = get_COCO_ann_dictionaries(COCO_json_path)

    # Get files to be analyzed
    ## files = sorted(os.listdir(COCO_ann_dir))
    depth_file_ids = {f.split('.')[0] for f in os.listdir(out_depth_train_dir)}
    files = sorted([f+'.png' for f in depth_file_ids])

    # Init progress bar
    pbar = tqdm(total=len(files))

    def update(x):
        pbar.update()

    print("Number of images: %d" % len(files))
    print("Scheduling tasks...")
    #pool = Pool(10)
    results = []

    # Analyze all images
    for img in files:
        if img.endswith('.png'):
            image2scene_graph_RPCD(img, id_dict[img], annot_dict[img], cat_dict, COCO_ann_dir, COCO_depth_dir, loaded_model)
            # results.append(pool.apply_async(image2scene_graph, args=(img, id_dict[img], annot_dict[img],
            #                                                         cat_dict, COCO_ann_dir, loaded_model), callback=update))
    #pool.close()
    #pool.join()
    #pbar.close()

    # Collect Graph results
    # resultGraph = []
    # for graph_getter in results:
    #     graph = graph_getter.get()
    #     if graph is not None:
    #         # Get graph description for this image
    #         resultGraph.append(nx_to_json(graph))
    #
    # # Write graphs to file
    # with open(out_graphs_json_path, "w") as f:
    #     json.dump(resultGraph, f)
    # print("Done")