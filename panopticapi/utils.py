from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
import traceback
import json
import numpy as np
from PIL import Image

# The decorator is used to prints an error trhown inside process
from config import COCO_panoptic_cat_info_path, COCO_panoptic_cat_list_path, COCO_train_json_path, COCO_val_json_path, \
    COCO_train2014_captions_json_path, COCO_val2014_captions_json_path


def get_traceback(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print('Caught exception in worker thread:')
            traceback.print_exc()
            raise e

    return wrapper


class IdGenerator():
    '''
    The class is designed to generate unique IDs that have meaningful RGB encoding.
    Given semantic category unique ID will be generated and its RGB encoding will
    have color close to the predefined semantic category color.
    The RGB encoding used is ID = R * 256 * G + 256 * 256 + B.
    Class constructor takes dictionary {id: category_info}, where all semantic
    class ids are presented and category_info record is a dict with fields
    'isthing' and 'color'
    '''
    def __init__(self, categories):
        self.taken_colors = set([0, 0, 0])
        self.categories = categories
        for category in self.categories.values():
            if category['isthing'] == 0:
                self.taken_colors.add(tuple(category['color']))

    def get_color(self, cat_id):
        def random_color(base, max_dist=30):
            new_color = base + np.random.randint(low=-max_dist,
                                                 high=max_dist+1,
                                                 size=3)
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        category = self.categories[cat_id]
        if category['isthing'] == 0:
            return category['color']
        base_color_array = category['color']
        base_color = tuple(base_color_array)
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        else:
            while True:
                color = random_color(base_color_array)
                if color not in self.taken_colors:
                    self.taken_colors.add(color)
                    return color

    def get_id(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color)

    def get_id_and_color(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color), color


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def save_json(d, file):
    with open(file, 'w') as f:
        json.dump(d, f)


def load_png_annotation(img_path):
    """
    Load png annotation in panoptic-segmentation format
    :param img_path: path to input annotation
    :return: numpy array with annotation (ids of the segments)
    """
    pan_pred = np.array(Image.open(img_path), dtype=np.uint32)
    return rgb2id(pan_pred)

def load_panoptic_category_info():
    """
    :return: a list of dictionaries with category information
    """
    with open(COCO_panoptic_cat_info_path, 'r') as f:
        categories_list = json.load(f)
    return {el['id']: el for el in categories_list}

def load_panoptic_categ_list():
    """
    Return panoptic segmentation labels
    :return dictionary. Key=class id, value=label(string)
    """
    panoptic_classes = {}
    with open(COCO_panoptic_cat_list_path) as f:
        for line in f.readlines():
            id, label = line.rstrip('\n').split(":")
            panoptic_classes[int(id)] = label
    return panoptic_classes


def read_train_img_captions(split='train'):
    """
    Read COCO captions
    :param split 'train' or 'val' to indicate COCO train2017 or val2017
    :return map {imgId:[list of captions]}
    """
    with open(COCO_train2014_captions_json_path) as f:
        annotations_train = json.load(f)
    with open(COCO_val2014_captions_json_path) as f:
        annotations_val = json.load(f)
    if split == 'train':
        with open(COCO_train_json_path) as f:
            split_images = json.load(f)
    elif split == 'val':
        with open(COCO_val_json_path) as f:
            split_images = json.load(f)
    else:
        return None

    split_images_ids = {ann['image_id'] for ann in split_images['annotations']}
    img_captions = {}
    for ann in annotations_train['annotations'] + annotations_val['annotations']:
        if ann['image_id'] in split_images_ids:
            if ann['image_id'] in img_captions:
                img_captions[ann['image_id']].append(ann['caption'])
            else:
                img_captions[ann['image_id']] = [ann['caption']]
    return img_captions