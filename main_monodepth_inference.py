from PIL import Image

from config import out_depth_train_dir, COCO_img_train_dir, out_depth_val_dir, COCO_img_val_dir
from monodepth.monodepth import Monodepth
import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
import cv2
from os import getpid
import os
import time
from multiprocessing import Pool

def run_model(img_ids, input_dir, output_dir):

    pid = getpid()

    step = len(img_ids)//100
    if step==0:
        step=1
    print(f"Process id: {pid}, N. images to analyze: {len(img_ids)}")

    monodepth = Monodepth()

    print(f"P{pid} - Start to compute detections...")
    for i, img_id in enumerate(img_ids):
        # Read image and predict
        img = cv2.imread(os.path.join(input_dir, img_id+'.jpg'))
        pred = monodepth.predict(img)
        # Save detection
        outimg = Image.fromarray((pred*255).astype(np.uint8), mode='L')
        outimg.save(os.path.join(output_dir, img_id+'.png'))
        #cv2.imwrite(os.path.join(output_dir, img_id+'b.png'), img)

        if i%step == 0:
            print(f"P{pid} - {100.0*(i+1)/len(img_ids)}%, {i+1} images.")

    return 0

def main():
    ### Choose methods to be run ###
    class RUN_CONFIG:
        num_proc = 50
        compute_train2017_depth = True
        compute_val2017_depth = False

    if RUN_CONFIG.compute_train2017_depth==True:
        output_dir = out_depth_train_dir
        input_dir = COCO_img_train_dir
    elif RUN_CONFIG.compute_val2017_depth==True:
        output_dir = out_depth_val_dir
        input_dir = COCO_img_val_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Files to be analyzed
    input_files = {f.split('.')[0] for f in os.listdir(input_dir)}
    output_files = {f.split('.')[0] for f in os.listdir(output_dir)}
    files = sorted(list(input_files-output_files))

    pool = Pool(RUN_CONFIG.num_proc)
    start = time.time()

    # Apply to images
    chuncks = np.array_split(np.array(files), RUN_CONFIG.num_proc)
    pool.starmap(run_model, zip(chuncks, repeat(input_dir), repeat(output_dir)))

    end = time.time()
    print(end - start)
    pool.close()
    pool.join()


if __name__=='__main__':
    main()