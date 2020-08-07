from config import out_depth_train_dir, COCO_img_train_dir, out_depth_val_dir, COCO_img_val_dir
from monodepth.monodepth import Monodepth
import numpy as np
import matplotlib.pyplot as plt
import cv2
from os import getpid



def run_model(img_ids):
    pid = getpid()

    step = len(img_ids)//100
    if step==0:
        step=1
    print(f"Process id: {pid}, N. images to analyze: {len(img_ids)}")

    monodepth = Monodepth()

    print(f"P{pid} - Start to compute detections...")
    for i, img_id in enumerate(img_ids):
        # Read image and predict
        img = cv2.imread(img_id)
        pred = monodepth.predict(img)
        # Save detection
        out_depth_train_dir

        if i%step == 0:
            print(f"P{pid} - {100.0*(i+1)/len(img_ids)}%, {i+1} images.")

    return 0

def main():
    ### Choose methods to be run ###
    class RUN_CONFIG:
        compute_train2017_depth = True
        compute_val2017_depth = False

    if RUN_CONFIG.compute_train2017_depth==True:
        output_dir = out_depth_train_dir
        input_dir = COCO_img_train_dir
    elif RUN_CONFIG.compute_val2017_depth==True:
        output_dir = out_depth_val_dir
        input_dir = COCO_img_val_dir

    monodepth = Monodepth()
    image_path = "../COCO/images/train2017/000000000074.jpg"
    img = cv2.imread(image_path)
    res = monodepth.predict(img)

    # Saving colormapped depth image
    vmax = np.percentile(res, 95)

    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.imshow(img)
    plt.title("Input", fontsize=22)
    plt.axis('off')

    plt.subplot(212)
    plt.imshow(res, cmap='magma', vmax=vmax)
    plt.title("Disparity prediction", fontsize=22)
    plt.axis('off');

if __name__=='__main__':
    main()