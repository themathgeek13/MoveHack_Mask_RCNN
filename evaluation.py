import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.india import balloon

import cv2

import glob
#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BALLON_WEIGHTS_PATH = "mask_rcnn_balloon_0050.h5"  # TODO: update this path

config = balloon.BalloonConfig()
BALLOON_DIR = os.path.join(ROOT_DIR, "/home/rohan/AutoNUE/Mask_RCNN/datasets/anue")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Load validation dataset
dataset = balloon.BalloonDataset()
dataset.load_balloon(BALLOON_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)



# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
weights_path = "/home/rohan/AutoNUE/Mask_RCNN/mask_rcnn_balloon_0050.h5"

# Or, load the last model you trained
# weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
model.save("mask_rcnn_model.h5")
"""
#filename = "/home/rohan/val_instance/pred/test/119/903127_leftImg8bit.png"
for filename in glob.iglob("/home/rohan/83df760fdc8bbc2c6a60cf1e8353a5f4_instance/pred/test/"+"**/*/*", recursive=True):
	print(filename)
	img = cv2.imread(filename)
	imgfilename = filename.split('/')[-1]
	txtfilename = filename.split('.')[0]+".txt"
	f=open(txtfilename,"w+")
	results = model.detect([img], verbose=1)
	r = results[0]

	ax = get_ax(1)
	#r = results[0]
	visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
	h,w,c = r['masks'].shape		################
	netstr = []
	for ind in range(c):
		maskfilename = filename.split('.')[0]+"_"+str(ind)+".png"
		print(r['masks'].shape)
		mask = r['masks'][:,:,ind]
		mask.dtype = 'uint8'
		cv2.imwrite(maskfilename,mask*255)
		netstr.append(maskfilename.split('/')[-1]+" "+str(r["class_ids"][ind]+5)+" "+str(r["scores"][ind]))
	f.write('\n'.join(netstr))
	f.close()				################
	#os.remove(filename)

#img = cv2.imread(filename)

#results = model.detect([img], verbose=1)
#class_names = open("/home/rohan/AutoNUE/Mask_RCNN/datasets/anue/allclasses.txt").read().split()[6:19]
#r=results[0]
#visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
#                            dataset.class_names, r['scores'])

#print(r['masks'].shape)
"""
