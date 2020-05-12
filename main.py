import os
import re
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from create_ground_truth_helper import *
from helper import *

parser = argparse.ArgumentParser(description='Script to create the Ground Truth masks')
parser.add_argument("--root_dir", default = "/home/jovyan/work/LineMOD_Dataset/",help="path to dataset directory")
args = parser.parse_args()
root_dir = args.root_dir

# list_all_images = []
# for root, dirs, files in os.walk("/home/jovyan/work/LineMOD_Dataset"): 
#     for file in files:
#         if file.endswith(".jpg"): #images that exist
#             list_all_images.append(os.path.join(root, file))

# save_obj(list_all_images,"/home/jovyan/work/LineMOD_Dataset/all_images_adr")

# test_size = 0.2
# num_images = len(list_all_images)
# indices = list(range(num_images))
# np.random.shuffle(indices)
# split = int(np.floor(test_size * num_images))
# train_idx, test_idx = indices[split:], indices[:split]

# save_obj(train_idx,"/home/jovyan/work/LineMOD_Dataset/train_images_indices")
# save_obj(test_idx,"/home/jovyan/work/LineMOD_Dataset/test_images_indices")