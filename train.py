import os
import re
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from helper import save_obj
from pose_block import initial_pose_estimation
from create_renderings import create_refinement_inputs
from pose_refinement import train_pose_refinement
from correspondence_block import train_correspondence_block
from create_ground_truth import create_GT_masks, create_UV_XYZ_dictionary, dataset_dir_structure

parser = argparse.ArgumentParser(
    description='Script to create the Ground Truth masks')
parser.add_argument("--root_dir", default="/home/jovyan/work/LineMOD_Dataset/",
                    help="path to dataset directory")
parser.add_argument("--bgd_dir", default="/home/jovyan/work/val2017/",
                    help="path to background images dataset directory")
parser.add_argument("--split", default=0.7, help="test:train split ratio")
args = parser.parse_args()

root_dir = args.root_dir
background_dir = args.bgd_dir
list_all_images = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".jpg"):  # images that exist
            list_all_images.append(os.path.join(root, file))

num_images = len(list_all_images)
indices = list(range(num_images))
np.random.seed(69)
np.random.shuffle(indices)
split = int(np.floor(args.split * num_images))
train_idx, test_idx = indices[split:], indices[:split]

print("Total number of images: ", num_images)
print(" Total number of training images: ",len(train_idx))
print(" Total number of testing images: ",len(test_idx))

save_obj(list_all_images, root_dir + "all_images_adr")
save_obj(train_idx, root_dir + "train_images_indices")
save_obj(test_idx, root_dir + "test_images_indices")

dataset_dir_structure(root_dir)

# Intrinsic Parameters of the Camera
fx = 572.41140
px = 325.26110
fy = 573.57043
py = 242.04899
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

classes = {'ape': 1, 'benchviseblue': 2, 'cam': 3, 'can': 4, 'cat': 5, 'driller': 6,
           'duck': 7, 'eggbox': 8, 'glue': 9, 'holepuncher': 10, 'iron': 11, 'lamp': 12, 'phone': 13}

print("------ Start creating ground truth ------")
create filled ground truth masks
create_GT_masks(root_dir, background_dir, intrinsic_matrix,classes)
create_UV_XYZ_dictionary(root_dir)  # create UV - XYZ dictionaries
print("----- Finished creating ground truth -----")



print("------ Started training of the correspondence block ------")
train_correspondence_block(root_dir, classes, epochs=20)
print("------ Training Finished ------")

print("------ Started Initial pose estimation ------")
initial_pose_estimation(root_dir, classes, intrinsic_matrix)
print("------ Finished Initial pose estimation -----")

print("----- Started creating inputs for DL based pose refinement ------")
create_refinement_inputs(root_dir, classes, intrinsic_matrix)
print("----- Finished creating inputs for DL based pose refinement")


print("----- Started training DL based pose refiner ------")
train_pose_refinement(root_dir, classes, epochs=10)
print("----- Finished training DL based pose refiner ------")
