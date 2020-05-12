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
from helper import save_obj

parser = argparse.ArgumentParser(description='Script to create the Ground Truth masks')
parser.add_argument("--root_dir", default = "/home/jovyan/work/LineMOD_Dataset/",help="path to dataset directory")
args = parser.parse_args()

root_dir = args.root_dir

fx=572.41140; px=325.26110; fy=573.57043; py=242.04899 # Intrinsic Parameters of the Camera
intrinsic_matrix =  np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

#create_GT_masks(root_dir,intrinsic_matrix)  # create filled ground truth masks 

classes = ['ape', 'benchviseblue', 'bowl', 'can', 'cat', 'cup', 'driller', 'duck', 'glue', 'holepuncher', 
            'iron', 'lamp', 'phone', 'cam','eggbox']

create_UV_XYZ_dictionary(root_dir,classes)  # create UV - XYZ dictionaries