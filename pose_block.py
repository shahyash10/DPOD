import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import unet_model as UNET
from torch.utils.data.sampler import SubsetRandomSampler
from create_ground_truth_helper import *
from helper import load_obj
%load_ext autoreload
%autoreload 2


correspondence_block.load_state_dict(torch.load('correspondence_block.pt'
							,map_location=torch.device('cpu')))

fx=572.41140; px=325.26110; fy=573.57043; py=242.04899 # Intrinsic Parameters of the Camera
intrinsic_matrix =  np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]],dtype = "double")

def create_6D_poses(root_dir,mapping_dir):
	"""
    Use PnP and Ransac to find the 6D pose and store these files
	Args:
		root_dir (str): address where you want to store pose predictions
	Returns:
		rigid transformation (np array): rotation and translation matrix combined
    """
	base = '00000'
	for i in range(len(train_data)):
	    if i % 100 == 0:
	        print(str(i) + "/1214 finished!")
	    img,_, _ , _ = train_data[i]
	    idmask_pred2,umask_pred2,vmask_pred2 = correspondence_block(img.cuda())
	    # convert the masks to 240,320 shape
	    temp = torch.argmax(idmask_pred2,dim=1).squeeze().cpu() 
	    upred = torch.argmax(umask_pred2,dim=1).squeeze().cpu()
	    vpred = torch.argmax(vmask_pred2,dim=1).squeeze().cpu()
	    for idx,obj in enumerate(classes[:-1]):
	        coord_2d = (temp == idx+1).nonzero(as_tuple=True)
	        str_i = base[:-len(str(i))] + str(i)
	        adr = root_dir + obj + "/info_" + str_i + ".txt"
	        if len(coord_2d): # check of the object is in the mask
	            coord_2d = torch.cat((coord_2d[0].view(coord_2d[0].shape[0],1),coord_2d[1].view(coord_2d[1].shape[0],1)),1)
	            uvalues = upred[coord_2d[:,0],coord_2d[:,1]]
	            vvalues = vpred[coord_2d[:,0],coord_2d[:,1]]
	            dct_keys = torch.cat((uvalues.view(-1,1),vvalues.view(-1,1)),1)
	            dct_keys = tuple(dct_keys.numpy())
	            dct = load_obj(mapping_dir + obj)
	            mapping_2d = []
	            mapping_3d = []
	            for count,(u, v) in enumerate(dct_keys):
	                if (u, v) in dct:
	                    mapping_2d.append(np.array(coord_2d[count]))
	                    mapping_3d.append(dct[(u,v)])
	            if len(mapping_2d) >= 6 or len(mapping_3d) >= 6: 
	            # PnP needs atleast 6 unique 2D-3D correspondences to run
	                _,rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(mapping_3d,
	                	dtype=np.float32),np.array(mapping_2d,dtype = np.float32)
	                	,intrinsic_matrix,distCoeffs = None,
	                    iterationsCount = 150, reprojectionError = 1.0,flags = cv2.SOLVEPNP_P3P)
	                rot, _ = cv2.Rodrigues(rvecs,jacobian = None)
	                rot_tra = np.append(rot,tvecs,axis = 1)
	                # save the predicted pose
	                np.savetxt(adr,rot_tra)
	            else: # save an empty file
	                np.savetxt(adr,np.array([]))
	        else: # save an empty file
	            np.savetxt(adr,np.array([]))

create_6D_poses(root_dir = "/home/jovyan/work/OcclusionChallengeICCV2015/predicted_pose/",
				mapping_dir = "/home/jovyan/work/UV-XYZ Mapping/")