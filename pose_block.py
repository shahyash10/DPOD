import os
import re
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import unet_model as UNET
from helper import load_obj
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from dataset_classes import LineMODDataset


def initial_pose_estimation(root_dir, classes, intrinsic_matrix):

    # LineMOD Dataset
    train_data = LineMODDataset(root_dir, classes=classes,
                                transform=transforms.Compose([transforms.ToTensor()]))

    # load the best correspondence block weights
    correspondence_block = UNET.UNet(
        n_channels=3, out_channels_id=14, out_channels_uv=256, bilinear=True)
    correspondence_block.cuda()
    correspondence_block.load_state_dict(torch.load(
        'correspondence_block.pt', map_location=torch.device('cpu')))

    # initial 6D pose prediction
    regex = re.compile(r'\d+')
    outliers = 0
    for i in range(len(train_data)):
        if i % 1000 == 0:
            print(str(i) + "/" + str(len(train_data)) + " finished!")
        img_adr, img, idmask, _, _ = train_data[i]
        label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
        idx = regex.findall(os.path.split(img_adr)[1])[0]
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
        idmask_pred, umask_pred, vmask_pred = correspondence_block(img.cuda())
        # convert the masks to 240,320 shape
        temp = torch.argmax(idmask_pred, dim=1).squeeze().cpu()
        upred = torch.argmax(umask_pred, dim=1).squeeze().cpu()
        vpred = torch.argmax(vmask_pred, dim=1).squeeze().cpu()
        coord_2d = (temp == classes[label]).nonzero(as_tuple=True)

        adr = root_dir + label + "/predicted_pose/" + \
            "info_" + str(idx) + ".txt"

        coord_2d = torch.cat((coord_2d[0].view(
            coord_2d[0].shape[0], 1), coord_2d[1].view(coord_2d[1].shape[0], 1)), 1)
        uvalues = upred[coord_2d[:, 0], coord_2d[:, 1]]
        vvalues = vpred[coord_2d[:, 0], coord_2d[:, 1]]
        dct_keys = torch.cat((uvalues.view(-1, 1), vvalues.view(-1, 1)), 1)
        dct_keys = tuple(dct_keys.numpy())
        dct = load_obj(root_dir + label + "/UV-XYZ_mapping")
        mapping_2d = []
        mapping_3d = []
        for count, (u, v) in enumerate(dct_keys):
            if (u, v) in dct:
                mapping_2d.append(np.array(coord_2d[count]))
                mapping_3d.append(dct[(u, v)])
        # Get the 6D pose from rotation and translation matrices
        # PnP needs atleast 6 unique 2D-3D correspondences to run
        if len(mapping_2d) >= 4 or len(mapping_3d) >= 4:
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(mapping_3d, dtype=np.float32),
                                                          np.array(mapping_2d, dtype=np.float32), intrinsic_matrix, distCoeffs=None,
                                                          iterationsCount=150, reprojectionError=1.0, flags=cv2.SOLVEPNP_P3P)
            rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
            rot[np.isnan(rot)] = 1
            tvecs[np.isnan(tvecs)] = 1
            tvecs = np.where(-100 < tvecs, tvecs, np.array([-100.]))
            tvecs = np.where(tvecs < 100, tvecs, np.array([100.]))
            rot_tra = np.append(rot, tvecs, axis=1)
            # save the predicted pose
            np.savetxt(adr, rot_tra)
        else:  # save a pose full of zeros
            outliers += 1
            rot_tra = np.ones((3, 4))
            rot_tra[:, 3] = 0
            np.savetxt(adr, rot_tra)
    print("Number of instances where PnP couldn't be used: ", outliers)
