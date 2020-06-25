import os
import re
import cv2
import torch
import numpy as np
import torch.nn as nn
import unet_model as UNET
import matplotlib.image as mpimg
from helper import *
from torch.utils.data import Dataset
from torchvision import transforms
from dataset_classes import LineMODDataset
from create_ground_truth import get_rot_tra


def create_rendering(root_dir, intrinsic_matrix, obj, idx):
    # helper function to help with creating renderings
    pred_pose_adr = root_dir + obj + \
        '/predicted_pose' + '/info_' + str(idx) + ".txt"
    rgb_values = np.loadtxt(root_dir + obj + '/object.xyz',
                            skiprows=1, usecols=(6, 7, 8))
    coords_3d = np.loadtxt(root_dir + obj + '/object.xyz',
                           skiprows=1, usecols=(0, 1, 2))
    ones = np.ones((coords_3d.shape[0], 1))
    homogenous_coordinate = np.append(coords_3d, ones, axis=1)
    rigid_transformation = np.loadtxt(pred_pose_adr)
    # Perspective Projection to obtain 2D coordinates
    homogenous_2D = intrinsic_matrix @ (
        rigid_transformation @ homogenous_coordinate.T)
    homogenous_2D[2, :][np.where(homogenous_2D[2, :] == 0)] = 1
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)
    rendered_image = np.zeros((480, 640, 3))
    x_2d = np.clip(coord_2D[:, 0], 0, 479)
    y_2d = np.clip(coord_2D[:, 1], 0, 639)
    rendered_image[x_2d, y_2d, :] = rgb_values
    temp = np.sum(rendered_image, axis=2)
    non_zero_indices = np.argwhere(temp > 0)
    min_x = non_zero_indices[:, 0].min()
    max_x = non_zero_indices[:, 0].max()
    min_y = non_zero_indices[:, 1].min()
    max_y = non_zero_indices[:, 1].max()
    cropped_rendered_image = rendered_image[min_x:max_x +
                                            1, min_y:max_y + 1, :]
    if cropped_rendered_image.shape[0] > 240 or cropped_rendered_image.shape[1] > 320:
        cropped_rendered_image = cv2.resize(np.float32(
            cropped_rendered_image), (320, 240), interpolation=cv2.INTER_AREA)
    return cropped_rendered_image


def create_refinement_inputs(root_dir, classes, intrinsic_matrix):
    correspondence_block = UNET.UNet(
        n_channels=3, out_channels_id=14, out_channels_uv=256, bilinear=True)
    correspondence_block.cuda()
    correspondence_block.load_state_dict(torch.load(
        'correspondence_block.pt', map_location=torch.device('cpu')))

    train_data = LineMODDataset(root_dir, classes=classes,
                                transform=transforms.Compose([transforms.ToTensor()]))

    upsampled = nn.Upsample(size=[240, 320], mode='bilinear',align_corners=False)

    regex = re.compile(r'\d+')
    count = 0
    for i in range(len(train_data)):
        if i % 1000 == 0:
            print(str(i) + "/" + str(len(train_data)) + " finished!")
        img_adr, img, _, _, _ = train_data[i]

        label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
        idx = regex.findall(os.path.split(img_adr)[1])[0]
        adr_rendered = root_dir + label + \
            "/pose_refinement/rendered/color" + str(idx) + ".png"
        adr_img = root_dir + label + \
            "/pose_refinement/real/color" + str(idx) + ".png"
        # find the object in the image using the idmask
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
        idmask_pred, _, _ = correspondence_block(img.cuda())
        idmask = torch.argmax(idmask_pred, dim=1).squeeze().cpu()
        coord_2d = (idmask == classes[label]).nonzero(as_tuple=True)
        if coord_2d[0].nelement() != 0:
            coord_2d = torch.cat((coord_2d[0].view(
                coord_2d[0].shape[0], 1), coord_2d[1].view(coord_2d[1].shape[0], 1)), 1)
            min_x = coord_2d[:, 0].min()
            max_x = coord_2d[:, 0].max()
            min_y = coord_2d[:, 1].min()
            max_y = coord_2d[:, 1].max()
            img = img.squeeze().transpose(1, 2).transpose(0, 2)
            obj_img = img[min_x:max_x+1, min_y:max_y+1, :]
            # saving in the correct format using upsampling
            obj_img = obj_img.transpose(0, 1).transpose(0, 2).unsqueeze(dim=0)
            obj_img = upsampled(obj_img)
            obj_img = obj_img.squeeze().transpose(0, 2).transpose(0, 1)
            mpimg.imsave(adr_img, obj_img.squeeze().numpy())

            # create rendering for an object
            cropped_rendered_image = create_rendering(
                root_dir, intrinsic_matrix, label, idx)
            rendered_img = torch.from_numpy(cropped_rendered_image)
            rendered_img = rendered_img.unsqueeze(dim=0)
            rendered_img = rendered_img.transpose(1, 3).transpose(2, 3)
            rendered_img = upsampled(rendered_img)
            rendered_img = rendered_img.squeeze().transpose(0, 2).transpose(0, 1)
            mpimg.imsave(adr_rendered, rendered_img.numpy())

        else:  # object not present in idmask prediction
            count += 1
            mpimg.imsave(adr_rendered, np.zeros((240, 320)))
            mpimg.imsave(adr_img, np.zeros((240, 320)))
    print("Number of outliers: ", count)
