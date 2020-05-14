import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from create_ground_truth import *
from helper import load_obj


class LineMODDataset(Dataset):

    """
    Args:
        root_dir (str): path to the dataset
        classes (dictionary): values of classes to extract from segmentation mask 
        transform : Transforms for input image
            """

    def __init__(self, root_dir, classes=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.list_all_images = load_obj(root_dir + "all_images_adr")
        self.training_images_idx = load_obj(root_dir + "train_images_indices")

    def __len__(self):
        return len(self.training_images_idx)

    def __getitem__(self, i):

        img_adr = self.list_all_images[self.training_images_idx[i]]
        label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
        regex = re.compile(r'\d+')
        idx = regex.findall(os.path.split(img_adr)[1])[0]
        
        if i % 100 != 0:  # read the image with changed background
            image = cv2.imread(self.root_dir + label +
                               "/changed_background/color" + str(idx) + ".png")
        else:
            image = cv2.imread(img_adr)

        IDmask = cv2.imread(self.root_dir + label + "/ground_truth/IDmasks/color" +
                            str(idx) + ".png", cv2.IMREAD_GRAYSCALE)
        Umask = cv2.imread(self.root_dir + label + "/ground_truth/Umasks/color" +
                           str(idx) + ".png", cv2.IMREAD_GRAYSCALE)
        Vmask = cv2.imread(self.root_dir + label + "/ground_truth/Vmasks/color" +
                           str(idx) + ".png", cv2.IMREAD_GRAYSCALE)
        # resize the masks
        image = cv2.resize(
            image, (image.shape[1]//2, image.shape[0]//2), interpolation=cv2.INTER_AREA)
        IDmask = cv2.resize(
            IDmask, (IDmask.shape[1]//2, IDmask.shape[0]//2), interpolation=cv2.INTER_AREA)
        Umask = cv2.resize(
            Umask, (Umask.shape[1]//2, Umask.shape[0]//2), interpolation=cv2.INTER_AREA)
        Vmask = cv2.resize(
            Vmask, (Vmask.shape[1]//2, Vmask.shape[0]//2), interpolation=cv2.INTER_AREA)
        if self.transform:
            image = self.transform(image)
        IDmask = (torch.from_numpy(IDmask)).type(torch.int64)
        Umask = (torch.from_numpy(Umask)).type(torch.int64)
        Vmask = (torch.from_numpy(Vmask)).type(torch.int64)
        return img_adr, image, IDmask, Umask, Vmask


class PoseRefinerDataset(Dataset):

    """
    Args:
        root_dir (str): path to the dataset directory
        classes (dict): dictionary containing classes as key  
        transform : Transforms for input image
            """

    def __init__(self, root_dir, classes=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.list_all_images = load_obj(root_dir + "all_images_adr")
        self.training_images_idx = load_obj(root_dir + "train_images_indices")

    def __len__(self):
        return len(self.training_images_idx)

    def __getitem__(self, i):
        img_adr = self.list_all_images[self.training_images_idx[i]]
        label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
        regex = re.compile(r'\d+')
        idx = regex.findall(os.path.split(img_adr)[1])[0]
        image = cv2.imread(self.root_dir + label +
                           '/pose_refinement/real/color' + str(idx) + ".png")
        rendered = cv2.imread(
            self.root_dir + label + '/pose_refinement/rendered/color' + str(idx) + ".png", cv2.IMREAD_GRAYSCALE)
        rendered = cv2.cvtColor(rendered.astype('uint8'), cv2.COLOR_GRAY2RGB)
        true_pose = get_rot_tra(self.root_dir + label + '/data/rot' + str(idx) + ".rot",
                                self.root_dir + label + '/data/tra' + str(idx) + ".tra")
        pred_pose_adr = self.root_dir + label + \
            '/predicted_pose/info_' + str(idx) + ".txt"
        pred_pose = np.loadtxt(pred_pose_adr)
        if self.transform:
            image = self.transform(image)
            rendered = self.transform(rendered)
        return label, image, rendered, true_pose, pred_pose
