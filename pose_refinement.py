import os
import cv2
import torch
import numpy as np
import unet_model as UNET
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.spatial.transform import Rotation as R
from dataset_classes import PoseRefinerDataset
from pose_refiner_architecture import Pose_Refiner


def fetch_ptcld_data(root_dir, label, bs):
    # detch pt cld data for batchsize
    pt_cld_data = []
    for i in range(bs):
        obj_dir = root_dir + label[i] + "/object.xyz"
        pt_cld = np.loadtxt(obj_dir, skiprows=1, usecols=(0, 1, 2))
        index = np.random.choice(pt_cld.shape[0], 3000, replace=False)
        pt_cld_data.append(pt_cld[index, :])
    pt_cld_data = np.stack(pt_cld_data, axis=0)
    return pt_cld_data


# no. of points is always 3000
def Matching_loss(pt_cld_rand, true_pose, pred_pose, bs, training=True):

    total_loss = torch.tensor([0.])
    total_loss.requires_grad = True
    for i in range(0, bs):
        pt_cld = pt_cld_rand[i, :, :].squeeze()
        TP = true_pose[i, :, :].squeeze()
        PP = pred_pose[i, :, :].squeeze()
        target = torch.tensor(pt_cld) @ TP[0:3, 0:3] + torch.cat(
            (TP[0, 3].view(-1, 1), TP[1, 3].view(-1, 1), TP[2, 3].view(-1, 1)), 1)
        output = torch.tensor(pt_cld) @ PP[0:3, 0:3] + torch.cat(
            (PP[0, 3].view(-1, 1), PP[1, 3].view(-1, 1), PP[2, 3].view(-1, 1)), 1)
        loss = (torch.abs(output - target).sum())/3000
        if loss < 100:
            total_loss = total_loss + loss
        else:  # so that loss isn't NaN
            total_loss = total_loss + torch.tensor([100.])

    return total_loss


def train_pose_refinement(root_dir, classes, epochs=5):

    train_data = PoseRefinerDataset(root_dir, classes=classes,
                                    transform=transforms.Compose([
                                        transforms.ToPILImage(mode=None),
                                        transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [
                                                             0.229, 0.224, 0.225]),
                                        transforms.ColorJitter(
                                            brightness=0, contrast=0, saturation=0, hue=0)
                                    ]))

    pose_refiner = Pose_Refiner()
    pose_refiner.cuda()
    # freeze resnet
    # pose_refiner.feature_extractor[0].weight.requires_grad = False

    batch_size = 4
    num_workers = 0
    valid_size = 0.2
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)

    optimizer = optim.Adam(pose_refiner.parameters(),
                           lr=3e-4, weight_decay=3e-5)

    # number of epochs to train the model
    n_epochs = epochs

    valid_loss_min = np.Inf  # track change in validation loss
    for epoch in range(1, n_epochs+1):

        print("----- Epoch Number: ", epoch, "--------")

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        pose_refiner.train()
        for label, image, rendered, true_pose, pred_pose in train_loader:
            # move tensors to GPU
            image, rendered = image.cuda(), rendered.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            xy, z, rot = pose_refiner(image, rendered, pred_pose, batch_size)
            # convert rot quarternion to rotational matrix
            rot[torch.isnan(rot)] = 1  # take care of NaN and inf values
            rot[rot == float("Inf")] = 1
            xy[torch.isnan(xy)] == 0
            z[torch.isnan(z)] == 0

            rot = torch.tensor(
                (R.from_quat(rot.detach().cpu().numpy())).as_matrix())
            # update predicted pose
            pred_pose[:, 0:3, 0:3] = rot
            pred_pose[:, 0, 3] = xy[:, 0]
            pred_pose[:, 1, 3] = xy[:, 1]
            pred_pose[:, 2, 3] = z.squeeze()
            # fetch point cloud data
            pt_cld = fetch_ptcld_data(root_dir, label, batch_size)
            # calculate the batch loss
            loss = Matching_loss(pt_cld, true_pose, pred_pose, batch_size)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()

        ######################
        # validate the model #
        ######################
        pose_refiner.eval()
        for label, image, rendered, true_pose, pred_pose in valid_loader:
            # move tensors to GPU
            image, rendered = image.cuda(), rendered.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            xy, z, rot = pose_refiner(image, rendered, pred_pose, batch_size)
            rot[torch.isnan(rot)] = 1  # take care of NaN and inf values
            rot[rot == float("Inf")] = 1            
            xy[torch.isnan(xy)] == 0
            z[torch.isnan(z)] == 0
            # convert R quarternion to rotational matrix
            rot = torch.tensor(
                (R.from_quat(rot.detach().cpu().numpy())).as_matrix())
            # update predicted pose
            pred_pose[:, 0:3, 0:3] = rot
            pred_pose[:, 0, 3] = xy[:, 0]
            pred_pose[:, 1, 3] = xy[:, 1]
            pred_pose[:, 2, 3] = z.squeeze()
            # fetch point cloud data
            pt_cld = fetch_ptcld_data(root_dir, label, batch_size)
            # calculate the batch loss
            loss = Matching_loss(pt_cld, true_pose, pred_pose, batch_size)
            # update average validation loss
            valid_loss += loss.item()

        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            torch.save(pose_refiner.state_dict(), 'pose_refiner.pt')
            valid_loss_min = valid_loss
