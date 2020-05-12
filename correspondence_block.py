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
from helper import load_obj, visualize
from dataset_classes import OcclusionDataset
%load_ext autoreload
%autoreload 2

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

classes = ['Ape','Can','Cat','Driller','Duck','Eggbox','Glue','Holepuncher','Background']

root_dir = "/home/jovyan/work/OcclusionChallengeICCV2015"
train_data = OcclusionDataset(
    root_dir, 
    classes = classes, 
    transform = transforms.Compose([transforms.ToTensor()])
)

criterion_id = nn.CrossEntropyLoss()
criterion_u = nn.CrossEntropyLoss()
criterion_v = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(correspondence_block.parameters(), lr=3e-4,weight_decay=3e-5)

# Training Loop
# number of epochs to train the model
n_epochs = 20
valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    correspondence_block.train()
    for image, idmask,umask,vmask in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            image, idmask,umask,vmask = image.cuda(), idmask.cuda(), umask.cuda(), vmask.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        idmask_pred,umask_pred,vmask_pred = correspondence_block(image)       
        # calculate the batch loss
        loss_id = criterion_id(idmask_pred, idmask)
        loss_u = criterion_u(umask_pred, umask)
        loss_v = criterion_v(vmask_pred, vmask)
        loss = loss_id + loss_u + loss_v
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()


    ######################    
    # validate the model #
    ######################
    correspondence_block.eval()
    for image, idmask,umask,vmask in valid_loader:       
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            image, idmask,umask,vmask = image.cuda(), idmask.cuda(), umask.cuda(), vmask.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        idmask_pred,umask_pred,vmask_pred = correspondence_block(image)
        # calculate the batch loss
        loss_id = criterion_id(idmask_pred, idmask)
        loss_u = criterion_u(umask_pred, umask)
        loss_v = criterion_v(vmask_pred, vmask)
        loss = loss_id + loss_u + loss_v
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
        valid_loss_min,
        valid_loss))
        torch.save(correspondence_block.state_dict(), 'correspondence_block.pt')
        valid_loss_min = valid_loss
