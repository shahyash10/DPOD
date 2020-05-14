""" Parts of the Deep Learning Based pose refiner model """
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy.spatial.transform import Rotation as R


class Pose_Refiner(nn.Module):

    def __init__(self):
        super(Pose_Refiner, self).__init__()
        self.feature_extractor = nn.Sequential(*list(models.resnet18(pretrained=True,
                                                                     progress=True).children())[:9])
        self.fc_xyhead_1 = nn.Linear(512, 253)
        self.fc_xyhead_2 = nn.Linear(256, 2)
        self.fc_zhead = nn.Sequential(nn.Linear(512, 256),
                                      nn.Linear(256, 1))
        self.fc_Rhead_1 = nn.Linear(512, 252)
        self.fc_Rhead_2 = nn.Linear(256, 4)

    def forward(self, image, idmask, pred_pose):
        # extracting the feature vector f
        f_image = torch.flatten(self.feature_extractor(image))
        f_idmask = torch.flatten(self.feature_extractor(idmask))
        f = f_image - f_idmask

        # Z refinement head
        z = self.fc_zhead(f)

        # XY refinement head
        f_xy1 = self.fc_xyhead_1(f)
        x_pred = pred_pose[0, 3]
        y_pred = pred_pose[1, 3]
        f_xy1 = torch.cat((f_xy1, torch.tensor([x_pred]).float().cuda()))
        f_xy1 = torch.cat((f_xy1, torch.tensor([y_pred]).float().cuda()))
        f_xy1 = torch.cat((f_xy1, z))
        xy = self.fc_xyhead_2(f_xy1.cuda())

        # Rotation head
        f_r1 = self.fc_Rhead_1(f)
        r = R.from_matrix(pred_pose[0:3, 0:3])
        r = r.as_quat()
        f_r1 = torch.cat((f_r1, torch.tensor([r]).float().squeeze().cuda()))
        rot = self.fc_Rhead_2(f_r1)

        return xy, z, rot
