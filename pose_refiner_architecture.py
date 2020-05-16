""" Parts of the Deep Learning Based pose refiner model """
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from scipy.spatial.transform import Rotation as R


class Pose_Refiner(nn.Module):

    def __init__(self):
        super(Pose_Refiner, self).__init__()
        self.feature_extractor_image = nn.Sequential(*list(models.resnet18(pretrained=True,
                                                                           progress=True).children())[:9])
        self.feature_extractor_rendered = nn.Sequential(*list(models.resnet18(pretrained=True,
                                                                              progress=True).children())[:9])
        self.fc_xyhead_1 = nn.Linear(512, 253)
        self.fc_xyhead_2 = nn.Linear(256, 2)
        self.fc_zhead = nn.Sequential(nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 1))
        self.fc_Rhead_1 = nn.Linear(512, 252)
        self.fc_Rhead_2 = nn.Linear(256, 4)

        self.relu_layer = nn.ReLU()

    def _initialize_weights(self):
        # weight initialization
        nn.init.constant_(self.fc_xyhead_1.weight, 0.)
        nn.init.constant_(self.fc_xyhead_1.bias, 0.)

        weights = torch.zeros((2, 256))
        weights[0, 253] = torch.tensor(1.)
        weights[1, 254] = torch.tensor(1.)
        self.fc_xyhead_2.weight = nn.Parameter(weights)
        nn.init.constant_(self.fc_xyhead_2.bias, 0.)

        nn.init.constant_(self.fc_zhead.weight, 0.)
        nn.init.constant_(self.fc_zhead.bias, 0.)

        nn.init.constant_(self.fc_Rhead_1.weight, 0.)
        nn.init.constant_(self.fc_Rhead_1.bias, 0.)

        rand_weights = torch.zeros((4, 256))
        rand_weights[0, 252] = torch.tensor(1.)
        rand_weights[1, 253] = torch.tensor(1.)
        rand_weights[2, 254] = torch.tensor(1.)
        rand_weights[3, 255] = torch.tensor(1.)
        self.fc_Rhead_2.weight = nn.Parameter(weights)
        nn.init.constant_(self.fc_Rhead_2.bias, 0.)

    def forward(self, image, rendered, pred_pose, bs=1):
        # extracting the feature vector f
        f_image = self.feature_extractor_image(image)
        f_rendered = self.feature_extractor_rendered(rendered)
        f_image = f_image.view(bs, -1)
        f_image = self.relu_layer(f_image)
        f_rendered = f_image.view(bs, -1)
        f_rendered = self.relu_layer(f_rendered)
        f = f_image - f_rendered

        # Z refinement head
        z = self.fc_zhead(f)

        # XY refinement head
        f_xy1 = self.fc_xyhead_1(f)
        f_xy1 = self.relu_layer(f_xy1)
        x_pred = np.reshape(pred_pose[:, 0, 3], (bs, -1))
        y_pred = np.reshape(pred_pose[:, 1, 3], (bs, -1))
        f_xy1 = torch.cat((f_xy1, x_pred.float().cuda()), 1)
        f_xy1 = torch.cat((f_xy1, y_pred.float().cuda()), 1)
        f_xy1 = torch.cat((f_xy1, z), 1)
        xy = self.fc_xyhead_2(f_xy1.cuda())

        # Rotation head
        f_r1 = self.fc_Rhead_1(f)
        f_r1 = self.relu_layer(f_r1)
        r = R.from_matrix(pred_pose[:, 0:3, 0:3])
        r = r.as_quat()
        r = np.reshape(r, (bs, -1))
        f_r1 = torch.cat(
            (f_r1, torch.from_numpy(r).float().cuda()), 1)
        rot = self.fc_Rhead_2(f_r1)

        return xy, z, rot
