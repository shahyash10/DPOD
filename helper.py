import cv2
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Pickle functions to save and load dictionaries
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# helper function to plot grpahs


def visualize(array):
    "Plot all images in the array of tensors in one row"
    for z in range(0, len(array)):
        temp = array[z]
        if temp.ndim > 3:  # tensor output in the form NCHW
            temp = (torch.argmax(temp, dim=1).squeeze())
        if len(temp.shape) >= 3:
            plt.figure()
            plt.imshow(np.transpose(
                temp.detach().numpy().squeeze(), (1, 2, 0)))
            plt.show()
        else:
            plt.figure()
            plt.imshow(temp.detach().numpy(), cmap='gray')


def create_bounding_box(img, pose, pt_cld_data, intrinsic_matrix):
    "Create a bounding box around the object"
    # 8 corner points of the ptcld data
    min_x, min_y, min_z = pt_cld_data.min(axis=0)
    max_x, max_y, max_z = pt_cld_data.max(axis=0)
    corners_3D = np.array([[max_x, min_y, min_z],
                           [max_x, min_y, max_z],
                           [min_x, min_y, max_z],
                           [min_x, min_y, min_z],
                           [max_x, max_y, min_z],
                           [max_x, max_y, max_z],
                           [min_x, max_y, max_z],
                           [min_x, max_y, min_z]])

    # convert these 8 3D corners to 2D points
    ones = np.ones((corners_3D.shape[0], 1))
    homogenous_coordinate = np.append(corners_3D, ones, axis=1)

    # Perspective Projection to obtain 2D coordinates for masks
    homogenous_2D = intrinsic_matrix @ (pose @ homogenous_coordinate.T)
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)

    # Draw lines between these 8 points
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[4]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[2]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[5]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[3]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[6]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(coord_2D[3]), tuple(coord_2D[7]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[7]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[5]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(coord_2D[5]), tuple(coord_2D[6]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(coord_2D[6]), tuple(coord_2D[7]), (0, 0, 255), 3)

    return img


def ADD_score(pt_cld, true_pose, pred_pose, diameter):
    pred_pose[0:3, 0:3][np.isnan(pred_pose[0:3, 0:3])] = 1
    pred_pose[:, 3][np.isnan(pred_pose[:, 3])] = 0
    target = pt_cld @ true_pose[0:3, 0:3] + np.array(
        [true_pose[0, 3], true_pose[1, 3], true_pose[2, 3]])
    output = pt_cld @ pred_pose[0:3, 0:3] + np.array(
        [pred_pose[0, 3], pred_pose[1, 3], pred_pose[2, 3]])
    #avg_distance = (torch.abs(output - target)).sum()/pt_cld.shape[0]
    avg_distance = (np.linalg.norm(output - target))/pt_cld.shape[0]
    threshold = diameter * 0.1
    if avg_distance <= threshold:
        return 1
    else:
        #print("Avg distance vs threshold: ", avg_distance, threshold)
        return 0


def compute_add_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred
    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
        mean_distances[i] = np.mean(distance)

    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score
