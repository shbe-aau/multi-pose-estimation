import torch
import numpy as np
import math
import cv2
import csv
import matplotlib.pyplot as plt
import os

import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

from pytorch3d.renderer.utils import convert_to_tensors_and_broadcast

def look_at_rotation_fixed(camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device: str = "cpu") -> torch.Tensor:
    # Format input and broadcast
    broadcasted_args = convert_to_tensors_and_broadcast(
        camera_position, at, up, device=device
    )
    camera_position, at, up = broadcasted_args
    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        #replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        #x_axis = torch.where(is_close, replacement, x_axis)
        x_axis = F.normalize(torch.cross(up + 5e-3, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)

    if torch.norm(x_axis) <= 0 or torch.norm(y_axis) <= 0 or torch.norm(z_axis) <= 0:
        raise ValueError("look_at_rotation: x, y or z axis is zero!")
    return R.transpose(1, 2)



def hinter_sampling(min_n_pts, radius=1):
    '''
    Sphere sampling based on refining icosahedron as described in:
    Hinterstoisser et al., Simultaneous Recognition and Homography Extraction of
    Local Patches with a Simple Linear Classifier, BMVC 2008

    :param min_n_pts: Minimum required number of points on the whole view sphere.
    :param radius: Radius of the view sphere.
    :return: 3D points on the sphere surface and a list that indicates on which
             refinement level the points were created.
    '''

    # Get vertices and faces of icosahedron
    a, b, c = 0.0, 1.0, (1.0 + math.sqrt(5.0)) / 2.0
    pts = [(-b, c, a), (b, c, a), (-b, -c, a), (b, -c, a), (a, -b, c), (a, b, c),
           (a, -b, -c), (a, b, -c), (c, a, -b), (c, a, b), (-c, a, -b), (-c, a, b)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
             (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
             (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]

    # Refinement level on which the points were created
    pts_level = [0 for _ in range(len(pts))]

    ref_level = 0
    while len(pts) < min_n_pts:
        ref_level += 1
        edge_pt_map = {} # Mapping from an edge to a newly added point on that edge
        faces_new = [] # New set of faces

        # Each face is replaced by 4 new smaller faces
        for face in faces:
            pt_inds = list(face) # List of point IDs involved in the new faces
            for i in range(3):
                # Add a new point if this edge hasn't been processed yet,
                # or get ID of the already added point.
                edge = (face[i], face[(i + 1) % 3])
                edge = (min(edge), max(edge))
                if edge not in edge_pt_map.keys():
                    pt_new_id = len(pts)
                    edge_pt_map[edge] = pt_new_id
                    pt_inds.append(pt_new_id)

                    pt_new = 0.5 * (np.array(pts[edge[0]]) + np.array(pts[edge[1]]))
                    pts.append(pt_new.tolist())
                    pts_level.append(ref_level)
                else:
                    pt_inds.append(edge_pt_map[edge])

            # Replace the current face with 4 new faces
            faces_new += [(pt_inds[0], pt_inds[3], pt_inds[5]),
                          (pt_inds[3], pt_inds[1], pt_inds[4]),
                          (pt_inds[3], pt_inds[4], pt_inds[5]),
                          (pt_inds[5], pt_inds[4], pt_inds[2])]
        faces = faces_new

    # Project the points to a sphere
    pts = np.array(pts)
    pts *= np.reshape(radius / np.linalg.norm(pts, axis=1), (pts.shape[0], 1))

    # Collect point connections
    pt_conns = {}
    for face in faces:
        for i in range(len(face)):
            pt_conns.setdefault(face[i], set()).add(face[(i + 1) % len(face)])
            pt_conns[face[i]].add(face[(i + 2) % len(face)])

    # Order the points - starting from the top one and adding the connected points
    # sorted by azimuth
    top_pt_id = np.argmax(pts[:, 2])
    pts_ordered = []
    pts_todo = [top_pt_id]
    pts_done = [False for _ in range(pts.shape[0])]

    def calc_azimuth(x, y):
        two_pi = 2.0 * math.pi
        return (math.atan2(y, x) + two_pi) % two_pi

    while len(pts_ordered) != pts.shape[0]:
        # Sort by azimuth
        pts_todo = sorted(pts_todo, key=lambda i: calc_azimuth(pts[i][0], pts[i][1]))
        pts_todo_new = []
        for pt_id in pts_todo:
            pts_ordered.append(pt_id)
            pts_done[pt_id] = True
            pts_todo_new += [i for i in pt_conns[pt_id]] # Find the connected points

        # Points to be processed in the next iteration
        pts_todo = [i for i in set(pts_todo_new) if not pts_done[i]]

    # Re-order the points and faces
    pts = pts[np.array(pts_ordered), :]
    pts_level = [pts_level[i] for i in pts_ordered]
    pts_order = np.zeros((pts.shape[0],))
    pts_order[np.array(pts_ordered)] = np.arange(pts.shape[0])
    for face_id in range(len(faces)):
        faces[face_id] = [pts_order[i] for i in faces[face_id]]

    return pts, pts_level

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
        input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
        filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


# Create list of rotation matrices
# from list of euler angles ('xyz')
# i.e. [[x1, y1, z1],[x2, y2, z2]]
from scipy.spatial.transform import Rotation as R
def prepareViews(eulerList):
    views = []
    for e in eulerList:
        rot = R.from_euler('xyz', e, degrees=True)
        rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
        views.append(rot_mat)
    return views

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def append2file(data, file_name):
    with open(file_name, 'a') as f:
        wr = csv.writer(f, delimiter='\n')
        wr.writerow(data)

def prepareDir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def plotLoss(csv_name, file_name, validation_csv=None):
    with open(csv_name) as f:
        reader = csv.reader(f, delimiter='\n')
        loss = list(reader)
    if validation_csv:
        with open(validation_csv) as f:
            val_reader = csv.reader(f, delimiter='\n')
            val_loss = list(val_reader)
        val_loss = np.array(val_loss, dtype=np.float32).flatten()
        print(val_loss)
    loss = np.array(loss, dtype=np.float32).flatten()
    print(loss)

    fig = plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.plot(loss, label='train')
    if validation_csv:
        plt.plot(val_loss, label='validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    fig.tight_layout()
    fig.savefig(file_name, dpi=fig.dpi)
    plt.close()
    return val_loss # used in other stuff, don't want to load twice

def calcMeanVar(br, data, device, t):
    num_samples = len(data["codes"])
    data_indeces = np.arange(num_samples)
    np.random.shuffle(data_indeces)
    batch_size = br.batch_size

    all_data = []
    for i,curr_batch in enumerate(batch(data_indeces, batch_size)):
        # Render the ground truth images
        T = np.array(t, dtype=np.float32)
        Rs = []
        ts = []
        for b in curr_batch:
            Rs.append(data["Rs"][b])
            ts.append(T.copy())
        gt_images = br.renderBatch(Rs, ts)
        all_data.append(gt_images)
        print("Step: {0}/{1}".format(i,round(num_samples/batch_size)))
    result = torch.cat(all_data)
    print(torch.mean(result))
    print(torch.std(result))
    return torch.mean(result), torch.std(result)

def plotView(currView, numViews, vmin, vmax, input_images, groundtruth, predicted, predicted_pose, loss, batch_size, threshold=9999, img_num=0):
    # Plot AE input
    plt.subplot(1, 4, 1)
    plt.imshow((input_images[img_num]*255).astype(np.uint8))
    plt.title("Input to AE")

    # Plot depth map render from ground truth
    plt.subplot(1, 4, 2)
    plt.imshow(groundtruth[img_num].detach().cpu().numpy())#,
               #vmin=vmin, vmax=vmax)
    plt.title("Depth Render - GT")

    # Plot depth map render from prediction
    plt.subplot(1, 4, 3)
    plt.imshow(predicted[img_num].detach().cpu().numpy())#,
               #vmin=vmin, vmax=vmax)

    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=30)
    plt.title("Predicted: \n " + np.array2string((predicted_pose[img_num][:numViews]).detach().cpu().numpy(),precision=2))

    # if(currView == 0):
    #     plt.title("Predicted: \n " + np.array2string((predicted_pose[currView*batch_size]).detach().cpu().numpy(),precision=2))
    # else:
    #     plt.title("Predicted")

    # Plot difference between depth maps
    loss_contrib = np.abs((groundtruth[img_num]).detach().cpu().numpy() - (predicted[img_num]).detach().cpu().numpy())
    loss_contrib[loss_contrib > threshold] = threshold
    plt.subplot(1, 4, 4)
    plt.imshow(loss_contrib)#, vmin=0.0, vmax=20.0)
    plt.title("Loss: \n " + np.array2string((loss[img_num]).detach().cpu().numpy()))

# Convert quaternion to rotation matrix
# from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    #norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    #norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    #w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    norm_quat = quat/quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

# -*- coding: utf-8 -*-
# transform.py

# Copyright (c) 2006-2015, Christoph Gohlke
# Copyright (c) 2006-2015, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

def random_quaternion(rand=None):
    """Return uniform random unit quaternion.

    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.

    >>> q = random_quaternion()
    >>> numpy.allclose(1, vector_norm(q))
    True
    >>> q = random_quaternion(numpy.random.random(3))
    >>> len(q.shape), q.shape[0]==4
    (1, True)

    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array([np.cos(t2)*r2, np.sin(t1)*r1,
                        np.cos(t1)*r1, np.sin(t2)*r2])



def q2m(quat):
    _EPS = np.finfo(float).eps * 4.0
    q = torch.tensor(quat, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def random_rotation_matrix(rand=None):
    """Return uniform random rotation matrix.

    rand: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.

    >>> R = random_rotation_matrix()
    >>> np.allclose(np.dot(R.T, R), np.identity(4))
    True

    """
    return quaternion_matrix(random_quaternion(rand))


def calc_2d_bbox(xs, ys, im_size):
    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))
    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]


def extract_square_patch(scene_img, bb_xywh, pad_factor=1.2,resize=(128,128),
                         interpolation=cv2.INTER_NEAREST,black_borders=False):

        x, y, w, h = np.array(bb_xywh).astype(np.int32)
        size = int(np.maximum(h, w) * pad_factor)

        left = int(np.maximum(x+w/2-size/2, 0))
        right = int(np.minimum(x+w/2+size/2, scene_img.shape[1]))
        top = int(np.maximum(y+h/2-size/2, 0))
        bottom = int(np.minimum(y+h/2+size/2, scene_img.shape[0]))

        scene_crop = scene_img[top:bottom, left:right].copy()

        if black_borders:
            scene_crop[:(y-top),:] = 0
            scene_crop[(y+h-top):,:] = 0
            scene_crop[:,:(x-left)] = 0
            scene_crop[:,(x+w-left):] = 0

        scene_crop = cv2.resize(scene_crop, resize, interpolation = interpolation)
        return scene_crop
