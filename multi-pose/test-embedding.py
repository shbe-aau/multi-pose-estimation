import os
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import argparse
import glob
import hashlib
import cv2 as cv

import time
import pickle
from utils.utils import *
from utils.tools import *

import imgaug as ia
import imgaug.augmenters as iaa

from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as scipyR

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes, Textures, list_to_padded

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    HardPhongShader, PointLights, DirectionalLights
)

from utils.pytless import inout, misc
from vispy import app, gloo
from utils.pytless.renderer import Renderer

from utils.sundermeyer.pysixd import view_sampler

from Encoder import Encoder
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

visualize = False
batch_size = 10
iterations = 500

def main():
    device = torch.device("cuda:0")


    datagen = DatasetGenerator("./data/VOC2012/JPEGImages/",
                               ["./data/cad-files/ply-files/obj_10.ply"],
                               [[0.0, 0.0, 375.0]],
                               batch_size,
                               "not_used",
                               device,
                               "sundermeyer-random",
                               random_light=True,
                               num_bgs=200)
    datagen.max_rel_offset = 0.0
    datagen.pad_factor = 3.0
    datagen.img_size = 256
    datagen.backgrounds = datagen.load_bg_images("backgrounds",
                                                 "./data/VOC2012/JPEGImages/",
                                                 200, 256, 256)
    # create output directory
    if not os.path.exists('embedding-output/'):
        os.makedirs('embedding-output/')

    # load model
    encoder = Encoder("./data/obj1-18/encoder.npy").to(device)
    encoder.eval()

    shifts = [(0.0,0.0), #0
              (-1.0,-1.0), #1
              (0.0,-1.0), #2
              (1.0,-1.0), #3
              (1.0,0.0), #4
              (1.0,1.0), #5
              (0.0,1.0), #6
              (-1.0,1.0), #7
              (-1.0,0.0)] #8

    Rin = torch.from_numpy(np.vstack([np.eye(3)]*batch_size))
    tin = torch.from_numpy(np.array([0.0,0.0,375.0]))

    print(Rin.shape)

    labels = []
    codes = []
    #np.random.seed(seed=10)
    data = datagen.generate_image_batch(augment=True)
    curr_img = 0
    for i in range(iterations):
        org_img = data["images"][curr_img]

        curr_img += 1
        if(curr_img == batch_size):
            #np.random.seed(seed=10)
            data = datagen.generate_image_batch(augment=True)
            curr_img = 0

        # Sample the same random shift
        obj_bb = 50,50,156,156
        rand_trans_x = np.random.uniform(0.25, 0.5) * 156
        rand_trans_y = np.random.uniform(0.25, 0.5) * 156

        for k,s in enumerate(shifts):
            print("img{0}-shift{1}.png".format(i,k))
            # Apply shift
            obj_bb_off = obj_bb + np.array([s[0]*rand_trans_x,
                                            s[1]*rand_trans_y,
                                            0,0])

            cropped = extract_square_patch(org_img, obj_bb_off,
                                           pad_factor=1.2,
                                           resize=(128,128))

            # Normalize image
            img = cropped
            img_max = np.max(img)
            img_min = np.min(img)
            img = (img - img_min)/(img_max - img_min)

            # Run image through encoder
            img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).to(device)
            code = encoder(img.float())
            code = code.detach().cpu().numpy()[0]
            norm_code = code / np.linalg.norm(code)

            # Add labels
            codes.append(norm_code)
            labels.append(k)
            #labels.append(curr_img)

            if(visualize):
                fig = plt.figure()
                plt.imshow(cropped)
                fig.savefig("embedding-output/test{0}-shift{1}.png".format(i,k), dpi=fig.dpi)
                plt.close()
            #break

    # Apply t-SNE
    X = np.array(codes)
    X_embedded = TSNE(n_components=2, n_jobs=-1,
                      learning_rate=200.0,
                      perplexity=30.0).fit_transform(X)
    print(X_embedded.shape)

    fig = plt.figure()
    plt.scatter(X[:,0], X[:,1], c=np.array(labels), cmap="tab10")
    fig.savefig("embedding-output/embedding-{0}samples.png".format(iterations), dpi=fig.dpi)
    plt.close()

class DatasetGenerator():
    def __init__(self, background_path, obj_paths, obj_distance, batch_size,
                 _, device, sampling_method="sphere", random_light=True,
                 num_bgs=5000, seed=None):

        self.random_light = random_light
        self.realistic_occlusions = False
        self.random_renders = []
        self.curr_samples = 0
        self.max_samples = 1000
        self.device = device
        self.poses = []
        self.obj_paths = obj_paths
        self.batch_size = batch_size
        self.dist = obj_distance
        self.img_size = 128
        self.render_size = 3*self.img_size
        self.max_rel_offset = 0.2
        self.max_rel_scale = None
        self.pad_factor =  1.2
        self.K = np.array([1075.65, 0, self.render_size/2,
                           0, 1073.90, self.render_size/2,
                           0, 0, 1]).reshape(3,3)

        self.aug = self.setup_augmentation()

        self.backgrounds = self.load_bg_images("backgrounds", background_path, num_bgs,
                                               self.img_size, self.img_size)

        # Stuff for viewsphere aug sampling
        self.view_sphere = None
        self.view_sphere_indices = []
        self.random_aug = None

        # Prepare renders for each object
        self.renderers = []
        for o in obj_paths:
            if('.ply' not in o):
                print("Error! {0} is not a .ply file!".format(o))
                return None
            curr_model = inout.load_ply(o)
            curr_rend= Renderer(curr_model, (self.render_size,self.render_size),
                                self.K, surf_color=(1, 1, 1), mode='rgb')
            self.renderers.append(curr_rend)

        self.pose_reuse = False
        if(sampling_method.split("-")[-1] == "reuse"):
            self.pose_reuse = True
        sampling_method = sampling_method.replace("-reuse","")

        self.hard_samples = []
        self.hard_mining = False
        if(sampling_method.split("-")[-1] == "hard"):
            self.hard_mining = True
        sampling_method = sampling_method.replace("-hard","")
        self.hard_sample_ratio = 0.2
        self.hard_mining_ratio = 0.3

        self.simple_pose_sampling = False
        if(sampling_method == "tless"):
            self.pose_sampling = self.tless_sampling
            self.simple_pose_sampling = False
        elif(sampling_method == "tless-simple"):
            self.pose_sampling = self.tless_sampling
            self.simple_pose_sampling = True
        elif(sampling_method == "sphere"):
            self.pose_sampling = self.sphere_sampling
            self.simple_pose_sampling = False
        elif(sampling_method == "sphere-simple"):
            self.pose_sampling = self.sphere_sampling
            self.simple_pose_sampling = True
        elif(sampling_method == "sphere-fixed"):
            self.pose_sampling = self.sphere_sampling_fixed
            self.simple_pose_sampling = False
        elif(sampling_method == "sphere-wolfram"):
            self.pose_sampling = self.sphere_wolfram_sampling
            self.simple_pose_sampling = False
        elif(sampling_method == "sphere-wolfram-fixed"):
            self.pose_sampling = self.sphere_wolfram_sampling_fixed
            self.simple_pose_sampling = False
        elif(sampling_method == "fixed"): # Mainly for debugging purposes
            self.pose_sampling = self.fixed_sampling
            self.simple_pose_sampling = False
        elif(sampling_method == "viewsphere"):
            self.pose_sampling = self.viewsphere_sampling
            self.simple_pose_sampling = False
        elif(sampling_method == "sundermeyer-random"):
            self.pose_sampling = self.sm_quat_random
            self.simple_pose_sampling = False
        elif(sampling_method == "viewsphere-aug"):
            self.pose_sampling = self.viewsphere_aug
            self.simple_pose_sampling = False
        elif(sampling_method == "viewsphere-aug-no-conv"):
            self.pose_sampling = self.viewsphere_aug_no_conv
            self.simple_pose_sampling = False
        elif(sampling_method == "mixed"):
            self.pose_sampling = self.mixed
            self.simple_pose_sampling = False
        elif(sampling_method == "quat"):
            self.pose_sampling = self.quat_sampling
            self.simple_pose_sampling = False
        elif(".p" in sampling_method):
            self.pose_sampling = self.pickle_sampling
            self.pose_path = sampling_method
            self.poses = []
            self.simple_pose_sampling = False
        else:
            print("ERROR! Invalid view sampling method: {0}".format(sampling_method))

        if(self.pose_reuse == True):
            self.poses = []
            for i in np.arange(20*1000):
                R, t = self.pose_sampling()
                self.poses.append(R)
                #print("generated random pose: ", len(self.poses))
            self.pose_sampling = self.reuse_poses

    def reuse_poses(self):
        rand_id = np.random.choice(20*1000,1,replace=False)[0]
        #print("re-using pose: ", rand_id)
        R = self.poses[rand_id]
        return R


    def viewsphere_for_embedding(self, num_views, num_inplane):
        azimuth_range = (0, 2 * np.pi)
        elev_range = (-0.5 * np.pi, 0.5 * np.pi)
        views, _ = view_sampler.sample_views(
            num_views,
            1000.0,
            azimuth_range,
            elev_range
        )

        Rs = np.empty( (len(views)*num_inplane, 3, 3) )
        i = 0
        for view in views:
            for cyclo in np.linspace(0, 2.*np.pi, num_inplane):
                rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
                Rs[i,:,:] = rot_z.dot(view['R'])
                i += 1
        return Rs

    def load_bg_images(self, output_path, background_path, num_bg_images, h, w, c=3):
        if(background_path == ""):
            return []

        bg_img_paths = glob.glob(background_path + "*.jpg")
        noof_bg_imgs = min(num_bg_images, len(bg_img_paths))
        shape = (h, w, c)
        bg_imgs = np.empty( (noof_bg_imgs,) + shape, dtype=np.uint8 )

        current_config_hash = hashlib.md5((str(shape) + str(noof_bg_imgs) + str(background_path)).encode('utf-8')).hexdigest()
        current_file_name = os.path.join(output_path + '-' + current_config_hash +'.npy')
        if os.path.exists(current_file_name):
            bg_imgs = np.load(current_file_name)
        else:
            file_list = bg_img_paths[:noof_bg_imgs]
            print(len(file_list))
            from random import shuffle
            shuffle(file_list)

            for j,fname in enumerate(file_list):
                print('loading bg img %s/%s' % (j,noof_bg_imgs))
                bgr = cv2.imread(fname)
                H,W = bgr.shape[:2]
                y_anchor = int(np.random.rand() * (H-shape[0]))
                x_anchor = int(np.random.rand() * (W-shape[1]))
                # bgr = cv2.resize(bgr, shape[:2])
                bgr = bgr[y_anchor:y_anchor+shape[0],x_anchor:x_anchor+shape[1],:]
                if bgr.shape[0]!=shape[0] or bgr.shape[1]!=shape[1]:
                    continue
                if shape[2] == 1:
                    bgr = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
                bg_imgs[j] = bgr
            np.save(current_file_name,bg_imgs)
            print('loaded %s bg images' % noof_bg_imgs)
        return bg_imgs

    def setup_augmentation(self):
        # Augmentation
        # aug = iaa.Sequential([
        #     #iaa.Sometimes(0.5, iaa.PerspectiveTransform(0.05)),
        #     #iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.05, 0.1))),
        #     #iaa.Sometimes(0.5, iaa.Affine(scale=(1.0, 1.2))),
        #     iaa.Sometimes(0.5, iaa.CoarseDropout( p=0.05, size_percent=0.01) ),F
        #     iaa.Sometimes(0.5, iaa.GaussianBlur(1.2*np.random.rand())),
        #     iaa.Sometimes(0.5, iaa.Add((-0.1, 0.1), per_channel=0.3)),
        #     iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
        #     iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
        #     iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
        #     iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3))],
        #                      random_order=False)
        # aug = iaa.Sequential([
        #     #iaa.Sometimes(0.5, iaa.CoarseDropout( p=0.25, size_percent=0.02) ),
        #     iaa.Sometimes(0.5, iaa.GaussianBlur(1.2*np.random.rand())),
        #     iaa.Sometimes(0.5, iaa.Add((-60, 60), per_channel=0.3)),
        #     iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
        #     iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
        #     iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3))],
        #                      random_order=False)


        aug = iaa.Sequential([
            #iaa.Sometimes(0.5, PerspectiveTransform(0.05)),
            #iaa.Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
            #iaa.Sometimes(0.5, iaa.Affine(scale=(1.0, 1.2))),
            #iaa.Sometimes(0.5, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
            iaa.Sometimes(0.5,
                iaa.SomeOf(2, [ iaa.CoarseDropout( p=0.2, size_percent=0.05),
                                iaa.Cutout(fill_mode="constant", cval=(0, 255),
                 fill_per_channel=0.5),
                                iaa.Cutout(fill_mode="constant", cval=(255)),
                                iaa.CoarseSaltAndPepper(0.05, size_px=(4, 16)),
                                iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1))
                                ])),
            iaa.Sometimes(0.5, iaa.GaussianBlur(1.2*np.random.rand())),
            iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3)),
            iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
            iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3))],
                             random_order=False)
        return aug

    # Randomly sample poses from .p file (pickle)
    def pickle_sampling(self):
        # Load poses from .p file once
        if(len(self.poses) == 0):
            with open(self.pose_path, "rb") as f:
                self.poses = pickle.load(f, encoding="latin1")["Rs"]
                print("Read pickle: ", len(self.poses))

        # Sample pose randomly
        #random.shuffle(self.poses)
        index = np.random.randint(0,len(self.poses))
        R = torch.tensor(self.poses[index], dtype=torch.float32)
        return R

    def quat_random(self):
        # Sample random quaternion
        rand = np.random.rand(3)
        r1 = np.sqrt(1.0 - rand[0])
        r2 = np.sqrt(rand[0])
        pi2 = math.pi * 2.0
        t1 = pi2 * rand[1]
        t2 = pi2 * rand[2]
        random_quat = np.array([np.cos(t2)*r2, np.sin(t1)*r1,
                                np.cos(t1)*r1, np.sin(t2)*r2])

        # Convert quaternion to rotation matrix
        q = np.array(random_quat, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < 0.0001: #_EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        R = np.array([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
            [                0.0,                 0.0,                 0.0, 1.0]])
        R = R[:3,:3]
        return R

    # Randomly sample poses from SM view sphere
    def viewsphere_aug(self):
        if(self.view_sphere is None):
            self.view_sphere = self.viewsphere_for_embedding(600, 36)

        if(len(self.view_sphere_indices) == 0):
            self.view_sphere_indices = list(np.random.choice(self.view_sphere.shape[0],
                                                             self.max_samples, replace=False))
            # Sample new rotation aug for each new list!
            self.random_aug = self.quat_random()

        # Pop random index and associated R matrix
        rand_i = self.view_sphere_indices.pop()
        curr_R = self.view_sphere[rand_i]

        # Apply random augmentation R matrix
        aug_R = np.dot(curr_R, self.random_aug)

        # Convert R matrix from opengl to pytorch format
        xy_flip = np.eye(3, dtype=np.float)
        xy_flip[0,0] = -1.0
        xy_flip[1,1] = -1.0
        R_conv = np.transpose(aug_R)
        R_conv = np.dot(R_conv,xy_flip)

        # Convert to tensors
        R = torch.from_numpy(R_conv)
        return R

    # Randomly sample poses from SM view sphere
    def viewsphere_aug_no_conv(self):
        if(self.view_sphere is None):
            self.view_sphere = self.viewsphere_for_embedding(600, 18)

        if(len(self.view_sphere_indices) == 0):
            self.view_sphere_indices = list(np.random.choice(self.view_sphere.shape[0],
                                                             self.max_samples, replace=False))
            # Sample new rotation aug for each new list!
            self.random_aug = self.quat_random()

        # Pop random index and associated R matrix
        rand_i = self.view_sphere_indices.pop()
        curr_R = self.view_sphere[rand_i]

        # Apply random augmentation R matrix
        aug_R = np.dot(curr_R, self.random_aug)

        # Convert to tensors
        R = torch.from_numpy(aug_R)
        return R


    # Randomly sample poses from SM view sphere
    def viewsphere_sampling(self):
        # Load poses from .p file
        pose_path = './data/view-sphere.p'
        if(len(self.poses) == 0):
            with open(pose_path, "rb") as f:
                self.poses = pickle.load(f, encoding="latin1")["Rs"]
                print("Read pickle: ", len(self.poses))
                np.random.shuffle(self.poses)

        # Sample pose randomly
        R = torch.tensor(self.poses.pop(-1), dtype=torch.float32)
        return R

    def tless_sampling(self):
        theta_sample = np.random.uniform(low=0.0, high=2.0*np.pi, size=1)[0]
        phi_sample = np.random.uniform(low=0.0, high=2.0*np.pi, size=1)[0]

        x = np.sin(theta_sample)*np.cos(phi_sample)
        y = np.sin(theta_sample)*np.sin(phi_sample)
        z = np.cos(theta_sample)

        cam_position = torch.tensor([float(x), float(y), float(z)]).unsqueeze(0)
        if(z < 0):
            R = look_at_rotation(cam_position, up=((0, 0, -1),)).squeeze()
        else:
            R = look_at_rotation(cam_position, up=((0, 0, 1),)).squeeze()

        # Rotate in-plane
        if(not self.simple_pose_sampling):
            rot_degrees = np.random.uniform(low=-90.0, high=90.0, size=1)
            rot = scipyR.from_euler('z', rot_degrees, degrees=True)
            rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
            R = torch.matmul(R, rot_mat)
            R = R.squeeze()

        return R


    # Based on Sundermeyer
    def mixed(self):
        rand = np.random.uniform(low=-100, high=100, size=1)[0]
        if(rand > 0):
            return self.sphere_wolfram_sampling_fixed()
        return self.sm_quat_random()

    # Based on Sundermeyer
    def sm_quat_random(self):
        # Sample random quaternion
        rand = np.random.rand(3)
        r1 = np.sqrt(1.0 - rand[0])
        r2 = np.sqrt(rand[0])
        pi2 = math.pi * 2.0
        t1 = pi2 * rand[1]
        t2 = pi2 * rand[2]
        random_quat = np.array([np.cos(t2)*r2, np.sin(t1)*r1,
                                np.cos(t1)*r1, np.sin(t2)*r2])

        # Convert quaternion to rotation matrix
        q = np.array(random_quat, dtype=np.float64, copy=True)
        n = np.dot(q, q)

        if n < np.finfo(float).eps * 4.0:
            R = np.identity(4)
        else:
            q *= math.sqrt(2.0 / n)
            q = np.outer(q, q)
            R = np.array([
                [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
                [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
                [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
                [                0.0,                 0.0,                 0.0, 1.0]])
        R = R[:3,:3]

        # Convert R matrix from opengl to pytorch format
        xy_flip = np.eye(3, dtype=np.float)
        xy_flip[0,0] = -1.0
        xy_flip[1,1] = -1.0
        R_conv = np.transpose(R)
        R_conv = np.dot(R_conv,xy_flip)

        # Convert to tensors
        R = torch.from_numpy(R_conv)
        return R

    def quat_sampling(self):
        R = get_sampled_rotation_matrices_by_quat(1).squeeze()
        return R

    # Truely random
    # Based on: https://mathworld.wolfram.com/SpherePointPicking.html
    def sphere_wolfram_sampling(self):
        x1 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
        x2 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
        test = x1**2 + x2**2

        while(test >= 1.0):
                x1 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
                x2 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
                test = x1**2 + x2**2

        x = 2.0*x1*(1.0 -x1**2 - x2**2)**(0.5)
        y = 2.0*x2*(1.0 -x1**2 - x2**2)**(0.5)
        z = 1.0 - 2.0*(x1**2 + x2**2)

        cam_position = torch.tensor([x, y, z]).unsqueeze(0)
        if(z < 0):
            R = look_at_rotation(cam_position, up=((0, 0, -1),)).squeeze()
        else:
            R = look_at_rotation(cam_position, up=((0, 0, 1),)).squeeze()

        # Rotate in-plane
        if(not self.simple_pose_sampling):
            rot_degrees = np.random.uniform(low=-90.0, high=90.0, size=1)
            rot = scipyR.from_euler('z', rot_degrees, degrees=True)
            rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
            R = torch.matmul(R, rot_mat)
            R = R.squeeze()
        return R

    # Truely random
    # Based on: https://mathworld.wolfram.com/SpherePointPicking.html
    def sphere_wolfram_sampling_fixed(self):
        x1 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
        x2 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
        test = x1**2 + x2**2

        while(test >= 1.0):
                x1 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
                x2 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
                test = x1**2 + x2**2

        x = 2.0*x1*(1.0 -x1**2 - x2**2)**(0.5)
        y = 2.0*x2*(1.0 -x1**2 - x2**2)**(0.5)
        z = 1.0 - 2.0*(x1**2 + x2**2)

        cam_position = torch.tensor([x, y, z]).unsqueeze(0)
        if(z < 0):
            R = look_at_rotation_fixed(cam_position, up=((0, 0, -1),)).squeeze()
        else:
            R = look_at_rotation_fixed(cam_position, up=((0, 0, 1),)).squeeze()

        # Rotate in-plane
        if(not self.simple_pose_sampling):
            rot_degrees = np.random.uniform(low=-90.0, high=90.0, size=1)
            rot = scipyR.from_euler('z', rot_degrees, degrees=True)
            rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
            R = torch.matmul(R, rot_mat)
            R = R.squeeze()

        return R


    # Truely random
    # Based on: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    def sphere_sampling(self):
        z = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
        theta_sample = np.random.uniform(low=0.0, high=2.0*np.pi, size=1)[0]
        x = np.sqrt((1.0**2 - z**2))*np.cos(theta_sample)
        y = np.sqrt((1.0**2 - z**2))*np.sin(theta_sample)

        cam_position = torch.tensor([x, y, z]).unsqueeze(0)
        if(z < 0):
            R = look_at_rotation(cam_position, up=((0, 0, -1),)).squeeze()
        else:
            R = look_at_rotation(cam_position, up=((0, 0, 1),)).squeeze()

        # Rotate in-plane
        if(not self.simple_pose_sampling):
            rot_degrees = np.random.uniform(low=-90.0, high=90.0, size=1)
            rot = scipyR.from_euler('z', rot_degrees, degrees=True)
            rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
            R = torch.matmul(R, rot_mat)
            R = R.squeeze()
        return R

    # Truely random
    # Based on: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    def sphere_sampling_fixed(self):
        z = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
        theta_sample = np.random.uniform(low=0.0, high=2.0*np.pi, size=1)[0]
        x = np.sqrt((1.0**2 - z**2))*np.cos(theta_sample)
        y = np.sqrt((1.0**2 - z**2))*np.sin(theta_sample)

        cam_position = torch.tensor([x, y, z]).unsqueeze(0)
        if(z < 0):
            R = look_at_rotation_fixed(cam_position, up=((0, 0, -1),)).squeeze()
        else:
            R = look_at_rotation_fixed(cam_position, up=((0, 0, 1),)).squeeze()

        # Rotate in-plane
        if(not self.simple_pose_sampling):
            rot_degrees = np.random.uniform(low=-90.0, high=90.0, size=1)
            rot = scipyR.from_euler('z', rot_degrees, degrees=True)
            rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
            R = torch.matmul(R, rot_mat)
            R = R.squeeze()
        return R

    def generate_random_renders(self,num):
        image_renders = []
        images = []

        for k in np.arange(num):
            print("Rendering random objects: ", k)
            R, t = self.pose_sampling()

            R = R.detach().cpu().numpy()
            t = t.detach().cpu().numpy()

            # Convert R matrix from pytorch to opengl format
            # for rendering only!
            xy_flip = np.eye(3, dtype=np.float)
            xy_flip[0,0] = -1.0
            xy_flip[1,1] = -1.0
            R_opengl = np.dot(R,xy_flip)
            R_opengl = np.transpose(R_opengl)

            # Render images
            random_id = np.random.randint(1,31)
            obj_path = "./data/tless-obj{0:02d}/cad/obj_{1:02d}.obj".format(random_id, random_id)
            model = inout.load_ply(obj_path.replace(".obj",".ply"))

            # Normalize pts
            verts = model['pts']
            center = np.mean(verts, axis=0)
            verts_normed = verts - center
            scale = np.max(np.max(np.abs(verts_normed), axis=0))
            verts_normed = (verts_normed / scale)
            model['pts'] = verts_normed*100.0

            renderer = Renderer(model, (self.render_size,self.render_size),
                                     self.K, surf_color=(1, 1, 1), mode='rgb')


            ren_rgb = renderer.render(R_opengl, t)
            image_renders.append(ren_rgb)

            for i in range(10):
                # Calc bounding box and crop image
                org_img = image_renders[k]
                ys, xs = np.nonzero(org_img[:,:,0] > 0)
                obj_bb = calc_2d_bbox(xs,ys,[self.render_size,self.render_size])

                # Add relative offset when cropping - like Sundermeyer
                x, y, w, h = obj_bb

                rand_trans_x = np.random.uniform(-2.0, 2.0) * w
                rand_trans_y = np.random.uniform(-2.0, 2.0) * h

                scale = np.random.uniform(0.2, 0.8)
                obj_bb_off = obj_bb + np.array([rand_trans_x,rand_trans_y,
                                            w*scale,h*scale])

                try:
                    cropped = extract_square_patch(org_img, obj_bb_off)
                    images.append(cropped)
                except:
                    continue
        return images


    def generate_image_batch(self, Rin=None, tin=None, augment=True):
        # Generate random poses
        curr_Rs = []
        curr_ts = []
        curr_ids = []

        image_renders = []

        if(self.hard_mining == True):
            print("num hard samples: ", len(self.hard_samples))

        for k in np.arange(self.batch_size):
            obj_id = 0
            if Rin is None:
                R = self.pose_sampling()
                if(len(self.renderers) > 1):
                    obj_id = np.random.randint(0, len(self.renderers), size=1)[0]
                else:
                    obj_id = 0
                t = torch.tensor([0.0, 0.0, self.dist[obj_id][-1]])
            else:
                R = Rin[k]
                t = tin

            if(self.hard_mining == True):
                if(len(self.hard_samples) > 0):
                    rand = np.random.uniform(low=0.0, high=1.0, size=1)[0]
                    if(rand <= self.hard_sample_ratio):
                        rani = np.random.uniform(low=0, high=len(self.hard_samples)-1, size=1)[0]
                        #random.shuffle(self.hard_samples)
                        R = self.hard_samples.pop(int(rani))

            R = R.detach().cpu().numpy()
            t = t.detach().cpu().numpy()

            # Convert R matrix from pytorch to opengl format
            # for rendering only!
            xy_flip = np.eye(3, dtype=np.float)
            xy_flip[0,0] = -1.0
            xy_flip[1,1] = -1.0
            R_opengl = np.dot(R,xy_flip)
            R_opengl = np.transpose(R_opengl)

            # Randomize light position for rendering if enabled
            if(self.random_light is True):
                random_light_pos = (np.random.uniform(-1.0, 1.0, size=3)*self.dist).astype(np.float32)
            else:
                random_light_pos = None

            # Render images
            ren_rgb = self.renderers[obj_id].render(R_opengl, t, random_light_pos)

            curr_Rs.append(R)
            curr_ts.append(t)
            curr_ids.append(obj_id)

            image_renders.append(ren_rgb)

        if(len(self.backgrounds) > 0):
            bg_im_isd = np.random.choice(len(self.backgrounds), self.batch_size, replace=False)

        images = []
        for k in np.arange(self.batch_size):

            # Calc bounding box and crop image
            org_img = image_renders[k]
            ys, xs = np.nonzero(org_img[:,:,0] > 0)
            obj_bb = calc_2d_bbox(xs,ys,[self.render_size,self.render_size])

            # Add relative offset when cropping - like Sundermeyer
            x, y, w, h = obj_bb

            if augment:
                rand_trans_x = np.random.uniform(-self.max_rel_offset, self.max_rel_offset) * w
                rand_trans_y = np.random.uniform(-self.max_rel_offset, self.max_rel_offset) * h
            else:
                rand_trans_x = 0
                rand_trans_y = 0
            obj_bb_off = obj_bb + np.array([rand_trans_x,rand_trans_y,0,0])
            if(augment and self.max_rel_scale is not None):
                scale = np.random.uniform(-self.max_rel_scale, self.max_rel_scale)
                pad_factor = self.pad_factor + scale
            else:
                pad_factor = self.pad_factor

            cropped = extract_square_patch(org_img, obj_bb_off,
                                           pad_factor=pad_factor,
                                           resize=(self.img_size,self.img_size))

            if(self.realistic_occlusions):
                # Apply random renders behind
                num_behind = np.random.randint(0,4)
                for n in range(num_behind):
                    random_int = int(np.random.uniform(0, len(self.random_renders)-1))
                    behind = self.random_renders[random_int]
                    sum_img = np.sum(cropped[:,:,:3], axis=2)
                    mask = sum_img == 0
                    cropped[mask] = behind[mask]

                # Apply random renders behind
                num_front = np.random.randint(0,2)
                for n in range(num_front):
                    random_int = int(np.random.uniform(0, len(self.random_renders)-1))
                    front = self.random_renders[random_int]
                    sum_img = np.sum(front[:,:,:3], axis=2)
                    mask = sum_img != 0
                    cropped[mask] = front[mask]

            # Apply background
            if(len(self.backgrounds) > 0):
                img_back = self.backgrounds[bg_im_isd[k]]
                img_back = cv.cvtColor(img_back, cv.COLOR_BGR2RGBA).astype(float)
                alpha = cropped[:, :, 0:3].astype(float)
                sum_img = np.sum(cropped[:,:,:3], axis=2)
                alpha[sum_img > 0] = 1

                cropped[:, :, 0:3] = cropped[:, :, 0:3] * alpha + img_back[:, :, 0:3] * (1 - alpha)
            else:
                cropped = cropped[:, :, 0:3]

            # Augment data
            image_aug = np.array([cropped])
            if augment:
                image_aug = self.aug(images=image_aug)

            ## Convert to float and discard alpha channel
            image_aug = image_aug[0].astype(np.float)/255.0
            images.append(image_aug[:,:,:3])

        data = {"ids":curr_ids,
                "images":images,
                "Rs":curr_Rs}
        return data

    def generate_images(self, num_samples):
        data = {"ids":[],
                "images":[],
                "Rs":[]}
        while(len(data["images"]) < num_samples):
            curr_data = self.generate_image_batch()
            data["images"] = data["images"] + curr_data["images"]
            data["Rs"] = data["Rs"] + curr_data["Rs"]
            data["ids"] = data["ids"] + curr_data["ids"]
        data["images"] = data["images"][:num_samples]
        data["Rs"] = data["Rs"][:num_samples]
        data["ids"] = data["ids"][:num_samples]
        return data

    def __iter__(self):
        self.curr_samples = 0
        return self

    def __next__(self):
        if(self.curr_samples < self.max_samples):
            self.curr_samples += self.batch_size
            return self.generate_samples(self.batch_size)
        else:
            raise StopIteration

    def generate_samples(self, num_samples):
        data = self.generate_images(num_samples)
        return data


if __name__ == '__main__':
    main()
