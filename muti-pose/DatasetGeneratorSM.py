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
from scipy.spatial.transform import Rotation as scipyR
from utils.utils import *
from utils.sundermeyer.pysixd import view_sampler

import configparser
from dataset import Dataset

def build_dataset(dataset_path, args):
    dataset_args = { k:v for k,v in
        args.items('Dataset') +
        args.items('Paths') +
        args.items('Augmentation')+
        args.items('Queue') +
        args.items('Embedding')}
    dataset = Dataset(dataset_path, **dataset_args)
    return dataset

class DatasetGenerator():

    def __init__(self, background_path, obj_path, obj_distance, batch_size,
                 _, device, sampling_method="sphere", random_light=True,
                 num_bgs=15000):
        self.curr_samples = 0
        self.max_samples = 1000
        self.obj_path = obj_path
        self.batch_size = batch_size
        self.dist = obj_distance

        args = configparser.ConfigParser()
        args.read("test.cfg")

        if('reconst' in obj_path):
            args.set('Dataset', 'MODEL', 'reconst')
        elif('cad' in obj_path):
            args.set('Dataset', 'MODEL', 'cad')
        else:
            print("Can determine render type (reconst/cad) from model path: ", obj_path)

        args.set('Paths', 'MODEL_PATH', obj_path)
        args.set('Dataset', 'NOOF_TRAINING_IMGS', str(int(batch_size)))
        args.set('Dataset', 'NOOF_BG_IMGS', str(int(num_bgs)))

        self.dataset = build_dataset("./", args)
        self.dataset.load_bg_images("./")

        self.hard_samples = []
        self.hard_mining = False
        if(sampling_method.split("-")[-1] == "hard"):
            self.hard_mining = True
        sampling_method = sampling_method.replace("-hard","")
        self.hard_sample_ratio = 0.3
        self.hard_mining_ratio = 0.2

        self.pose_sampling = None
        if(sampling_method == "sphere-wolfram-fixed"):
            self.pose_sampling = self.sphere_wolfram_sampling_fixed
        elif(sampling_method == "sundermeyer-random"):
            self.pose_sampling = self.sm_quat_random
        elif(sampling_method == "mixed"):
            self.pose_sampling = self.mixed
        elif(sampling_method == "viewsphere-aug"):
            self.pose_sampling = self.viewsphere_aug
            # Stuff for viewsphere aug sampling
            self.view_sphere = None
            self.view_sphere_indices = []
            self.random_aug = None
        else:
            print("ERROR! Invalid view sampling method: {0}".format(sampling_method))

    def mixed(self):
        rand = np.random.uniform(low=-100, high=100, size=1)[0]
        if(rand > 0):
            #print("using wolfram sampling!")
            return self.sphere_wolfram_sampling_fixed()
        #print("using sm sampling!")
        return self.sm_quat_random()

    # Truely random
    # Based on: https://mathworld.wolfram.com/SpherePointPicking.html
    def sphere_wolfram_sampling_fixed(self):
        #print("wolfram sphere sampling!")
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
        rot_degrees = np.random.uniform(low=-90.0, high=90.0, size=1)
        rot = scipyR.from_euler('z', rot_degrees, degrees=True)
        rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
        R = torch.matmul(R, rot_mat)
        R = R.squeeze()
        return R

    # Based on Sundermeyer
    def sm_quat_random(self):
        #print("sm quat random!")
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
        #t = torch.tensor([0.0, 0.0, self.dist])
        return R #,t

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
        #print("viewsphere aug sampling!")
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
        t = torch.tensor([0.0, 0.0, self.dist])
        return R,t

    def generate_images(self, num_samples):
        data = {"images":[],
                "Rs":[]}

        for i in range(num_samples):
            # Sample rotation matrix
            R = self.pose_sampling()

            # Sample from hard samples if possible
            if(self.hard_mining == True):
                if(len(self.hard_samples) > 0):
                    rand = np.random.uniform(low=0.0, high=1.0, size=1)[0]
                    if(rand <= self.hard_sample_ratio):
                        rani = np.random.uniform(low=0, high=len(self.hard_samples)-1, size=1)[0]
                        rani = int(rani)
                        R = self.hard_samples.pop(rani).detach().cpu().numpy()
                        #print("using hard sample. {0} hard samples left!".format(len(self.hard_samples)))

            # Convert R matrix from pytorch to opengl format
            # for rendering only!
            xy_flip = np.eye(3, dtype=np.float)
            xy_flip[0,0] = -1.0
            xy_flip[1,1] = -1.0
            R_opengl = np.dot(R,xy_flip)
            R_opengl = np.transpose(R_opengl)

            # Render image
            img = self.dataset.render_training_image(R_opengl)

            # Convert BGR to RGB
            img = np.flip(img,axis=2)

            data["images"].append(img)
            data["Rs"].append(R)
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

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_path", help="path to the .obj file")
    parser.add_argument("-b", help="batch size, each batch will have the same pose but different augmentations", type=int, default=8)
    parser.add_argument("-d", help="distance to the object", type=float, default=2000.0)
    parser.add_argument("-n", help="number of total samples", type=int, default=1000)
    parser.add_argument("-v", help="visualize the data", default=False)
    parser.add_argument("-o", help="output path", default="")
    parser.add_argument("-bg", help="background images path", default="")
    parser.add_argument("-s", help="pose sampling method", default="tless-simple")
    parser.add_argument("-e", help="path to .npy encoder weights", default=None)
    parser.add_argument("-rl", help="enable random light", default=True)
    parser.add_argument("-ng", help="number of backgrounds", type=int, default=17000)
    arguments = parser.parse_args()

    # Create dataset generator
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dg = DatasetGenerator(background_path=arguments.bg,
                          obj_path=arguments.obj_path,
                          obj_distance=arguments.d,
                          batch_size=arguments.b,
                          sampling_method=arguments.s,
                          encoder_weights=arguments.e,
                          random_light=str2bool(arguments.rl),
                          num_bgs=arguments.ng,
                          device=device)

    # Generate data
    data = dg.generate_samples(num_samples=arguments.n)

    # Visualize it (optional)
    if(str2bool(arguments.v)):
        for i,img in enumerate(data["images"]):
            window_name = "Sample {0}/{1}".format(i,arguments.n)
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name,42,42)
            # Flip last axis to convert from RGB to BGR before showing using cv2
            cv2.imshow(window_name, np.flip(img,axis=2))
            key = cv2.waitKey(0)
            cv2.destroyWindow(window_name)
            if(key == ord("q")):
                break

    # Save generated data
    output_path = arguments.o
    if(output_path == ""):
        output_path = "./training-images.p"
    pickle.dump(data, open(output_path, "wb"), protocol=2)
    print("Saved dataset to: ", output_path)
