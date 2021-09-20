import os
import shutil
import torch
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import configparser
import json
import argparse
import glob
import cv2

from utils.utils import *

from Model import Model
from BatchRender import BatchRender
from losses import Loss

from scipy.spatial.transform import Rotation as R

def eqv_dist_points(n):
    pi = np.pi
    sin = np.sin
    cos = np.cos
    n_count = 0
    a = 4*pi/n # 4 pi r^2 / N for r = 1
    d = np.sqrt(a)
    m_theta = int(np.floor(pi/d))
    d_theta = pi/m_theta
    d_phi = a/d_theta

    points = []
    for i in range(m_theta):
        theta = pi*(i + 0.5)/m_theta
        m_phi = int(np.floor(2*pi*sin(theta)/d_phi))
        for j in range(m_phi):
            point = {}
            phi = 2*pi*j/m_phi
            point['spherical'] = [theta, phi]
            point['cartesian'] = [sin(theta)*cos(phi), \
                                  sin(theta)*sin(phi), \
                                  cos(theta)]
            n_count += 1
            points.append(point)
    return points

def plot_points(points, name):
    cart = [point['cartesian'] for point in points]
    print(cart[0:1])
    x, y, z = zip(*cart)
    fig = plt.figure()
    plt.scatter(x, y, z)
    fig.savefig(name, dpi=fig.dpi)
    #plt.show()

def toMatArray(point):
    r = R.from_euler('yz', point['spherical']) # select point wanted for comparison here
    Rs = []
    Rs.append(r.as_matrix())
    return Rs


def render_point(point, ts, renderer, mean, std, views):
    Rs = toMatArray(point)
    Rs = torch.tensor(np.stack(Rs), device=renderer.device,
                            dtype=torch.float32)
    images = []
    for v in views:
        # Render images
        Rs_new = torch.matmul(Rs, v.to(renderer.device))
        imgs = renderer.renderBatch(Rs_new, ts)
        imgs = (imgs-mean)/std
        images.append(imgs)
    return torch.cat(images, dim=1)

def main():
    global learning_rate, optimizer, views, epoch
    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    arguments = parser.parse_args()

    cfg_file_path = os.path.join(arguments.path)
    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    # Prepare rotation matrices for multi view loss function
    eulerViews = json.loads(args.get('Rendering', 'VIEWS'))
    views = prepareViews(eulerViews)

    # Set the cuda device
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    # Handle loading of multiple object paths
    try:
        model_path_loss = json.loads(args.get('Dataset', 'MODEL_PATH_LOSS'))
    except:
        model_path_loss = [args.get('Dataset', 'MODEL_PATH_LOSS')]

    # Set up batch renderer
    br = BatchRender(model_path_loss,
                     device,
                     batch_size=args.getint('Training', 'BATCH_SIZE'),
                     render_method=args.get('Rendering', 'SHADER'),
                     image_size=args.getint('Rendering', 'IMAGE_SIZE'))

    output_path = args.get('Training', 'OUTPUT_PATH')
    batch_img_dir = os.path.join(output_path, 'images')
    prepareDir(batch_img_dir)

    # collect points to use
    points = eqv_dist_points(int(args.get('Sampling', 'NUM_SAMPLES')))
    plot_points(points, os.path.join(batch_img_dir, 'test2.png'))
    np.save(os.path.join(output_path, 'points.npy'), points)

    # Testing using existing rotaions, to be replaced
    # data = pickle.load(open(args.get('Dataset', 'TRAIN_DATA_PATH'),'rb'), encoding='latin1')

    t=json.loads(args.get('Rendering', 'T'))
    T = np.array(t, dtype=np.float32)
    ts = []
    #for b in curr_batch:
    #    Rs.append(data['Rs'][b])
    #    ts.append(T.copy())
    # Rs.append(data['Rs'][0])
    # print(Rs)
    ts.append(T.copy())

    ref_num = int(args.get('Sampling', 'REFERENCE_NUM', fallback=0)) % len(points)
    # referene point data, right now it's the first point in the set
    ref_image = render_point(points[ref_num], ts, br, 0, 1, views)
    ref_pose = toMatArray(points[ref_num])

    loss_method=args.get('Training', 'LOSS')

    i = 0
    losses = []
    for point in points:
        if i % 100 is 0:
            print("{} of {}".format(i, len(points)))
        #image = render_point(point, ts, br, 0, 1, views)

        prepareDir(output_path)
        shutil.copy(cfg_file_path, os.path.join(output_path, cfg_file_path.split('/')[-1]))

        #gt_img = (image[0]).detach().cpu().numpy()

        #im = gt_img
        #im = np.array(gt_img * 255, dtype = np.uint8)
        #threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        #if args.getboolean('Training', 'SAVE_IMAGES'):
        #    cv2.imwrite(os.path.join(batch_img_dir, '{}.png'.format(i)), im)

        flattened = np.append([0], toMatArray(point)[0].flatten())

        poses = torch.tensor([flattened], dtype=torch.float32, device=device)

        loss, batch_loss, gt_images, predicted_images = Loss(poses, ref_pose, br, ts, config=args, views=views, fixed_gt_images=ref_image)
        loss = (loss).detach().cpu().numpy()
        im = (predicted_images).detach().cpu().numpy()
        if args.get('Rendering', 'SHADER')=="hard-phong":
            im = np.reshape(im, (len(views)*args.getint('Rendering', 'IMAGE_SIZE'), args.getint('Rendering', 'IMAGE_SIZE'),3))
            im = im * 255
        else:
            im = np.reshape(im, (len(views)*args.getint('Rendering', 'IMAGE_SIZE'), args.getint('Rendering', 'IMAGE_SIZE')))
        if args.getboolean('Training', 'SAVE_IMAGES'):
            cv2.imwrite(os.path.join(batch_img_dir, '{}.png'.format(i)), im)
        losses.append(loss)
        i += 1

    np.save(os.path.join(output_path, 'losses.npy'), losses)

if __name__ == '__main__':
    main()
