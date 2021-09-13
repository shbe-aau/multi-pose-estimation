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
import gc
import re

from utils.utils import *
from utils.onecyclelr import OneCycleLR

from Model import Model
from Encoder import Encoder
from Pipeline import Pipeline
from BatchRender import BatchRender
from losses import Loss
from DatasetGeneratorOpenGL import DatasetGenerator
#from DatasetGeneratorSM import DatasetGenerator

optimizer = None
lr_reducer = None
pipeline = None
views = []
epoch = 0

dbg_memory = False

def dbg(message, flag):
    if flag:
        print(message)

from torch.optim.lr_scheduler import _LRScheduler
class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries
    over a number of iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # In earlier Pytorch versions last_epoch starts at -1, while in recent versions
        # it starts at 0. We need to adjust the math a bit to handle this. See
        # discussion at: https://github.com/davidtvs/pytorch-lr-finder/pull/42
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

def main():
    global optimizer, lr_reducer, views, epoch, pipeline
    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    arguments = parser.parse_args()

    cfg_file_path = os.path.join("./experiments", arguments.experiment_name)
    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    # Prepare rotation matrices for multi view loss function
    eulerViews = json.loads(args.get('Rendering', 'VIEWS'))
    views = prepareViews(eulerViews)

    # Set the cuda device
    device = torch.device("cuda:0")
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
                     faces_per_pixel=args.getint('Rendering', 'FACES_PER_PIXEL'),
                     render_method=args.get('Rendering', 'SHADER'),
                     image_size=args.getint('Rendering', 'IMAGE_SIZE'),
                     norm_verts=args.getboolean('Rendering', 'NORMALIZE_VERTICES'))

    # Set size of model output depending on pose representation - deprecated?
    pose_rep = args.get('Training', 'POSE_REPRESENTATION')
    if(pose_rep == '6d-pose'):
        pose_dim = 6
    elif(pose_rep == 'quat'):
        pose_dim = 4
    elif(pose_rep == 'axis-angle'):
        pose_dim = 4
    elif(pose_rep == 'euler'):
        pose_dim = 3
    else:
        print("Unknown pose representation specified: ", pose_rep)
        pose_dim = -1

    # Initialize a model using the renderer, mesh and reference image
    model = Model(num_views=len(views))
    model.to(device)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    low_lr = args.getfloat('Training', 'LEARNING_RATE_LOW')
    high_lr = args.getfloat('Training', 'LEARNING_RATE_HIGH')
    optimizer = torch.optim.Adam(model.parameters(), lr=low_lr)
    lr_reducer = ExponentialLR(optimizer, high_lr, args.getfloat('Training', 'NUM_ITER'))

    # Prepare output directories
    output_path = args.get('Training', 'OUTPUT_PATH')
    prepareDir(output_path)
    shutil.copy(cfg_file_path, os.path.join(output_path, cfg_file_path.split("/")[-1]))

    # Prepare pipeline
    encoder = Encoder(args.get('Dataset', 'ENCODER_WEIGHTS')).to(device)
    encoder.eval()
    pipeline = Pipeline(encoder, model, device)

    # Handle loading of multiple object paths and translations
    try:
        model_path_data = json.loads(args.get('Dataset', 'MODEL_PATH_DATA'))
        translations = np.array(json.loads(args.get('Rendering', 'T')))
    except:
        model_path_data = [args.get('Dataset', 'MODEL_PATH_DATA')]
        translations = [np.array(json.loads(args.get('Rendering', 'T')))]

    # Prepare datasets
    bg_path = "../../autoencoder_ws/data/VOC2012/JPEGImages/"
    training_data = DatasetGenerator(args.get('Dataset', 'BACKGROUND_IMAGES'),
                                     model_path_data,
                                     translations,
                                     args.getint('Training', 'BATCH_SIZE'),
                                     "not_used",
                                     device,
                                     args.get('Training', 'VIEW_SAMPLING'))
    training_data.max_samples = args.getint('Training', 'NUM_SAMPLES')

    # Start training
    np.random.seed(seed=args.getint('Training', 'RANDOM_SEED'))
    while(epoch < args.getint('Training', 'NUM_ITER')):
        # Train on synthetic data
        model = model.train() # Set model to train mode
        loss = runEpoch(br, training_data, model, device, output_path,
                          t=translations, config=args)
        append2file([loss], os.path.join(output_path, "train-loss.csv"))
        append2file([lr_reducer.get_lr()], os.path.join(output_path, "learning-rate.csv"))

        # Plot losses
        val_losses = plotLoss(os.path.join(output_path, "train-loss.csv"),
                              os.path.join(output_path, "train-loss.png"),
                              validation_csv=os.path.join(output_path, "train-loss.csv"),)
        print("-"*20)
        print("Epoch: {0} - train loss: {1}".format(epoch,loss))
        print("-"*20)
        epoch = epoch+1

def runEpoch(br, dataset, model,
               device, output_path, t, config):
    global optimizer, lr_reducer
    dbg("Before train memory: {}".format(torch.cuda.memory_summary(device=device, abbreviated=False)), dbg_memory)

    print("Epoch: {0} - current learning rate: {1}".format(epoch, lr_reducer.get_lr()))
    dataset.hard_samples = [] # Reset hard samples
    torch.set_grad_enabled(True)

    losses = []
    batch_size = br.batch_size
    hard_indeces = []

    for i,curr_batch in enumerate(dataset):
        if(model.training):
            optimizer.zero_grad()

        # Fetch images
        input_images = curr_batch["images"]

        # Predict poses
        predicted_poses = pipeline.process(input_images)

        # Prepare ground truth poses for the loss function
        T = np.array(t, dtype=np.float32)
        Rs = curr_batch["Rs"]
        ids = curr_batch["ids"]
        ts = [np.array(t[curr_id], dtype=np.float32) for curr_id in ids]

        # Calculate the loss
        loss, batch_loss, gt_images, predicted_images = Loss(predicted_poses, Rs, br,
                                                             ts,
                                                             ids=ids,
                                                             views=views,
                                                             config=config)

        Rs = torch.tensor(np.stack(Rs), device=device, dtype=torch.float32)

        print("Grad: ", loss.requires_grad)

        if(model.training):
            loss.backward()
            optimizer.step()

        #detach all from gpu
        loss.detach().cpu().numpy()
        gt_images.detach().cpu().numpy()
        predicted_images.detach().cpu().numpy()

        losses = losses + batch_loss.data.detach().cpu().numpy().tolist()

        batch_img_dir = os.path.join(output_path, "images/epoch{0}".format(epoch))
        prepareDir(batch_img_dir)
        gt_img = (gt_images[0]).detach().cpu().numpy()
        predicted_img = (predicted_images[0]).detach().cpu().numpy()

        vmin = np.linalg.norm(T)*0.9
        vmax = max(np.max(gt_img), np.max(predicted_img))

        fig = plt.figure(figsize=(12,3+len(views)*2))
        plotView(0, len(views), vmin, vmax, input_images, gt_images, predicted_images,
                 predicted_poses, batch_loss, batch_size, threshold=config['Loss_parameters'].getfloat('DEPTH_MAX'))
        fig.tight_layout()
        fig.savefig(os.path.join(batch_img_dir, "epoch{0}-batch{1}.png".format(epoch,i)), dpi=fig.dpi)
        plt.close()

        lr_reducer.step()
        break

    # Memory management
    dbg("After train memory: {}".format(torch.cuda.memory_summary(device=device, abbreviated=False)), dbg_memory)
    gc.collect()
    return np.mean(losses)


if __name__ == '__main__':
    main()
