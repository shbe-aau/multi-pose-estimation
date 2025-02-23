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

import imgaug as ia
import random

optimizer = None
lr_reducer = None
pipeline = None
num_views = 10
epoch = 0

dbg_memory = False

def dbg(message, flag):
    if flag:
        print(message)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def latestCheckpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, "*.pt"))
    checkpoints_sorted = sorted(checkpoints, key=natural_keys)

    if(len(checkpoints_sorted) > 0):
        return checkpoints_sorted[-1]
    return None

def loadDataset(file_list, batch_size=2, obj_id=0):
    #data = {"codes":[],"Rs":[],"images":[]}
    data = []
    for f in file_list:
        print("Loading dataset: {0}".format(f))
        with open(f, "rb") as f:
            curr_data = pickle.load(f, encoding="latin1")
            curr_batch = {"Rs":[],"images":[], "ids":[]}
            for i in range(len(curr_data["Rs"])):
                curr_pose = curr_data["Rs"][i]

                # Convert from T-LESS to Pytorch3D format
                xy_flip = np.eye(3, dtype=np.float)
                xy_flip[0,0] = -1.0
                xy_flip[1,1] = -1.0
                curr_pose = np.transpose(curr_pose)
                curr_pose = np.dot(curr_pose, xy_flip)
                curr_batch["Rs"].append(curr_pose)

                # Normalize image
                curr_image = curr_data["images"][i]
                curr_image = curr_image/np.max(curr_image)
                curr_batch["images"].append(curr_image)

                # Temp fix for loading pickle without ids
                curr_batch["ids"].append(obj_id)

                if(len(curr_batch["Rs"]) >= batch_size):
                    data.append(curr_batch)
                    curr_batch = {"Rs":[],"images":[],"ids":[]}
            data.append(curr_batch)
    return data

def main():
    global optimizer, lr_reducer, num_views, epoch, pipeline
    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    arguments = parser.parse_args()

    if(arguments.experiment_name.startswith('./experiments') or arguments.experiment_name.startswith('experiments')):
        cfg_file_path = arguments.experiment_name
    else:
        cfg_file_path = os.path.join("./experiments", arguments.experiment_name)
    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    # Set random seed
    seed=args.getint('Training', 'RANDOM_SEED')
    if(seed is not None):
        torch.manual_seed(seed)
        #torch.use_deterministic_algorithms(True) # Requires pytorch>=1.8.0
        #torch.backends.cudnn.deterministic = True
        np.random.seed(seed=seed)
        ia.seed(seed)
        random.seed(seed)

    model_seed=args.getint('Training', 'MODEL_RANDOM_SEED', fallback=None)
    if(model_seed is not None):
        torch.manual_seed(model_seed)

    # Prepare rotation matrices for multi view loss function
    eulerViews = json.loads(args.get('Rendering', 'VIEWS'))
    if isinstance(eulerViews, int):
        num_views = eulerViews
    else:
        num_views = len(eulerViews)
    #views = prepareViews(eulerViews) # remnant from when we set views by hand

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
    model = Model(num_views=num_views,
                  num_objects=len(model_path_loss),
                  finetune_encoder=args.getboolean('Training','FINETUNE_ENCODER', fallback=False),
                  classify_objects=args.getboolean('Training','CLASSIFY_OBJECTS', fallback=False),
                  weight_init_name=args.get('Training', 'WEIGHT_INIT_NAME', fallback=""))
    model.to(device)

    # Fine-tune the last FC layer in the encoder
    encoder = Encoder(args.get('Dataset', 'ENCODER_WEIGHTS')).to(device)
    if(model.finetune_encoder):
        # Copy FC layer from the encoder to the model
        model.encoder.state_dict()['weight'].copy_(encoder.encoder_dense_MatMul.state_dict()['weight'])
        model.encoder.state_dict()['bias'].copy_(encoder.encoder_dense_MatMul.state_dict()['bias'])
        encoder.encoder_dense_MatMul = None

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    low_lr = args.getfloat('Training', 'LEARNING_RATE_LOW')
    high_lr = args.getfloat('Training', 'LEARNING_RATE_HIGH')
    optimizer = torch.optim.Adam(model.parameters(), lr=low_lr)
    if(low_lr != high_lr):
        lr_reducer = OneCycleLR(optimizer, num_steps=args.getfloat('Training', 'NUM_ITER'), lr_range=(low_lr, high_lr))
    else:
        lr_reducer = None

    # Prepare output directories
    output_path = args.get('Training', 'OUTPUT_PATH')
    prepareDir(output_path)
    shutil.copy(cfg_file_path, os.path.join(output_path, cfg_file_path.split("/")[-1]))

    # Setup early stopping if enabled
    early_stopping = args.getboolean('Training', 'EARLY_STOPPING', fallback=False)
    if early_stopping:
        window = args.getint('Training', 'STOPPING_WINDOW', fallback=10)
        time_limit = args.getint('Training', 'STOPPING_TIME_LIMIT', fallback=10)
        window_means = []
        lowest_mean = np.inf
        lowest_x = 0
        timer = 0

    # Load checkpoint for last epoch if it exists
    model_path = latestCheckpoint(os.path.join(output_path, "models/"))
    if(model_path is not None):
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch'] + 1

        # Load model
        model.load_state_dict(checkpoint['model'])

        # Load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Load LR reducer if it exists
        try:
            lr_reducer.load_state_dict(checkpoint['lr_reducer'])
        except:
            lr_reducer = None
        print("Loaded the checkpoint: \n" + model_path)

    if early_stopping:
        validation_csv=os.path.join(output_path, "validation-loss.csv")
        if os.path.exists(validation_csv):
            with open(validation_csv) as f:
                val_reader = csv.reader(f, delimiter='\n')
                val_loss = list(val_reader)
            val_losses = np.array(val_loss, dtype=np.float32).flatten()
            for epoch in range(window,len(val_loss)):
                timer += 1
                w_mean = np.mean(val_losses[epoch-window:epoch])
                window_means.append(w_mean)
                if w_mean < lowest_mean:
                    lowest_mean = w_mean
                    lowest_x = epoch
                    timer = 0


    # Prepare pipeline
    pipeline = Pipeline(encoder, model, device)
    encoder.eval()

    # Handle loading of multiple object paths and translations
    try:
        model_path_data = json.loads(args.get('Dataset', 'MODEL_PATH_DATA'))
        translations = np.array(json.loads(args.get('Rendering', 'T')))
    except:
        model_path_data = [args.get('Dataset', 'MODEL_PATH_DATA')]
        translations = [np.array(json.loads(args.get('Rendering', 'T')))]


    # Check if training based on PBR using DataLoader is enabled
    if(args.has_option('Dataset', 'PBR_PATH')):
        pbr_path = args.get('Dataset', 'PBR_PATH')
        obj_ids = json.loads(args.get('Dataset', 'PBR_OBJ_IDS'))

        from DatasetPBR import DatasetGenerator
        training_dataset = DatasetGenerator(pbr_path, obj_ids)
        training_data = torch.utils.data.DataLoader(training_dataset, args.getint('Training', 'BATCH_SIZE'), shuffle=True)
    else: # Default to old approach in case PBR path is not specified
        # Prepare datasets
        from DatasetGeneratorOpenGL import DatasetGenerator
        bg_path = "../../autoencoder_ws/data/VOC2012/JPEGImages/"
        training_data = DatasetGenerator(args.get('Dataset', 'BACKGROUND_IMAGES'),
                                         model_path_data,
                                         translations,
                                         args.getint('Training', 'BATCH_SIZE'),
                                         "not_used",
                                         device,
                                         sampling_method = args.get('Training', 'VIEW_SAMPLING'),
                                         max_rel_offset = args.getfloat('Training', 'MAX_REL_OFFSET', fallback=0.2),
                                         augment_imgs = args.getboolean('Training', 'AUGMENT_IMGS', fallback=True),
                                         seed=args.getint('Training', 'RANDOM_SEED'))
        training_data.max_samples = args.getint('Training', 'NUM_SAMPLES')

    # Load the validationset
    try:
        valid_data_paths = json.loads(args.get('Dataset', 'VALID_DATA_PATH'))
    except:
        valid_data_paths = [args.get('Dataset', 'VALID_DATA_PATH')]


    validation_data = []
    for curr_obj_id,v in enumerate(valid_data_paths):
        validation_data.append(loadDataset([v],
                                           args.getint('Training', 'BATCH_SIZE'),
                                           obj_id=curr_obj_id))
    print("Loaded {0} validation sets!".format(len(validation_data)))

    # Start training
    while(epoch < args.getint('Training', 'NUM_ITER')):
        # Set random seed based on current epoch
        seed=args.getint('Training', 'RANDOM_SEED')
        if(seed is not None):
            seed += epoch
            torch.manual_seed(seed)
            #torch.use_deterministic_algorithms(True) # Requires pytorch>=1.8.0
            #torch.backends.cudnn.deterministic = True
            np.random.seed(seed=seed)
            ia.seed(seed)
            random.seed(seed)

        model_seed=args.getint('Training', 'MODEL_RANDOM_SEED', fallback=None)
        if(model_seed is not None):
            model_seed += epoch
            torch.manual_seed(model_seed)

        # Train on synthetic data
        model = model.train() # Set model to train mode
        loss = runEpoch(br, training_data, model, device, output_path,
                          t=translations, config=args)
        append2file([loss], os.path.join(output_path, "train-loss.csv"))
        if(lr_reducer is not None):
            append2file([lr_reducer.get_lr()], os.path.join(output_path, "learning-rate.csv"))
        else:
            append2file([optimizer.param_groups[0]['lr']], os.path.join(output_path, "learning-rate.csv"))

        # Test on validation data
        model = model.eval() # Set model to eval mode
        val_loss_list = []
        for curr_obj_id,v in enumerate(validation_data):
            val_loss = runEpoch(br, v, model, device, output_path, t=translations, config=args)
            val_loss_list.append(val_loss)
            append2file([val_loss], os.path.join(output_path,
                                                 "validation-obj{0}-loss.csv".format(curr_obj_id)))
            plotLoss(os.path.join(output_path, "train-loss.csv"),
                     os.path.join(output_path, "validation-obj{0}-loss.png".format(curr_obj_id)),
                     validation_csv=os.path.join(output_path,"validation-obj{0}-loss.csv".format(curr_obj_id)))
        val_loss = np.mean(np.array(val_loss_list))
        append2file([val_loss], os.path.join(output_path, "validation-loss.csv"))

        # Plot losses
        val_losses = plotLoss(os.path.join(output_path, "train-loss.csv"),
                 os.path.join(output_path, "train-loss.png"),
                 validation_csv=os.path.join(output_path, "validation-loss.csv"))
        print("-"*20)
        print("Epoch: {0} - train loss: {1} - validation loss: {2}".format(epoch,loss,val_loss))
        print("-"*20)
        if early_stopping and epoch >= window:
            timer += 1
            if timer > time_limit:
                # print stuff here
                print()
                print("-"*60)
                print("Validation loss seems to have plateaued, stopping early.")
                print("Best mean loss value over an epoch window of size {} was found at epoch {} ({:.8f} mean loss)".format(window, lowest_x, lowest_mean))
                print("-"*60)
                break
            w_mean = np.mean(val_losses[epoch-window:epoch])
            window_means.append(w_mean)
            if w_mean < lowest_mean:
                lowest_mean = w_mean
                lowest_x = epoch
                timer = 0
        epoch = epoch+1

def runEpoch(br, dataset, model,
               device, output_path, t, config):
    global optimizer, lr_reducer
    dbg("Before train memory: {}".format(torch.cuda.memory_summary(device=device, abbreviated=False)), dbg_memory)

    if(model.training):
        print("Current mode: train!")
        if(lr_reducer is not None):
            print("Epoch: {0} - current learning rate: {1}".format(epoch, lr_reducer.get_lr()))
        else:
            print("Epoch: {0} - current learning rate: {1}".format(epoch, optimizer.param_groups[0]['lr']))
        dataset.hard_samples = [] # Reset hard samples
        torch.set_grad_enabled(True)
    else:
        print("Current mode: eval!")
        torch.set_grad_enabled(False)

    losses = []
    batch_size = br.batch_size
    hard_indeces = []

    for i,curr_batch in enumerate(dataset):
        if(model.training):
            optimizer.zero_grad()

        if(isinstance(dataset, torch.utils.data.DataLoader)):
            # We must be using DataLoader
            # re-format data to fit the old way
            curr_batch_tmp = {}
            curr_batch_tmp["images"] = curr_batch[0].numpy()
            curr_batch_tmp["Rs"] = curr_batch[1][1]
            curr_batch_tmp["ids"] = curr_batch[1][0]
            curr_batch = curr_batch_tmp

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
                                                             views=num_views,
                                                             config=config)

        Rs = torch.tensor(np.stack(Rs), device=device, dtype=torch.float32)

        print("Grad: ", loss.requires_grad)

        if(model.training):
            loss.backward()
            optimizer.step()

            # # DEBUG! REMOVE
            # print("Encoder conv: ", torch.sum(pipeline.encoder.encoder_conv2d_Conv2D.weight))
            # print("Encoder FC (encoder): ", None if pipeline.encoder.encoder_dense_MatMul is None else torch.sum(pipeline.encoder.encoder_dense_MatMul.weight))
            # print("Encoder FC (model): ", None if pipeline.model.encoder is None else torch.sum(pipeline.model.encoder.weight))
            # print("Pose network: ", torch.sum(pipeline.model.l3.weight))

        #detach all from gpu
        loss.detach().cpu().numpy()
        gt_images.detach().cpu().numpy()
        predicted_images.detach().cpu().numpy()


        # Check for nan loss
        if(torch.isnan(loss)):
            print("nan loss !!!! OH NOOOOO!!!")
            stop


        if(model.training):
            print("Batch: {0}/{1} (size: {2}) - loss: {3}".format(i+1,len(dataset), len(Rs),torch.mean(batch_loss)))
        else:
            print("Test batch: {0}/{1} (size: {2}) - loss: {3}".format(i+1,len(dataset), len(Rs),torch.mean(batch_loss)))
            #print("Test batch: {0}/{1} (size: {2}) - loss: {3}".format(i+1, round(dataset.max_samples/batch_size), len(Rs),torch.mean(batch_loss)))
        losses = losses + batch_loss.data.detach().cpu().numpy().tolist()

        if(config.getboolean('Training', 'SAVE_IMAGES')):
            if(model.training):
                batch_img_dir = os.path.join(output_path, "images/epoch{0}".format(epoch))
            else:
                batch_img_dir = os.path.join(output_path, "val-images/epoch{0}/obj{1}".format(epoch,ids[0]))
            prepareDir(batch_img_dir)
            gt_img = (gt_images[0]).detach().cpu().numpy()
            predicted_img = (predicted_images[0]).detach().cpu().numpy()

            vmin = np.linalg.norm(T)*0.9
            vmax = max(np.max(gt_img), np.max(predicted_img))

            fig = plt.figure(figsize=(12,3+num_views*2))
            #for viewNum in np.arange(num_views):
            plotView(0, num_views, vmin, vmax, input_images, gt_images, predicted_images,
                     predicted_poses, batch_loss, batch_size, threshold=config['Loss_parameters'].getfloat('DEPTH_MAX'))
            fig.tight_layout()

            fig.savefig(os.path.join(batch_img_dir, "epoch{0}-batch{1}.png".format(epoch,i)), dpi=fig.dpi)
            plt.close()

    if(model.training):
        # Save current model
        model_dir = os.path.join(output_path, "models/")
        prepareDir(model_dir)
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'lr_reducer': lr_reducer.state_dict() if lr_reducer is not None else None,
                 'epoch': epoch}
        torch.save(state, os.path.join(model_dir,"model-epoch{0}.pt".format(epoch)))
        if(lr_reducer is not None):
            lr_reducer.step()

    # Memory management
    dbg("After train memory: {}".format(torch.cuda.memory_summary(device=device, abbreviated=False)), dbg_memory)
    gc.collect()
    return np.mean(losses)


if __name__ == '__main__':
    main()
