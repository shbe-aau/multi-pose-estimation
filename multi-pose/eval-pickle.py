import sys
sys.path.append("..")

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

import copy

from utils.utils import *
from utils.tools import *

from Pipeline import Pipeline
from Model import Model
#from ModelSkipOnlyFourExtra import Model
#from ModelSkipOnly import Model
#from ModelFourExtra import Model
#from ModelTwoExtra import Model
#from ModelBig import Model
#from ModelSmall import Model
#from ModelDeep import Model
from Encoder import Encoder
from utils.pytless import inout, misc
from utils.pytless.renderer import Renderer

def arr2str(arr):
    flat_arr = arr.flatten().tolist()
    str_arr = ""
    for i in np.arange(len(flat_arr)):
        str_arr += "{0:.8f} ".format(flat_arr[i])
    return str_arr[:-1]

def correct_trans_offset(R, t_est):
    # Translation offset correction
    d_alpha_x = np.arctan(t_est[0]/t_est[2])
    d_alpha_y = np.arctan(t_est[1]/t_est[2])
    R_corr_x = np.array([[1,0,0],
                         [0,np.cos(d_alpha_y),-np.sin(d_alpha_y)],
                         [0,np.sin(d_alpha_y),np.cos(d_alpha_y)]])
    R_corr_y = np.array([[np.cos(d_alpha_x),0,-np.sin(d_alpha_x)],
                         [0,1,0],
                         [np.sin(d_alpha_x),0,np.cos(d_alpha_x)]])
    R_corrected = np.dot(R_corr_y,np.dot(R_corr_x,R))
    return R_corrected

def main():
    visualize = True

    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", help="path to the model checkpoint")
    parser.add_argument("-ep", help="path to the encoder weights")
    parser.add_argument("-pi", help="path to the pickle input file")
    parser.add_argument("-op", help="path to the CAD model for the object", default=None)
    parser.add_argument("-o", help="output path", default="./output.csv")
    args = parser.parse_args()

    # Load dataset
    data = pickle.load(open(args.pi,"rb"), encoding="latin1")

    # Load configuration file
    cfg_file_path = args.mp.replace("/models/{0}".format(args.mp.split('/')[-1]),"")
    cfg_file_path = glob.glob('{0}/*.cfg'.format(cfg_file_path))[0]
    conf = configparser.ConfigParser()
    conf.read(cfg_file_path)

    try:
        model_path_loss = json.loads(conf.get('Dataset', 'MODEL_PATH_LOSS'))
    except:
        model_path_loss = [conf.get('Dataset', 'MODEL_PATH_LOSS')]

    num_objects = len(model_path_loss)
    num_views = len(json.loads(conf.get('Rendering', 'VIEWS')))
    tune_encoder = conf.getboolean('Training','FINETUNE_ENCODER', fallback=False)

    # Prepare object mapping for using GT obj IDs
    obj_mapping = {}
    #obj_mapping[real_obj_id] = obj number in NN output
    obj_mapping[1] = 0
    obj_mapping[2] = 0
    obj_mapping[5] = 1
    obj_mapping[6] = 1
    obj_mapping[10] = 2
    obj_mapping[17] = 3
    obj_mapping[18] = 3
    obj_mapping[19] = 4
    obj_mapping[20] = 4

    # Run prepare our model if needed
    if("Rs_predicted" not in data):

        # Set the cuda device
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        # Initialize the model
        model = Model(num_views=num_views,
                      num_objects=num_objects,
                      finetune_encoder=tune_encoder)
        model.to(device)

        # Load model checkpoint
        checkpoint = torch.load(args.mp)

        # Load model
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()

        # Load and prepare encoder
        encoder = Encoder(args.ep).to(device)
        if(tune_encoder):
            encoder.encoder_dense_MatMul = None
        encoder.eval()

        # Setup the pipeline
        pipeline = Pipeline(encoder, model, device)

    # Prepare renderer if defined
    obj_path = args.op
    if(obj_path is not None):
        obj_model = inout.load_ply(obj_path.replace(".obj",".ply"))
        img_size = 128
        K = np.array([1075.65091572, 0.0, 128.0/2.0,
                      0.0, 1073.90347929, 128.0/2.0,
                      0.0, 0.0, 1.0]).reshape(3,3)
        renderer = Renderer(obj_model, (img_size,img_size), K,
                            surf_color=(1, 1, 1), mode='rgb', random_light=False)
    else:
        renderer = None

    # Store results in a dict
    results = {"scene_id":[],
               "im_id":[],
               "obj_id":[],
               "score":[],
               "R":[],
               "t":[],
               "time":[]}

    # Loop through dataset
    for i,img in enumerate(data["images"]):
        print("Current image: {0}/{1}".format(i+1,len(data["images"])))

        if("Rs_predicted" in data):
            R_predicted = data["Rs_predicted"][i]
        else:

            # Run through model
            predicted_poses = pipeline.process([img])

            # Find best pose
            pose_start = 0
            pose_end = pose_start + 6
            best_pose = 0.0
            R_predicted = None

            # Seperate prdiction into confidences and poses
            confs = predicted_poses[:,:(num_views*num_objects)]
            poses = predicted_poses[:,(num_views*num_objects):]

            # Mask stuff according to ID if outputting multiple objects
            if(num_objects > 1):
                ids = [obj_mapping[data["obj_ids"][i]]]

                idx_mask = torch.tensor(ids)
                confs = confs.reshape(-1,num_objects,num_views)
                confs = confs[torch.arange(confs.size(0)), idx_mask].squeeze(1)

                poses = poses.reshape(-1,num_objects,num_views*6)
                poses = poses[torch.arange(poses.size(0)), idx_mask].squeeze(1)

            for k in range(num_views):
                # Extract current pose and move to next one
                curr_pose = poses[:,pose_start:pose_end]
                Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
                Rs_predicted = Rs_predicted.detach().cpu().numpy()[0]
                pose_start = pose_end
                pose_end = pose_start + 6

                conf = confs[:,k].detach().cpu().numpy()[0]
                if(conf > best_pose):
                    R_predicted = Rs_predicted
                    best_pose = conf

            # Invert xy axes
            xy_flip = np.eye(3, dtype=np.float)
            xy_flip[0,0] = -1.0
            xy_flip[1,1] = -1.0
            R_predicted = R_predicted.dot(xy_flip)

            # Inverse rotation matrix
            R_predicted = np.transpose(R_predicted)

        results["scene_id"].append(data["scene_ids"][i])
        results["im_id"].append(data["img_ids"][i])
        results["obj_id"].append(data["obj_ids"][i])
        results["score"].append(-1)
        results["R"].append(arr2str(R_predicted))
        results["t"].append(arr2str(data["ts"][i]))
        results["time"].append(-1)

        if(renderer is None):
            visualize = False

        if(visualize):
            t_gt = np.array(data["ts"][i])
            t = np.array([0,0,t_gt[2]])

            # Render predicted pose
            R_predicted = correct_trans_offset(R_predicted,t_gt)
            ren_predicted = renderer.render(R_predicted, t)

            # Render groundtruth pose
            R_gt = data["Rs"][i]
            R_gt = correct_trans_offset(R_gt,t_gt)
            ren_gt = renderer.render(R_gt, t)

            cv2.imshow("gt render", np.flip(ren_gt,axis=2))
            cv2.imshow("predict render", np.flip(ren_predicted,axis=2))

            cv2.imshow("input image", np.flip(img,axis=2))
            if("codebook_images" in data):
                cv2.imshow("codebook image",
                           np.flip(data["codebook_images"][i],axis=2))

            print(ren_gt.shape)
            print(ren_predicted.shape)
            print(img.shape)
            numpy_horizontal_concat = np.concatenate((np.flip(ren_gt,axis=2), np.flip(ren_predicted,axis=2), np.flip(img,axis=2)), axis=1)
            cv2.imshow("gt - prediction - input", numpy_horizontal_concat)
            key = cv2.waitKey(0)
            if(key == ord("q")):
                exit()
                visualize = False
                #break
                continue

    # Save to CSV
    output_path = args.o
    print("Saving to: ", output_path)
    with open(output_path, "w") as f:
        col_names = list(results.keys())
        w = csv.DictWriter(f, results.keys())
        w.writeheader()
        num_lines = len(results[col_names[0]])

        for i in np.arange(num_lines):
            row_dict = {}
            for c in col_names:
                row_dict[c] = results[c][i]
            w.writerow(row_dict)

if __name__ == '__main__':
    main()
