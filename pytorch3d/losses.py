import torch
import numpy as np
from utils.utils import *
from utils.tools import *
import configparser

from pytorch3d.renderer import look_at_rotation

dbg_losses = False

def dbg(message, flag):
    if flag:
        print(message)

def Loss(predicted_poses,
         gt_poses,
         renderer,
         ts,
         ids=[],
         views=None,
         config=None,
         fixed_gt_images=None):
    Rs_gt = torch.tensor(np.stack(gt_poses), device=renderer.device,
                            dtype=torch.float32)
    if config is None:
        config = configparser.ConfigParser()

    loss_method = config.get('Training', 'LOSS', fallback='vsd-union')
    pose_rep = config.get('Training', 'POSE_REPRESENTATION', fallback='6d-pose')

    pose_rep_func = None
    if fixed_gt_images is None:
        if(pose_rep == '6d-pose'):
            pose_rep_func = compute_rotation_matrix_from_ortho6d
        #elif(pose_rep == 'rot-mat'): # Deprecated
            #batch_size = predicted_poses.shape[0]
            #Rs_predicted = predicted_poses.view(batch_size, 3, 3)
        elif(pose_rep == 'quat'):
            pose_rep_func = compute_rotation_matrix_from_quaternion
        #elif(pose_rep == 'euler'): # Deprecated
            #Rs_predicted = look_at_rotation(predicted_poses).to(renderer.device)
        elif(pose_rep == 'axis-angle'):
            pose_rep_func = compute_rotation_matrix_from_axisAngle
        else:
            print("Unknown pose representation specified: ", pose_rep)
            return -1.0
    else: # this version is for using loss with prerendered ref image and regular rot matrix for predicted pose
        gt_imgs = fixed_gt_images

    if(loss_method=="vsd-union"):
        depth_max = config.getfloat('Loss_parameters', 'DEPTH_MAX', fallback=30.0)
        pose_max = config.getfloat('Loss_parameters', 'POSE_MAX', fallback=40.0)
        num_views = len(views)
        gamma = config.getfloat('Loss_parameters', 'GAMMA', fallback=1.0 / num_views)
        pose_start = num_views
        pose_end = pose_start + 6

        # Prepare gt images
        gt_images = []
        predicted_images = []
        gt_imgs = renderer.renderBatch(Rs_gt, ts, ids)

        losses = []
        confs = predicted_poses[:,:num_views]
        prev_poses = []
        pose_losses = []
        for i,v in enumerate(views):
            # Extract current pose and move to next one
            if fixed_gt_images is None:
                curr_pose = predicted_poses[:,pose_start:pose_end]
                Rs_predicted = pose_rep_func(curr_pose)
            else:
                pose_matrix = predicted_poses[:,1:].reshape(1,3,3)
                Rs_predicted = pose_matrix
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts, ids)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Visiblity mask
            mask_gt = gt_imgs != 0
            mask_pd = imgs != 0
            mask_union = torch.zeros_like(gt_imgs)
            mask_union[mask_gt] = 1.0
            mask_union[mask_pd] = 1.0

            # Calculate loss
            diff = torch.abs(gt_imgs - imgs)
            diff = torch.clamp(diff, 0.0, depth_max)/depth_max
            batch_loss = torch.sum(diff*mask_union, dim=(1,2))/torch.sum(mask_union, dim=(1,2))

            batch_loss = (batch_loss*confs[:,i] + gamma*batch_loss)/2.0
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)



        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        if(num_views == 1):
            pose_losses = torch.zeros_like(losses)
        else:
            pose_losses = torch.cat(pose_losses, dim=1)
        pose_losses = torch.mean(pose_losses, dim=1)
        depth_losses = torch.sum(losses, dim=1)

        dbg("depth loss {}".format(torch.mean(depth_losses)), dbg_losses)
        dbg("pose loss  {}".format(torch.mean(pose_losses)), dbg_losses)

        batch_loss = depth_losses + pose_losses
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss)
        return loss, batch_loss, gt_imgs, predicted_imgs

    if(loss_method=="vsd-union-multiobject"):
        print(predicted_poses.shape)

        depth_max = config.getfloat('Loss_parameters', 'DEPTH_MAX', fallback=30.0)
        pose_max = config.getfloat('Loss_parameters', 'POSE_MAX', fallback=40.0)
        num_views = len(views)
        gamma = config.getfloat('Loss_parameters', 'GAMMA', fallback=1.0 / num_views)
        num_objects = config.getint('Training', 'NUM_OBJECTS_OUTPUT', fallback=1)
        pose_start = 0
        pose_end = pose_start + 6

        # Prepare gt images
        gt_images = []
        predicted_images = []
        gt_imgs = renderer.renderBatch(Rs_gt, ts, ids)

        losses = []
        confs = predicted_poses[:,:(num_views*num_objects)]
        poses = predicted_poses[:,(num_views*num_objects):]

        # Mask stuff according to ID if outputting multiple objects
        print(poses.shape)
        if(num_objects > 1):
            idx_mask = torch.tensor(ids)
            confs = confs.reshape(-1,num_objects,num_views)
            confs = confs[torch.arange(confs.size(0)), idx_mask].squeeze(1)

            poses = poses.reshape(-1,num_objects,num_views*6)
            poses = poses[torch.arange(poses.size(0)), idx_mask].squeeze(1)
            print(poses.shape)
            print("--------------")

        prev_poses = []
        pose_losses = []
        for i,v in enumerate(views):
            # Extract current pose and move to next one
            if fixed_gt_images is None:
                curr_pose = poses[:,pose_start:pose_end]
                Rs_predicted = pose_rep_func(curr_pose)
            else:
                pose_matrix = poses[:,1:].reshape(1,3,3)
                Rs_predicted = pose_matrix
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts, ids)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Visiblity mask
            mask_gt = gt_imgs != 0
            mask_pd = imgs != 0
            mask_union = torch.zeros_like(gt_imgs)
            mask_union[mask_gt] = 1.0
            mask_union[mask_pd] = 1.0

            # Calculate loss
            diff = torch.abs(gt_imgs - imgs)
            diff = torch.clamp(diff, 0.0, depth_max)/depth_max
            batch_loss = torch.sum(diff*mask_union, dim=(1,2))/torch.sum(mask_union, dim=(1,2))

            batch_loss = (batch_loss*confs[:,i] + gamma*batch_loss)/2.0
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)

        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        if(num_views == 1):
            pose_losses = torch.zeros_like(losses)
        else:
            pose_losses = torch.cat(pose_losses, dim=1)
        pose_losses = torch.mean(pose_losses, dim=1)
        depth_losses = torch.sum(losses, dim=1)

        dbg("depth loss {}".format(torch.mean(depth_losses)), dbg_losses)
        dbg("pose loss  {}".format(torch.mean(pose_losses)), dbg_losses)

        batch_loss = depth_losses + pose_losses
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss)
        return loss, batch_loss, gt_imgs, predicted_imgs


    print("Unknown loss specified")
    return -1, None, None, None
