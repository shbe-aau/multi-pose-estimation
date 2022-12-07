import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pylab import cm
import cv2 as cv

from os import sys, path
sys.path.append('../..')

from utils.utils import *
from utils.tools import *

from pytorch3d.renderer.cameras import FoVPerspectiveCameras, PerspectiveCameras
from CustomRenderers import *
# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, SoftPhongShader, PointLights, DirectionalLights, HardPhongShader
)

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes, list_to_padded
from pytorch3d.renderer.mesh.textures import TexturesVertex

def setup_renderer(device, render_size_width, render_size_height, center_focal_point=False):
    # From T-LESS test scene groundtruth
    fx = 1075.65091572
    fy = 1073.90347929

    px = 373.06888344
    py = 301.72159802

    if(center_focal_point):
        px = render_size_width/2.0
        py = render_size_height/2.0

    cameras = PerspectiveCameras(device=device,
                            focal_length=((fx, fy),),
                            principal_point=((px, py),),
                            image_size=((render_size_width, render_size_height),))


    raster_settings = RasterizationSettings(
        image_size=(render_size_height, render_size_width),
        blur_radius= 0,
        faces_per_pixel= 1
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardDepthShader()
    )
    return renderer

def opengl2pytorch3d(R,t):
    # Convert R matrix from opengl format to pytorch
    # for rendering only!
    xy_flip = np.eye(3, dtype=np.float)
    xy_flip[0,0] = -1.0
    xy_flip[1,1] = -1.0
    R_pytorch = np.transpose(R)
    R_pytorch = np.dot(R_pytorch,xy_flip)
    t_pytorch = t*np.array([-1.0,-1.0,1.0])
    return R_pytorch, t_pytorch


def render_object(R, t, obj_id=10, render_width=720, render_height=540, center_obj=False):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Load CAD and prepare 3D model
    obj_path = "obj_{1:02d}.obj".format(obj_id, obj_id)
    # Load the obj and ignore the textures and materials.
    verts, faces_idx, _ = load_obj(obj_path)
    facs = faces_idx.verts_idx

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)  # (V, 3)

    # Load meshes based on object ids
    batch_verts_rgb = list_to_padded([verts_rgb])
    batch_textures = TexturesVertex(verts_features=batch_verts_rgb.to(device))
    batch_verts=[verts.to(device)]
    batch_faces=[facs.to(device)]

    mesh = Meshes(
        verts=batch_verts,
        faces=batch_faces,
        textures=batch_textures
    )

    # Render the darn thing!
    renderer = setup_renderer(device, render_width,
                              render_height, center_focal_point=center_obj)

    # Render using pytorch3d
    Rs = [R]
    ts = [t]
    batch_R = torch.tensor(np.stack(Rs), device=device, dtype=torch.float32)
    batch_T = torch.tensor(np.stack(ts), device=device, dtype=torch.float32) # Bx3
    pytorch_depth = renderer(meshes_world=mesh, R=batch_R, T=batch_T)
    return pytorch_depth[0].detach().cpu().numpy()

def offset_img(img, t):
    # Center depth image around translation distance
    non_zero = img[img > 0]
    offset = np.median(non_zero)
    non_zero_std = np.std(non_zero)
    non_zero_mean = np.mean(non_zero)
    img_offset = img
    img_offset[img_offset == -1] = (non_zero_mean - 2*non_zero_std)
    #avg_dist = np.linalg.norm(t)
    #img_offset = img
    #img_offset[img_offset == -1] = 1050.0 #avg_dist
    return img_offset

def save_img(save_path, img, t=None):

    # Plot and save using opencv
    colormap = cm.jet
    #colorized = colormap((img-np.min(img))/(np.max(img)-np.min(img)))
    colorized = colormap((img-950)/(1050-950))
    cv2.imwrite(save_path, colorized*255)

def correct_R(R, t_est):
    # correcting the rotation matrix
    # the codebook consists of centered object views, but the test image crop is not centered
    # we determine the rotation that preserves appearance when translating the object

    # SHBE fix - x and y translation should be negative/inverted like opengl2pytorch conversion?
    t_est = t_est*np.array([-1.0,-1.0,1.0])

    d_alpha_y = np.arctan(t_est[0]/np.sqrt(t_est[2]**2+t_est[1]**2))
    d_alpha_x = - np.arctan(t_est[1]/t_est[2])
    R_corr_x = np.array([[1,0,0],
                         [0,np.cos(d_alpha_x),-np.sin(d_alpha_x)],
                         [0,np.sin(d_alpha_x),np.cos(d_alpha_x)]])
    R_corr_y = np.array([[np.cos(d_alpha_y),0,np.sin(d_alpha_y)],
                         [0,1,0],
                         [-np.sin(d_alpha_y),0,np.cos(d_alpha_y)]])
    R_corrected = np.dot(R_corr_y,np.dot(R_corr_x,R))
    return R_corrected


def test_pose(R_in, t_in, label="test_pose1", gt_img_path=None):
    center_render_size = 128

    # Render with original rotation and translation
    R1, t1 = opengl2pytorch3d(R_in,t_in)
    img = render_object(R1, t1)
    img = offset_img(img, t1)
    #save_img(label+"_original_pose.png", img_offset)
    img_org = img

    # Centered render without corrected rotation
    t_center = np.array([0.0, 0.0, t_in[-1]])
    R2, t2 = opengl2pytorch3d(R_in, t_center)
    img = render_object(R2, t2, render_width=center_render_size,
                        render_height=center_render_size, center_obj=True)
    img = offset_img(img, t2)
    #save_img(label+"_centered_no_correction.png", img_offset)
    img_centered = img

    # Centered with corrected rotation
    R_corrected = correct_R(R_in, t_in)
    R3, t3 = opengl2pytorch3d(R_corrected, t_center)
    img = render_object(R3, t3, render_width=center_render_size,
                        render_height=center_render_size, center_obj=True)
    img = offset_img(img, t3)
    #save_img(label+"_centered_with_correction.png", img_offset)
    img_centered_corrected = img

    # Save in same plot
    fig = plt.figure(figsize=(32,16))

    if(gt_img_path is None):
        plt.subplot(1, 3, 1)
        plt.imshow((img_org))#.astype(np.uint8))
        plt.title("full render")

        plt.subplot(1, 3, 2)
        plt.imshow((img_centered))#.astype(np.uint8))
        plt.title("small render")

        plt.subplot(1, 3, 3)
        plt.imshow((img_centered_corrected))#.astype(np.uint8))
        plt.title("small render - corrected")
    else:
        org_img = cv2.imread(gt_img_path)
        plt.subplot(2, 2, 1)
        plt.imshow((org_img))#.astype(np.uint8))
        plt.title("original image")

        plt.subplot(2, 2, 2)
        plt.imshow((img_org))#.astype(np.uint8))
        plt.title("full render")

        plt.subplot(2, 2, 3)
        plt.imshow((img_centered))#.astype(np.uint8))
        plt.title("small render")

        plt.subplot(2, 2, 4)
        plt.imshow((img_centered_corrected))#.astype(np.uint8))
        plt.title("small render - corrected")

    fig.tight_layout()
    fig.savefig(label+".png", dpi=fig.dpi)
    plt.close()


# Test pose 1 - img 173 - tless test scene 5
R = np.array([-0.99983527, -0.01806117, -0.00180892,
             -0.01305949, 0.78496276, -0.61940498,
             0.01260709, -0.61927948, -0.78506983]).reshape(3,3)
t = np.array([35.34602246, -91.73294348, 822.05119118])
test_pose(R, t, label="test1", gt_img_path="scene5-0173.png")


# Test pose 2 - img 224 - tless test scene 5
R = np.array([0.28054173, 0.95984022, 0.00188783,
              0.66742699, -0.19366099, -0.71905247,
              -0.68981014, 0.20298377, -0.69495254]).reshape(3,3)
t = np.array([-110.08269362, -36.32032878, 778.81290559])
test_pose(R, t, label="test2", gt_img_path="scene5-0224.png")


# Test pose 3 - img 267 - tless test scene 16
R = np.array([-0.38061412, -0.92458875, 0.01638160,
              -0.62964205, 0.24614153, -0.73686100,
              0.67726103, -0.29077422, -0.67584500]).reshape(3,3)
t = np.array([117.04503082, -12.69796681, 765.34612140])
test_pose(R, t, label="test3", gt_img_path="scene16-0267.png")
