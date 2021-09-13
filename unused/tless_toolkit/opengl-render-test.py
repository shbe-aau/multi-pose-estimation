#!/usr/local/bin/python3
# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# A script to render 3D object models into the training images. The models are
# rendered at the 6D poses that are associated with the training images.
# The visualizations are saved into the folder specified by "output_dir".

from pytless import inout, renderer, misc
import os
import numpy as np
import imageio

#model_path = "/media/shbe/data/share-to-docker/autoencoder_ws/data/T-LESS/models_cad/obj_19.ply"

model_path = "/shared-folder/AugmentedAutoencoder/pytorch3d/data/t-less-obj06/cad/obj_06.ply"

# Get intrinsic camera parameters and object pose
K = np.array([1075.65091572, 0.0, 396.06888344, 0.0, 1073.90347929, 235.72159802, 0.0, 0.0, 1.0]).reshape(3,3)

R = np.array([-0.04952228, 0.64490459, 0.76265755, 0.33259176, -0.70936509, 0.62143685, 0.94176932, 0.28442848, -0.17936052]).reshape(3,3)
t_est = np.array([-69.35163193, 101.13001501, 737.73502139])
t = np.array([0, 0, 700.0])

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
R = R_corrected

im_size = (720, 540)

model = inout.load_ply(model_path)
surf_color = (1, 1, 1)
ren_rgb = renderer.render(model, im_size, K, R, t,
                          surf_color=surf_color, mode='rgb')

print(ren_rgb.shape)
imageio.imwrite("opengl-render.png", ren_rgb.astype(np.uint8))
