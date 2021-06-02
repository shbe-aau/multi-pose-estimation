import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2 as cv

from utils.utils import *
from utils.tools import *

from utils.pytless import inout, misc
from vispy import app, gloo
from utils.pytless.renderer import Renderer

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

# ------------ Params ------------------------
obj_id = 10
render_size_x = 400
render_size_y = 400

org_size_x = 720.0
org_size_y = 540.0

render_size_x = int(render_size_x*(org_size_x/org_size_y))

fx = 1075.65091572 * (render_size_x/org_size_x)
fy = 1073.90347929 * (render_size_y/org_size_y)

px = 367.06888344 * (render_size_x/org_size_x)
py = 247.72159802 * (render_size_y/org_size_y)


#  "42": {"cam_K": [1075.65091572, 0.0, 367.06888344, 0.0, 1073.90347929, 247.72159802, 0.0, 0.0, 1.0], "cam_R_w2c": [-0.529601, 0.848203, 0.00860459, 0.810945, 0.50926, -0.288136, -0.24878, -0.145619, -0.957551], "cam_t_w2c": [-5.19015, 23.2188, 789.59], "depth_scale": 0.1, "elev": 75, "mode": 0},

#{"cam_R_m2c": [-0.78604536, -0.61810859, 0.00860459, -0.59386273, 0.7512021, -0.288136, 0.17163531, -0.23159815, -0.957551], "cam_t_m2c": [58.84511603, -90.2855017, 790.53840201], "obj_id": 10}

K = np.array([fx, 0, px,
              0, fy, py,
              0, 0, 1]).reshape(3,3)

#R = np.array([0.6634139, -0.5566704,  0.5000000,
#              0.7350241,  0.6099232, -0.2961981,
#              -0.1400768,  0.5640140,  0.8137977]).reshape(3,3) # 20,30,40 degrees x,y,z
R = np.array([-0.78604536, -0.61810859, 0.00860459,
              -0.59386273, 0.7512021, -0.288136,
              0.17163531, -0.23159815, -0.957551]).reshape(3,3)

#t = np.array([0.0, 0.0, 600.0])
t = np.array([58.84511603, -90.2855017, 790.53840201])

# -------- OpenGL render ----------------------
obj_path = "./data/tless-obj{0:02d}/cad/obj_{1:02d}.obj".format(obj_id, obj_id)
model = inout.load_ply(obj_path.replace(".obj",".ply"))

renderer = Renderer(model, (render_size_x,render_size_y),
                    K, surf_color=(1, 1, 1), mode='rgb')

# # Convert R matrix from pytorch to opengl format
# # for rendering only!
# xy_flip = np.eye(3, dtype=np.float)
# xy_flip[0,0] = -1.0
# xy_flip[1,1] = -1.0
# R_opengl = np.dot(R,xy_flip)
# R_opengl = np.transpose(R_opengl)


opengl_rgb = renderer.render(R, t)

#opengl_rgb = np.swapaxes(opengl_rgb, 0, 1)
#cv2.imwrite("opengl_rgb.png", np.flip(opengl_rgb,axis=2))


# -------- PyTorch3D render ----------------------
device = torch.device("cuda:0")
torch.cuda.set_device(device)

#temp = render_size_x
#render_size_x = render_size_y
#render_size_y = temp

cameras = PerspectiveCameras(device=device,
                             focal_length=((fx, fy),),
                             principal_point=((px, py),),
                             image_size=((render_size_x, render_size_y),))

K = cameras.get_projection_transform()
print(K[0].get_matrix())
K = K[0].get_matrix()

#new_K = torch.t(K[0].get_matrix()[0]).unsqueeze(0)
#print(new_K)

cameras = PerspectiveCameras(device=device, K=K)

K = cameras.get_projection_transform()
print(K[0].get_matrix())

raster_settings = RasterizationSettings(
    image_size=(render_size_x, render_size_y),
    blur_radius= 0,
    faces_per_pixel= 20
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardDepthShader()
)


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


# Convert R matrix from opengl format to pytorch
# for rendering only!
xy_flip = np.eye(3, dtype=np.float)
xy_flip[0,0] = -1.0
xy_flip[1,1] = -1.0
R_pytorch = np.transpose(R)
R_pytorch = np.dot(R_pytorch,xy_flip)

Rs = [R_pytorch]
#t = t*np.array([-1.0,-1.0,1.0])
t = np.array([t[1],t[0],t[2]])
ts = [t]
batch_R = torch.tensor(np.stack(Rs), device=device, dtype=torch.float32)
batch_T = torch.tensor(np.stack(ts), device=device, dtype=torch.float32) # Bx3
pytorch_depth = renderer(meshes_world=mesh, R=batch_R, T=batch_T)
#print(pytorch_depth)
#cv2.imwrite("pytorch_depth.png", pytorch_depth[0].detach().cpu().numpy())




fig = plt.figure(figsize=(12,5))
# Plot AE input
plt.subplot(1, 3, 1)
plt.imshow((opengl_rgb*255).astype(np.uint8))
plt.title("opengl rgb")

# Plot depth map render from ground truth
swapped_img = np.swapaxes(pytorch_depth[0].detach().cpu().numpy(),0,1)
plt.subplot(1, 3, 2)
plt.imshow(swapped_img)#,
plt.title("pytorch3d depth")

# Plot depth map render from prediction
plt.subplot(1, 3, 3)
plt.imshow((swapped_img+(opengl_rgb*255).astype(np.uint8)[:,:,2]*4)/2)
plt.title("diff")


fig.tight_layout()

fig.savefig("test-renders.png", dpi=fig.dpi)
plt.close()
