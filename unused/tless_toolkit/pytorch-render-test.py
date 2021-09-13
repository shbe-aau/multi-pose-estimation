import cv2
import numpy as np
import torch

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes, list_to_padded
from pytorch3d.renderer.mesh.textures import TexturesVertex

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    HardPhongShader, PointLights, DirectionalLights
)

model_path = "/media/shbe/data/share-to-docker/autoencoder_ws/data/T-LESS/models_cad/obj_19.ply"

# Get intrinsic camera parameters and object pose
K = np.array([1075.65091572, 0.0, 396.06888344, 0.0, 1073.90347929, 235.72159802, 0.0, 0.0, 1.0]).reshape(3,3)

R = np.array([-0.04952228, 0.64490459, 0.76265755, 0.33259176, -0.70936509, 0.62143685, 0.94176932, 0.28442848, -0.17936052]).reshape(3,3)
t = np.array([69.35163193, -101.13001501, 737.73502139])
im_size = (720, 540)

# Inverse rotation matrix
R = np.transpose(R)

# Invert xy axes
xy_flip = np.eye(3, dtype=np.float)
xy_flip[0,0] = -1.0
xy_flip[1,1] = -1.0
R = R.dot(xy_flip)

device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj(model_path)
faces = faces_idx.verts_idx


# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
mesh = Meshes(
    verts=[verts.to(device)],
    faces=[faces.to(device)],
    textures=textures
)

# Initialize an OpenGL perspective camera.
cameras = OpenGLPerspectiveCameras(device=device,fov=25.0, aspect_ratio=1.1)

# We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=540,
    blur_radius=0.0,
    faces_per_pixel=1,
    bin_size=0
)
# We can add a point light in front of the object.
#lights = PointLights(device=device, location=((-1.0, -1.0, -2.0),))
#"ambient_color", "diffuse_color", "specular_color"
# 'ambient':0.4,'diffuse':0.8, 'specular':0.3
lights = DirectionalLights(device=device,
                     ambient_color=[[0.25, 0.25, 0.25]],
                     diffuse_color=[[0.6, 0.6, 0.6]],
                     specular_color=[[0.15, 0.15, 0.15]],
                     direction=[[-1.0, -1.0, 1.0]])
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, lights=lights)
)

R = torch.tensor(R, device=device).unsqueeze(0)
T = torch.tensor(t, device=device).unsqueeze(0)
image = phong_renderer(meshes_world=mesh, R=R, T=T)
image = image.cpu().numpy()[0,:,:,:3]*255
cv2.imwrite("pytorch-render.png", image.astype(np.uint8))

print(cameras.get_projection_transform().get_matrix())
