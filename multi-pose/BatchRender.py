import torch
import numpy as np
import torch.nn as nn

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes, list_to_padded
from pytorch3d.renderer.mesh.textures import TexturesVertex

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, SoftPhongShader, PointLights, DirectionalLights, HardPhongShader
)
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, PerspectiveCameras

from pytorch3d.ops import sample_points_from_meshes
from CustomRenderers import *

from utils.utils import *
from utils.tools import *

class BatchRender:
    def __init__(self, obj_paths, device, camera_params, batch_size=12, faces_per_pixel=16,
                 render_method="silhouette", norm_verts=False):
        self.batch_size = batch_size
        self.faces_per_pixel = faces_per_pixel
        self.batch_indeces = np.arange(self.batch_size)
        self.obj_paths = obj_paths
        self.device = device
        self.method = render_method
        self.points = None
        self.norm_verts = False #norm_verts

        # Setup batch of meshes
        self.vertices, self.faces, self.textures = self.initMeshes()

        # Initialize the renderer
        self.renderer = self.initRender(camera_params, method=self.method)

    def renderBatch(self, Rs, ts, ids=[]):
        if(type(Rs) is list):
            batch_R = torch.tensor(np.stack(Rs), device=self.device, dtype=torch.float32)
        else:
            batch_R = Rs
        if(type(ts) is list):
            batch_T = torch.tensor(np.stack(ts), device=self.device, dtype=torch.float32) # Bx3
        else:
            batch_T = ts

        if(len(ids) == 0):
            # No ids specified, assuming one object only
            ids = [0 for r in Rs]

        # Load meshes based on object ids
        batch_verts_rgb = list_to_padded([self.textures[i] for i in ids])
        batch_textures = TexturesVertex(verts_features=batch_verts_rgb.to(self.device))
        batch_verts=[self.vertices[i].to(self.device) for i in ids]
        batch_faces=[self.faces[i].to(self.device) for i in ids]

        mesh = Meshes(
            verts=batch_verts,
            faces=batch_faces,
            textures=batch_textures
        )

        images = self.renderer(meshes_world=mesh, R=batch_R, T=batch_T)
        if(self.method == "soft-silhouette"):
            images = images[..., 3]
        elif(self.method == "hard-silhouette"):
            images = images[..., 3]
        elif(self.method == "hard-phong"):
            images = images[..., :3]
        elif(self.method == "soft-phong"):
            images = images[..., :3]
        elif(self.method == "soft-depth"):
            images = images #[..., 0] #torch.mean(images, dim=3)
        elif(self.method == "hard-depth"):
            images = images #torch.mean(images, dim=3)
        elif(self.method == "blurry-depth"):
            images = torch.mean(images, dim=3)
        return images

    def initMeshes(self):
        textures = []
        vertices = []
        faces = []
        for p in self.obj_paths:
            # Load the obj and ignore the textures and materials.
            verts, faces_idx, _ = load_obj(p)
            facs = faces_idx.verts_idx

            # Normalize vertices
            # such that all objects measures 100 mm
            # along the biggest dimension (x,y,z)
            if(self.norm_verts):
                center = verts.mean(0)
                verts_normed = verts - center
                scale = max(verts_normed.abs().max(0)[0])
                verts = (verts / scale)*100.0

            # Initialize each vertex to be white in color.
            verts_rgb = torch.ones_like(verts)  # (V, 3)
            vertices.append(verts)
            textures.append(verts_rgb)
            faces.append(facs)
        return vertices, faces, textures


    def initRender(self, camera_params, method):
        render_width = camera_params['render_width']
        render_height = camera_params['render_height']

        cameras = PerspectiveCameras(device=self.device,
                                     focal_length=((camera_params['fx'], camera_params['fy']),),
                                     principal_point=((camera_params['px'], camera_params['py']),),
                                     image_size=((render_width,render_height),))

        if(method=="hard-depth"):
            raster_settings = RasterizationSettings(
                image_size=(render_height, render_width), #OBS! TODO: change back order when bug fixed
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
        else:
            print("Unknown render method!")
            return None
        return renderer
