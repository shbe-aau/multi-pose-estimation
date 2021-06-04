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
    def __init__(self, obj_paths, device, batch_size=12, faces_per_pixel=16,
                 render_method="silhouette", image_size=256, norm_verts=False):
        self.batch_size = batch_size
        self.faces_per_pixel = faces_per_pixel
        self.batch_indeces = np.arange(self.batch_size)
        self.obj_paths = obj_paths
        self.device = device
        self.method = render_method
        self.image_size = image_size
        self.points = None
        self.norm_verts = False #norm_verts

        # Setup batch of meshes
        self.vertices, self.faces, self.textures = self.initMeshes()

        # Initialize the renderer
        self.renderer = self.initRender(image_size=image_size, method=self.method)

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

        batch_T = torch.tensor(np.stack(ts), device=self.device, dtype=torch.float32)
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


    def initRender(self, method, image_size):
        render_size_width = 400
        render_size_height = 400

        org_size_width = 720.0
        org_size_height = 540.0

        render_size_width = int(render_size_width*(org_size_width/org_size_height))

        fx = 1075.65091572 * (render_size_width/org_size_width)
        fy = 1073.90347929 * (render_size_height/org_size_height)

        # These values should be half of render_size_width/height during training
        px = render_size_width/2
        py = render_size_height/2

        if False:  # local testing override for using a non-centered principal point
            px = 367.06888344 * (render_size_width/org_size_width)
            py = 247.72159802 * (render_size_height/org_size_height)

        cameras = PerspectiveCameras(device=self.device,
                                     focal_length=((fx, fy),),
                                     principal_point=((px, py),),
                                     image_size=((render_size_width,render_size_height),))
        K = cameras.get_projection_transform()
        print(K[0].get_matrix())

        if(method=="soft-silhouette"):
            blend_params = BlendParams(sigma=1e-7, gamma=1e-7)

            raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius=np.log(1. / 1e-7 - 1.) * blend_params.sigma,
                faces_per_pixel=self.faces_per_pixel
            )

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftSilhouetteShader(blend_params=blend_params)
            )
        elif(method=="hard-silhouette"):
            blend_params = BlendParams(sigma=1e-7, gamma=1e-7)

            raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius=np.log(1. / 1e-7 - 1.) * blend_params.sigma,
                faces_per_pixel=1
            )

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftSilhouetteShader(blend_params=blend_params)
            )
        elif(method=="soft-depth"):
            # Soft Rasterizer - from https://github.com/facebookresearch/pytorch3d/issues/95
            #blend_params = BlendParams(sigma=1e-7, gamma=1e-7)
            blend_params = BlendParams(sigma=1e-3, gamma=1e-4)
            raster_settings = RasterizationSettings(
                image_size=image_size,
                #blur_radius= np.log(1. / 1e-7 - 1.) * blend_params.sigma,
                blur_radius= np.log(1. / 1e-3 - 1.) * blend_params.sigma,
                faces_per_pixel=self.faces_per_pixel
            )

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftDepthShader(blend_params=blend_params)
            )
        elif(method=="hard-depth"):
            raster_settings = RasterizationSettings(
                image_size=(render_size_height, render_size_width), #OBS! TODO: change back order when bug fixed
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
        elif(method=="blurry-depth"):
            # Soft Rasterizer - from https://github.com/facebookresearch/pytorch3d/issues/95
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
            raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius= np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                faces_per_pixel=self.faces_per_pixel
            )

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftDepthShader(blend_params=blend_params)
            )
        elif(method=="soft-phong"):
            blend_params = BlendParams(sigma=1e-3, gamma=1e-3)

            raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius= np.log(1. / 1e-3 - 1.) * blend_params.sigma,
                faces_per_pixel=self.faces_per_pixel
            )

            # lights = DirectionalLights(device=self.device,
            #                            ambient_color=[[0.25, 0.25, 0.25]],
            #                            diffuse_color=[[0.6, 0.6, 0.6]],
            #                            specular_color=[[0.15, 0.15, 0.15]],
            #                            direction=[[0.0, 1.0, 0.0]])

            lights = DirectionalLights(device=self.device,
                                       direction=[[0.0, 1.0, 0.0]])

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(device=self.device,
                                       blend_params = blend_params,
                                       lights=lights)
            )

        elif(method=="hard-phong"):
            blend_params = BlendParams(sigma=1e-8, gamma=1e-8)

            raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius=0.0,
                faces_per_pixel=1
            )

            lights = DirectionalLights(device=self.device,
                                       ambient_color=[[0.25, 0.25, 0.25]],
                                       diffuse_color=[[0.6, 0.6, 0.6]],
                                       specular_color=[[0.15, 0.15, 0.15]],
                                       direction=[[-1.0, -1.0, 1.0]])
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=HardPhongShader(device=self.device, lights=lights)
            )


        else:
            print("Unknown render method!")
            return None
        return renderer
