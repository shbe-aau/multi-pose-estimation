import torch
import torch.nn as nn
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.blending import sigmoid_alpha_blend

class HardDepthShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        image = fragments.zbuf[..., 0]
        return image


class SoftDepthShader(nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        colors = torch.stack([fragments.zbuf,fragments.zbuf,fragments.zbuf]).permute(1,2,3,4,0)
        images = softmax_rgb_blend(colors, fragments, self.blend_params)
        print(images.shape)
        return images[...,2]
