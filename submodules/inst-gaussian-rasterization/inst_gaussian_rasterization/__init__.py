
from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D_template,
    scaling_template,
    rotation_template,
    shs_template,
    opacity_template,
    instance_transforms,
    xyz_offsets,
    scaling_offsets,
    rotation_offsets,
    shs_offsets,
    opacity_offsets,
    colors_precomp,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D_template,
        scaling_template,
        rotation_template,
        shs_template,
        opacity_template,
        instance_transforms,
        xyz_offsets,
        scaling_offsets,
        rotation_offsets,
        shs_offsets,
        opacity_offsets,
        colors_precomp,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D_template,
        scaling_template,
        rotation_template,
        shs_template,
        opacity_template,
        instance_transforms,
        xyz_offsets,
        scaling_offsets,
        rotation_offsets,
        shs_offsets,
        opacity_offsets,
        colors_precomp,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D_template,
            scaling_template,
            rotation_template,
            shs_template,
            opacity_template,
            instance_transforms,
            xyz_offsets,
            scaling_offsets,
            rotation_offsets,
            shs_offsets,
            opacity_offsets,
            colors_precomp,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_inst_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered

        return color, radii, invdepths


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, 
                means3D_template,
                scaling_template,
                rotation_template,
                shs_template,
                opacity_template,
                instance_transforms,
                xyz_offsets,
                scaling_offsets,
                rotation_offsets,
                shs_offsets,
                opacity_offsets,
                colors_precomp = None, 
                cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs_template is None and colors_precomp is None) or (shs_template is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scaling_template is None or rotation_template is None) and cov3D_precomp is None) or ((scaling_template is not None or rotation_template is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs_template is None:
            shs_template = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scaling_template is None:
            scaling_template = torch.Tensor([])
        if rotation_template is None:
            rotation_template = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D_template,
            scaling_template,
            rotation_template,
            shs_template,
            opacity_template,
            instance_transforms,
            xyz_offsets,
            scaling_offsets,
            rotation_offsets,
            shs_offsets,
            opacity_offsets,
            colors_precomp,
            cov3D_precomp,
            raster_settings, 
        )
