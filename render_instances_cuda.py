import torch
import numpy as np
import os
from gaussian_renderer import render, network_gui, instanced_render
import sys
from scene import InstScene
from scene.inst_gaussian_model import InstGaussianModel
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image


SCENE_NAME = "hkust-gz"


def sizeof_tensor(t: torch.Tensor):
    return t.numel() * t.element_size()

def compress(tensor: torch.Tensor, threshold: float):
    if torch.all(tensor == 0) or torch.all(tensor.abs() < threshold):
        print("All zero tensor, no need to compress.")
        
    flattened = tensor.view(tensor.shape[0], -1)
    mask = ~(flattened == 0).all(dim=1) & ~(flattened.abs() < threshold).all(dim=1)
    indices = mask.nonzero(as_tuple=True)[0].to(torch.int)  # int32, 4 bytes
    values = tensor[mask]  # float32, 4 bytes

    original_size = sizeof_tensor(tensor)
    sparse_size = sizeof_tensor(indices) + sizeof_tensor(values)

    print(
        f"Sparse size: {sparse_size/1024:.2f} KB, original size: {original_size/1024:.2f} KB, ratio: {sparse_size/original_size:.4f}"
    )
    

if __name__ == "__main__":

    model_dict = torch.load(
        f"./output/{SCENE_NAME}/chkpnt10000.pth", weights_only=False
    )
    for k, model in model_dict.items():
        template_gs = InstGaussianModel(sh_degree=3)
        template_gs.restore(model_args=model, training_args=None)
        break

    parser = ArgumentParser(description="Rendering script for Splat-n-Replace")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    args.source_path = f"./data/{SCENE_NAME}"
    args.model_path = f"./output/{SCENE_NAME}"

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    scene = InstScene(lp.extract(args), template_gs, shuffle=False)
    cameras = scene.getTrainCameras()

    save_path = f"./output/{SCENE_NAME}/rendered_images"
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for idx, view in enumerate(cameras):
            render_img = instanced_render(
                view, template_gs, None, pp.extract(args), background
            )["render"]
            save_image(
                render_img,
                f"{save_path}/{idx}.jpg",
            )

    # All done
    print("\nRender complete.")
