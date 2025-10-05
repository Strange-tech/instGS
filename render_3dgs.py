import torch
import numpy as np
import os
from gaussian_renderer import render, network_gui, instanced_render
import sys
from scene import Scene
from scene.inst_gaussian_model import InstGaussianModel
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
import time


SCENE_NAME = "dongsheng/15"


if __name__ == "__main__":
    vanilla_gaussians = GaussianModel(sh_degree=3)
    vanilla_gaussians.load_ply(
        f"./output/{SCENE_NAME}/inst_gs_1.ply"
    )
    # vanilla_gaussians.load_ply(
    #     f"./output/{SCENE_NAME}/point_cloud/iteration_30000/point_cloud.ply"
    # )

    parser = ArgumentParser(description="Rendering script for 3dgs")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    args.source_path = f"./data/{SCENE_NAME}"
    args.model_path = f"./output/{SCENE_NAME}"

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    scene = Scene(lp.extract(args), vanilla_gaussians, auto_load=False, shuffle=False)
    cameras = scene.getTrainCameras()

    save_path = f"./output/{SCENE_NAME}/3dgs_images"
    os.makedirs(save_path, exist_ok=True)


    with torch.no_grad():
        avg_render_time = 0.0

        for idx, view in enumerate(cameras):
            start_time = time.time()
            render_img = render(
                view, vanilla_gaussians, pp.extract(args), background
            )["render"]
            render_time = time.time() - start_time
            avg_render_time += render_time
            # save_image(
            #     render_img,
            #     f"{save_path}/{idx}.jpg",
            # )

    print(f"Total render time for {len(cameras)} views: {avg_render_time:.4f} seconds")
    avg_render_time /= len(cameras)
    # All done
    print("\nRender complete.")
    print(f"Average render time: {avg_render_time:.4f} seconds")
