/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeInstGaussiansCUDA(
	const torch::Tensor& background,
	// instancing相关参数
    const torch::Tensor& means3D_template,        // (num_gaussians, 3)
    const torch::Tensor& scaling_template,        // (num_gaussians, 3)
    const torch::Tensor& rotation_template,       // (num_gaussians, 4)
    const torch::Tensor& shs_template,            // (num_gaussians, D)
    const torch::Tensor& opacity_template,        // (num_gaussians, 1)
    const torch::Tensor& instance_transforms,     // (num_instances, 4, 4)
    const torch::Tensor& xyz_offsets,             // (P, 3)
    const torch::Tensor& scaling_offsets,         // (P, 3)
    const torch::Tensor& rotation_offsets,        // (P, 4)
    const torch::Tensor& shs_offsets,             // (P, D)
    const torch::Tensor& opacity_offsets,         // (P, 1)
	// 原有参数
	const torch::Tensor& colors,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);
