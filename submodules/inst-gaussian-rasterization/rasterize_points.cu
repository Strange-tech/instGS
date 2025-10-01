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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
	const bool debug)
{
  // 参数检查
  const int num_instances = instance_transforms.size(0);
  const int num_gaussians = means3D_template.size(0);
  const int P = num_instances * num_gaussians;
  const int H = image_height;
  const int W = image_width;

  auto float_opts = means3D_template.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepthptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepthptr = out_invdepth.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D_template.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(shs_template.size(0) != 0)
	  {
		M = shs_template.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		num_gaussians,
		means3D_template.contiguous().data_ptr<float>(),
		scaling_template.contiguous().data_ptr<float>(),
		rotation_template.contiguous().data_ptr<float>(),
		shs_template.contiguous().data_ptr<float>(),
		opacity_template.contiguous().data_ptr<float>(),
		instance_transforms.contiguous().data_ptr<float>(),
		xyz_offsets.contiguous().data_ptr<float>(),
		scaling_offsets.contiguous().data_ptr<float>(),
		rotation_offsets.contiguous().data_ptr<float>(),
		shs_offsets.contiguous().data_ptr<float>(),
		opacity_offsets.contiguous().data_ptr<float>(),
		// 原有参数
		colors.contiguous().data<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_invdepthptr,
		antialiasing,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
}


torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
