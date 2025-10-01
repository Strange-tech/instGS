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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <glm/glm.hpp>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			// --- instancing相关参数 ---
			int num_gaussians,
			const float* means3D_template,        // [num_gaussians, 3]
			const float* scaling_template,        // [num_gaussians, 3]
			const float* rotation_template,       // [num_gaussians, 4]
			const float* shs_template,            // [num_gaussians, D]
			const float* opacity_template,        // [num_gaussians, 1]
			const float* instance_transforms, // [num_instances]
			const float* xyz_offsets,             // [P, 3]
			const float* scaling_offsets,         // [P, 3]
			const float* rotation_offsets,        // [P, 4]
			const float* shs_offsets,             // [P, D]
			const float* opacity_offsets,         // [P, 1]
			// --- 原有参数 ---
			const float* colors_precomp,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* depth,
			bool antialiasing,
			int* radii,
			bool debug);

	};
};

#endif
