/*
 * Copyright (C) 2025, Fudan Zhang Vision Group
 * All rights reserved.
 * 
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
 */
#include <iostream>
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec4 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs, bool *clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec4 *sh = ((glm::vec4 *)shs) + idx * max_coeffs;
	glm::vec4 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
					 SH_C2[0] * xy * sh[4] +
					 SH_C2[1] * yz * sh[5] +
					 SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
					 SH_C2[3] * xz * sh[7] +
					 SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
						 SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
						 SH_C3[1] * xy * z * sh[10] +
						 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
						 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
						 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
						 SH_C3[5] * z * (xx - yy) * sh[14] +
						 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[4 * idx + 0] = (result.x < 0);
	clamped[4 * idx + 1] = (result.y < 0);
	clamped[4 * idx + 2] = (result.z < 0);
	clamped[4 * idx + 3] = (result.w < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3 &p_orig,
	const glm::vec3 scale,
	const glm::vec4 rot,
	const float *viewmatrix,
	const int W,
	const int H,
	glm::mat3 &T,
	float3 &normal)
{

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, 1.0f);
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1));

	glm::mat4 world2camera = glm::mat4(
		viewmatrix[0], viewmatrix[4], viewmatrix[8], viewmatrix[12],
		viewmatrix[1], viewmatrix[5], viewmatrix[9], viewmatrix[13],
		viewmatrix[2], viewmatrix[6], viewmatrix[10], viewmatrix[14],
		viewmatrix[3], viewmatrix[7], viewmatrix[11], viewmatrix[15]);

	glm::mat3x4 mat4x3_to_mat3 = glm::mat3x4(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);

	T = glm::transpose(splat2world) * world2camera * mat4x3_to_mat3;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float multiplier = sumf3(normal * p_view) < 0 ? 1 : -1;
	normal = multiplier * normal;
#endif
}

// 计算世界坐标到单位球极坐标
__device__ float3 computePanoramaCoordinate(const float3 &mean, const float *viewmatrix)
{
	float3 t = transformPoint4x3(mean, viewmatrix);

	float phi = atan2f(t.x, t.z);
	float theta = atan2f(sqrt(t.x * t.x + t.z * t.z), -t.y);
	float r = sqrt(t.x * t.x + t.y * t.y + t.z * t.z);

	return {theta, phi, r};
}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T,
	float cutoff,
	float2 &point_image,
	float2 &aabb_min,
	float2 &aabb_max,
	int W, int H,
	float VFOV_min,
	float VFOV_max,
	float HFOV_min,
	float HFOV_max)
{
	float3 Tu = {T[0][0], T[0][1], T[0][2]};
	float3 Tv = {T[1][0], T[1][1], T[1][2]};
	float3 Tw = {T[2][0], T[2][1], T[2][2]};

	point_image = {
		atan2f(Tu.z, Tw.z),
		atan2f(sqrtf(Tu.z * Tu.z + Tw.z * Tw.z), -Tv.z)};

	aabb_min = {INFINITY, INFINITY};
	aabb_max = {-INFINITY, -INFINITY};

	// Simulate AABB
	for (int i = 0; i < 12; i++)
	{
		float sample_theta = 2 * MY_PI * i / 12;
		glm::vec3 sample_point = glm::vec3(cutoff * sin(sample_theta), cutoff * cos(sample_theta), 1);
		glm::vec3 sample_point_image = sample_point * T;
		float phi = atan2f(sample_point_image.x, sample_point_image.z);
		float theta = atan2f(sqrt(sample_point_image.x * sample_point_image.x + sample_point_image.z * sample_point_image.z), -sample_point_image.y);

		float phi_pix = (phi - HFOV_min) * W / (HFOV_max - HFOV_min);
		float theta_pix = (theta - VFOV_min) * H / (VFOV_max - VFOV_min);

		aabb_min.x = min(aabb_min.x, phi_pix);
		aabb_max.x = max(aabb_max.x, phi_pix);
		aabb_min.y = min(aabb_min.y, theta_pix);
		aabb_max.y = max(aabb_max.y, theta_pix);
	}

	return true;
}

template <int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float *orig_points,
	const glm::vec3 *scales,
	const float scale_modifier,
	const glm::vec4 *rotations,
	const float *opacities,
	const float *shs,
	bool *clamped,
	const float *cov3D_precomp,
	const bool *mask,
	const float *colors_precomp,
	const float *viewmatrix,
	const float *projmatrix,
	const glm::vec3 *cam_pos,
	const int W, const int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int *radii_panorama,
	float2 *points_thph_image,
	float *depths_panorama,
	float *cov3Ds,
	float *rgb,
	float4 *normal_opacity_panorama,
	const dim3 grid_panorama,
	uint32_t *tiles_touched_panorama,
	bool prefiltered,
	const float vfov_min,
	const float vfov_max,
	const float hfov_min,
	const float hfov_max,
	const float scale_factor,
	float *transMats)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii_panorama[idx] = 0;
	tiles_touched_panorama[idx] = 0;

	float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
	float opacity = opacities[idx];

	// vfov 角度制转弧度制
	const float VFOV_max = MY_PI / 2 - vfov_min * MY_PI / 180;
	const float VFOV_min = MY_PI / 2 - vfov_max * MY_PI / 180;

	// hfov 角度制转弧度制
	const float HFOV_max = hfov_max * MY_PI / 180;
	const float HFOV_min = hfov_min * MY_PI / 180;

	// 计算全景图对应的极坐标&cov
	float3 mean3D_panorama = computePanoramaCoordinate(p_orig, viewmatrix); // theta phi r

	if (!(mask[idx]) || !in_frustum_panorama(mean3D_panorama, VFOV_min, VFOV_max, HFOV_min, HFOV_max, scale_factor))
		return;

	// 2d gaussian
	glm::mat3 T;
	float3 normal;
	compute_transmat(p_orig, scales[idx], rotations[idx], viewmatrix, W, H, T, normal);
	float3 *T_ptr = (float3 *)transMats;
	T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
	T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
	T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};

	float cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
	float2 point_image;
	float2 aabb_min;
	float2 aabb_max;
	bool ok = compute_aabb(T, cutoff, point_image, aabb_min, aabb_max, W, H, VFOV_min, VFOV_max, HFOV_min, HFOV_max);
	if (!ok)
		return;

	// 计算点在极坐标下的影响范围
	float2 point_image_panorama = {(mean3D_panorama.y - HFOV_min) * W / (HFOV_max - HFOV_min),
								   (mean3D_panorama.x - VFOV_min) * H / (VFOV_max - VFOV_min)};

	float radii = max(max(aabb_max.x - point_image_panorama.x, point_image_panorama.x - aabb_min.x),
					  max(aabb_max.y - point_image_panorama.y, point_image_panorama.y - aabb_min.y));
	if (radii < 0.3)
		return;
	int my_radius_panorama = ceil(radii);

	uint2 rect_min_panorama, rect_max_panorama;
	// getRect_bias(aabb_min, aabb_max, rect_min_panorama, rect_max_panorama, grid_panorama);
	getRect(point_image_panorama, my_radius_panorama, rect_min_panorama, rect_max_panorama, grid_panorama);
	if ((rect_max_panorama.x - rect_min_panorama.x) * (rect_max_panorama.y - rect_min_panorama.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec4 result;

		result = computeColorFromSH(idx, D, M, (glm::vec3 *)orig_points, *cam_pos, shs, clamped);

		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
		rgb[idx * C + 3] = result.w;
	}

	// 记录返回值
	depths_panorama[idx] = mean3D_panorama.z;
	radii_panorama[idx] = my_radius_panorama;
	points_thph_image[idx] = point_image_panorama;
	normal_opacity_panorama[idx] = {normal.x, normal.y, normal.z, opacity};
	tiles_touched_panorama[idx] = (rect_max_panorama.x - rect_min_panorama.x) * (rect_max_panorama.y - rect_min_panorama.y);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderCUDA(
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		const int S, int W, int H,
		const float2 *__restrict__ points_xy_image,
		const float *__restrict__ colors,
		const float *__restrict__ features,
		const float *__restrict__ transMats,
		const float *__restrict__ depths,
		const float4 *__restrict__ normal_opacity,
		float *__restrict__ final_T,
		int32_t *__restrict__ n_contrib,
		const float *__restrict__ bg_color,
		float *__restrict__ out_color,
		float *__restrict__ out_feature,
		float *__restrict__ out_depth,
		const float vfov_min,
		const float vfov_max,
		const float hfov_min,
		const float hfov_max,
		const float scale_factor)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	int32_t contributor = 0;
	int32_t last_contributor = 0;
	float C[CHANNELS] = {0};
	float F[13] = {0};
	float D = {0};
	float D2 = {0};

	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	int32_t median_contributor = 0;

	// vfov 角度制转弧度制
	const float VFOV_max = MY_PI / 2 - vfov_min * MY_PI / 180;
	const float VFOV_min = MY_PI / 2 - vfov_max * MY_PI / 180;

	// hfov 角度制转弧度制
	const float HFOV_max = hfov_max * MY_PI / 180;
	const float HFOV_min = hfov_min * MY_PI / 180;

	const float near = near_n * scale_factor;
	const float far = far_n * scale_factor;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8) in 2d gaussian
			const float2 xy = collected_xy[j];

			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];

			float phi = pixf.x * (HFOV_max - HFOV_min) / W + HFOV_min;
			float theta = pixf.y * (VFOV_max - VFOV_min) / H + VFOV_min;

			float3 k = cos(phi) * Tu - sin(phi) * Tw;
			float3 l = sin(phi) * cos(theta) * Tu + sin(theta) * Tv + cos(phi) * cos(theta) * Tw;
			float3 p = cross(k, l);
			if (p.z == 0.0)
				continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y);

			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

			// compute intersection and depth
			float rho = min(rho3d, rho2d);
			const float s_Tu = s.x * Tu.x + s.y * Tu.y + Tu.z;
			const float s_Tv = s.x * Tv.x + s.y * Tv.y + Tv.z;
			const float s_Tw = s.x * Tw.x + s.y * Tw.y + Tw.z;
			const float depth_3d = s_Tu * sin(theta) * sin(phi) - s_Tv * cos(theta) + s_Tw * sin(theta) * cos(phi);
			const float depth = (rho3d <= rho2d) ? depth_3d : depths[collected_id[j]];
			if (depth < near || depth > far)
				continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, opa * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1 - T;
			float m = far / (far - near) * (1 - near / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5)
			{
				median_depth = depth;
				median_contributor = contributor;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += colors[collected_id[j] * CHANNELS + ch] * alpha * T;
			for (int ch = 0; ch < S + 3; ch++)
			{
				if (ch < S)
					F[ch] += features[collected_id[j] * S + ch] * alpha * T;
				else
					F[ch] += normal[ch - S] * alpha * T; // Render normal map
			}

			// depth是射线与2d的交点
			D += depth * alpha * T;
			D2 += depth * depth * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		n_contrib[pix_id] = last_contributor;
		n_contrib[H * W + pix_id] = median_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		for (int ch = 0; ch < S + 3; ch++)
			out_feature[ch * H * W + pix_id] = F[ch];
		out_depth[pix_id] = D;
		out_depth[H * W + pix_id] = median_depth;
		out_depth[2 * H * W + pix_id] = distortion;
		out_depth[3 * H * W + pix_id] = D2;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2 *ranges,
	const uint32_t *point_list,
	const int S, int W, int H,
	const float2 *means2D,
	const float *colors,
	const float *features,
	const float *transMats,
	const float *depths,
	const float4 *normal_opacity,
	float *final_T,
	int32_t *n_contrib,
	const float *bg_color,
	float *out_color,
	float *out_feature,
	float *out_depth,
	const float vfov_min,
	const float vfov_max,
	const float hfov_min,
	const float hfov_max,
	const float scale_factor)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		S, W, H,
		means2D,
		colors,
		features,
		transMats,
		depths,
		normal_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_feature,
		out_depth,
		vfov_min,
		vfov_max,
		hfov_min,
		hfov_max,
		scale_factor);
}

void FORWARD::preprocess(
	int P, int D, int M,
	const float *means3D,
	const glm::vec3 *scales,
	const float scale_modifier,
	const glm::vec4 *rotations,
	const float *opacities,
	const float *shs,
	bool *clamped,
	const float *cov3D_precomp,
	const bool *mask,
	const float *colors_precomp,
	const float *viewmatrix,
	const float *projmatrix,
	const glm::vec3 *cam_pos,
	const int W, int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int *radii,
	float2 *means2D,
	float *depths,
	float *cov3Ds,
	float *rgb,
	float4 *normal_opacity,
	const dim3 grid,
	uint32_t *tiles_touched,
	bool prefiltered,
	const float vfov_min,
	const float vfov_max,
	const float hfov_min,
	const float hfov_max,
	const float scale_factor,
	float *transMats)
{
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		mask,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered,
		vfov_min,
		vfov_max,
		hfov_min,
		hfov_max,
		scale_factor,
		transMats);
}