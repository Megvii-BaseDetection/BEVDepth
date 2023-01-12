// Copyright (c) Megvii Inc. All rights reserved.
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_BLOCK_X 32
#define THREADS_BLOCK_Y 4
#define THREADS_PER_BLOCK THREADS_BLOCK_X *THREADS_BLOCK_Y
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void voxel_pooling_inference_forward_kernel(
    int batch_size, int num_cams, int num_depth, int num_height, int num_width,
    int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const half *depth_features,
    const half *context_features, half *output_features) {
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int sample_dim = THREADS_PER_BLOCK;
  const int idx_in_block = tidy * THREADS_BLOCK_X + tidx;
  const int batch_size_with_cams = batch_size * num_cams;
  const int block_sample_idx = bidx * sample_dim;
  const int thread_sample_idx = block_sample_idx + idx_in_block;

  const int total_samples =
      batch_size_with_cams * num_depth * num_height * num_width;
  __shared__ int geom_xyz_shared[THREADS_PER_BLOCK * 3];

  if (thread_sample_idx < total_samples) {
    const int sample_x = geom_xyz[thread_sample_idx * 3 + 0];
    const int sample_y = geom_xyz[thread_sample_idx * 3 + 1];
    const int sample_z = geom_xyz[thread_sample_idx * 3 + 2];
    geom_xyz_shared[idx_in_block * 3 + 0] = sample_x;
    geom_xyz_shared[idx_in_block * 3 + 1] = sample_y;
    geom_xyz_shared[idx_in_block * 3 + 2] = sample_z;
  }

  __syncthreads();

  for (int i = tidy;
       i < THREADS_PER_BLOCK && block_sample_idx + i < total_samples;
       i += THREADS_BLOCK_Y) {
    const int sample_x = geom_xyz_shared[i * 3 + 0];
    const int sample_y = geom_xyz_shared[i * 3 + 1];
    const int sample_z = geom_xyz_shared[i * 3 + 2];
    if (sample_x < 0 || sample_x >= num_voxel_x || sample_y < 0 ||
        sample_y >= num_voxel_y || sample_z < 0 || sample_z >= num_voxel_z) {
      continue;
    }
    const int sample_idx = block_sample_idx + i;
    const int batch_idx =
        sample_idx / (num_cams * num_depth * num_height * num_width);
    const int width_idx = sample_idx % num_width;
    const int height_idx = (sample_idx / num_width) % num_height;
    const int cam_idx =
        sample_idx / (num_depth * num_height * num_width) % num_cams;
    // if(batch_idx > 0 || cam_idx > 0)
    // printf("batch_idx: %d, width_idx: %d, height_idx: %d, cam_idx: %d \n",
    // batch_idx, width_idx, height_idx, cam_idx);
    const half depth_val = depth_features[sample_idx];
    half res;
    for (int j = tidx; j < num_channels; j += THREADS_BLOCK_X) {
      const half context_val = context_features
          [batch_idx * (num_cams * num_channels * num_height * num_width) +
           cam_idx * (num_channels * num_height * num_width) +
           j * (num_height * num_width) + height_idx * num_width + width_idx];
      res = __hmul(depth_val, context_val);
      atomicAdd(&output_features[(batch_idx * num_voxel_y * num_voxel_x +
                                  sample_y * num_voxel_x + sample_x) *
                                     num_channels +
                                 j],
                res);
    }
  }
}

__global__ void voxel_pooling_inference_forward_kernel(
    int batch_size, int num_cams, int num_depth, int num_height, int num_width,
    int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const float *depth_features,
    const float *context_features, float *output_features) {
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int sample_dim = THREADS_PER_BLOCK;
  const int idx_in_block = tidy * THREADS_BLOCK_X + tidx;
  const int batch_size_with_cams = batch_size * num_cams;
  const int block_sample_idx = bidx * sample_dim;
  const int thread_sample_idx = block_sample_idx + idx_in_block;

  const int total_samples =
      batch_size_with_cams * num_depth * num_height * num_width;
  // printf("Total sample:%d num_cams: %d, num_depth: %d num_height: %d
  // num_width: %d\n", total_samples, num_cams, num_depth, num_height,
  // num_width);
  __shared__ int geom_xyz_shared[THREADS_PER_BLOCK * 3];

  if (thread_sample_idx < total_samples) {
    const int sample_x = geom_xyz[thread_sample_idx * 3 + 0];
    const int sample_y = geom_xyz[thread_sample_idx * 3 + 1];
    const int sample_z = geom_xyz[thread_sample_idx * 3 + 2];
    geom_xyz_shared[idx_in_block * 3 + 0] = sample_x;
    geom_xyz_shared[idx_in_block * 3 + 1] = sample_y;
    geom_xyz_shared[idx_in_block * 3 + 2] = sample_z;
  }

  __syncthreads();

  for (int i = tidy;
       i < THREADS_PER_BLOCK && block_sample_idx + i < total_samples;
       i += THREADS_BLOCK_Y) {
    const int sample_x = geom_xyz_shared[i * 3 + 0];
    const int sample_y = geom_xyz_shared[i * 3 + 1];
    const int sample_z = geom_xyz_shared[i * 3 + 2];
    if (sample_x < 0 || sample_x >= num_voxel_x || sample_y < 0 ||
        sample_y >= num_voxel_y || sample_z < 0 || sample_z >= num_voxel_z) {
      continue;
    }
    const int sample_idx = block_sample_idx + i;
    const int batch_idx =
        sample_idx / (num_cams * num_depth * num_height * num_width);
    const int width_idx = sample_idx % num_width;
    const int height_idx = (sample_idx / num_width) % num_height;
    const int cam_idx =
        sample_idx / (num_depth * num_height * num_width) % num_cams;
    const float depth_val = depth_features[sample_idx];
    for (int j = tidx; j < num_channels; j += THREADS_BLOCK_X) {
      const float context_val = context_features
          [batch_idx * (num_cams * num_channels * num_height * num_width) +
           cam_idx * (num_channels * num_height * num_width) +
           j * (num_height * num_width) + height_idx * num_width + width_idx];
      atomicAdd(&output_features[(batch_idx * num_voxel_y * num_voxel_x +
                                  sample_y * num_voxel_x + sample_x) *
                                     num_channels +
                                 j],
                depth_val * context_val);
    }
  }
}

void voxel_pooling_inference_forward_kernel_launcher(
    int batch_size, int num_cams, int num_depth, int num_height, int num_width,
    int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const float *depth_features,
    const float *context_features, float *output_features,
    cudaStream_t stream) {
  cudaError_t err;
  dim3 blocks(DIVUP(batch_size * num_cams * num_depth * num_height * num_width,
                    THREADS_PER_BLOCK));
  dim3 threads(THREADS_BLOCK_X, THREADS_BLOCK_Y);

  voxel_pooling_inference_forward_kernel<<<blocks, threads, 0, stream>>>(
      batch_size, num_cams, num_depth, num_height, num_width, num_channels,
      num_voxel_x, num_voxel_y, num_voxel_z, geom_xyz, depth_features,
      context_features, output_features);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void voxel_pooling_inference_forward_kernel_launcher(
    int batch_size, int num_cams, int num_depth, int num_height, int num_width,
    int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const half *depth_features,
    const half *context_features, half *output_features, cudaStream_t stream) {
  cudaError_t err;
  dim3 blocks(DIVUP(batch_size * num_cams * num_depth * num_height * num_width,
                    THREADS_PER_BLOCK));
  dim3 threads(THREADS_BLOCK_X, THREADS_BLOCK_Y);

  voxel_pooling_inference_forward_kernel<<<blocks, threads, 0, stream>>>(
      batch_size, num_cams, num_depth, num_height, num_width, num_channels,
      num_voxel_x, num_voxel_y, num_voxel_z, geom_xyz, depth_features,
      context_features, output_features);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
