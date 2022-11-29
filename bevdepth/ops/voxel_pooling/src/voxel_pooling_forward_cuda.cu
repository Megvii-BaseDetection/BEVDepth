// Copyright (c) Megvii Inc. All rights reserved.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_BLOCK_X 32
#define THREADS_BLOCK_Y 4
#define THREADS_PER_BLOCK THREADS_BLOCK_X * THREADS_BLOCK_Y
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void voxel_pooling_forward_kernel(int batch_size, int num_points, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z, const int *geom_xyz, const float *input_features,
        float *output_features, int *pos_memo) {
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int sample_dim = THREADS_PER_BLOCK;
  const int idx_in_block = tidy * THREADS_BLOCK_X + tidx;

  const int block_sample_idx = bidx * sample_dim;
  const int thread_sample_idx = block_sample_idx + idx_in_block;

  const int total_samples = batch_size * num_points;

  __shared__ int geom_xyz_shared[THREADS_PER_BLOCK * 3];

  if (thread_sample_idx < total_samples) {
    const int sample_x = geom_xyz[thread_sample_idx * 3 + 0];
    const int sample_y = geom_xyz[thread_sample_idx * 3 + 1];
    const int sample_z = geom_xyz[thread_sample_idx * 3 + 2];
    geom_xyz_shared[idx_in_block * 3 + 0] = sample_x;
    geom_xyz_shared[idx_in_block * 3 + 1] = sample_y;
    geom_xyz_shared[idx_in_block * 3 + 2] = sample_z;
    if ((sample_x >= 0 && sample_x < num_voxel_x) && (sample_y >= 0 && sample_y < num_voxel_y) && (sample_z >= 0 && sample_z < num_voxel_z)) {
      pos_memo[thread_sample_idx * 3 + 0] = thread_sample_idx / num_points;
      pos_memo[thread_sample_idx * 3 + 1] = sample_y;
      pos_memo[thread_sample_idx * 3 + 2] = sample_x;
    }
  }

  __syncthreads();

  for (int i = tidy; i < THREADS_PER_BLOCK && block_sample_idx + i < total_samples; i += THREADS_BLOCK_Y) {
    const int sample_x = geom_xyz_shared[i * 3 + 0];
    const int sample_y = geom_xyz_shared[i * 3 + 1];
    const int sample_z = geom_xyz_shared[i * 3 + 2];
    if (sample_x < 0 || sample_x >= num_voxel_x || sample_y < 0 || sample_y >= num_voxel_y || sample_z < 0 || sample_z >= num_voxel_z) {
      continue;
    }
    const int batch_idx = (block_sample_idx + i) / num_points;
    for (int j = tidx; j < num_channels; j += THREADS_BLOCK_X) {
      atomicAdd(&output_features[(batch_idx * num_voxel_y * num_voxel_x + sample_y * num_voxel_x + sample_x) * num_channels + j], input_features[(block_sample_idx + i) * num_channels + j]);
    }
  }
}

void voxel_pooling_forward_kernel_launcher(int batch_size, int num_points, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z, const int *geom_xyz, const float *input_features,
    float *output_features, int *pos_memo, cudaStream_t stream) {
  cudaError_t err;

  dim3 blocks(DIVUP(batch_size * num_points, THREADS_PER_BLOCK));
  dim3 threads(THREADS_BLOCK_X, THREADS_BLOCK_Y);

  voxel_pooling_forward_kernel<<<blocks, threads, 0, stream>>>(batch_size, num_points, num_channels, num_voxel_x, num_voxel_y, num_voxel_z, geom_xyz, input_features, output_features, pos_memo);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
