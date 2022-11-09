// Copyright (c) Megvii Inc. All rights reserved.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void voxel_pooling_forward_kernel(int batch_size, int num_points, int num_channels, int num_voxel_x,
                                             int num_voxel_y, int num_voxel_z, const int *geom_xyz,
                                             const float *input_features, float *output_features, int *pos_memo) {
  // Each thread process only one channel of one voxel.
  int blk_idx = blockIdx.x;
  int thd_idx = threadIdx.x;
  int pt_idx = blk_idx * blockDim.x + thd_idx;
  if (pt_idx >= batch_size * num_points) {
    return;
  } else {
    int batch_idx = pt_idx / num_points;
    int x = geom_xyz[pt_idx * 3];
    int y = geom_xyz[pt_idx * 3 + 1];
    int z = geom_xyz[pt_idx * 3 + 2];
    // if coord of current voxel is out of boundary, return.
    if (x < 0 || x >= num_voxel_x || y < 0 || y >= num_voxel_y || z < 0 || z >= num_voxel_z) {
      return;
    }
    pos_memo[pt_idx * 3] = batch_idx;
    pos_memo[pt_idx * 3 + 1] = y;
    pos_memo[pt_idx * 3 + 2] = x;
    for (int channel_idx = 0; channel_idx < num_channels; channel_idx++) {
      atomicAdd(
          &output_features[(batch_idx * num_voxel_y * num_voxel_x + y * num_voxel_x + x) * num_channels + channel_idx],
          input_features[pt_idx * num_channels + channel_idx]);
    }
  }
}

void voxel_pooling_forward_kernel_launcher(int batch_size, int num_points, int num_channels, int num_voxel_x,
                                           int num_voxel_y, int num_voxel_z, const int *geom_xyz,
                                           const float *input_features, float *output_features, int *pos_memo,
                                           cudaStream_t stream) {
  cudaError_t err;

  dim3 blocks(DIVUP(batch_size * num_points, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  voxel_pooling_forward_kernel<<<blocks, threads, 0, stream>>>(batch_size, num_points, num_channels, num_voxel_x,
                                                               num_voxel_y, num_voxel_z, geom_xyz, input_features,
                                                               output_features, pos_memo);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
