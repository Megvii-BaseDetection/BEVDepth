// Copyright (c) Megvii Inc. All rights reserved.
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int voxel_pooling_inference_forward_wrapper(
    int batch_size, int num_cams, int num_depth, int num_height, int num_width,
    int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    at::Tensor geom_xyz_tensor, at::Tensor depth_features_tensor,
    at::Tensor context_features_tensor, at::Tensor output_features_tensor);

void voxel_pooling_inference_forward_kernel_launcher(
    int batch_size, int num_cams, int num_depth, int num_height, int num_width,
    int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const float *depth_features,
    const float *context_features, float *output_features, cudaStream_t stream);

void voxel_pooling_inference_forward_kernel_launcher(
    int batch_size, int num_cams, int num_depth, int num_height, int num_width,
    int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const half *depth_features,
    const half *context_features, half *output_features, cudaStream_t stream);

int voxel_pooling_inference_forward_wrapper(
    int batch_size, int num_cams, int num_depth, int num_height, int num_width,
    int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    at::Tensor geom_xyz_tensor, at::Tensor depth_features_tensor,
    at::Tensor context_features_tensor, at::Tensor output_features_tensor) {
  CHECK_INPUT(geom_xyz_tensor);
  CHECK_INPUT(depth_features_tensor);
  CHECK_INPUT(context_features_tensor);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int *geom_xyz = geom_xyz_tensor.data_ptr<int>();
  if (depth_features_tensor.dtype() == at::kFloat) {
    const float *depth_features = depth_features_tensor.data_ptr<float>();
    const float *context_features = context_features_tensor.data_ptr<float>();
    float *output_features = output_features_tensor.data_ptr<float>();
    voxel_pooling_inference_forward_kernel_launcher(
        batch_size, num_cams, num_depth, num_height, num_width, num_channels,
        num_voxel_x, num_voxel_y, num_voxel_z, geom_xyz, depth_features,
        context_features, output_features, stream);
  } else if (depth_features_tensor.dtype() == at::kHalf) {
    assert(num_channels % 2 == 0);
    const half *depth_features =
        (half *)depth_features_tensor.data_ptr<at::Half>();
    const half *context_features =
        (half *)context_features_tensor.data_ptr<at::Half>();
    half *output_features = (half *)output_features_tensor.data_ptr<at::Half>();
    voxel_pooling_inference_forward_kernel_launcher(
        batch_size, num_cams, num_depth, num_height, num_width, num_channels,
        num_voxel_x, num_voxel_y, num_voxel_z, geom_xyz, depth_features,
        context_features, output_features, stream);
  }

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_pooling_inference_forward_wrapper",
        &voxel_pooling_inference_forward_wrapper,
        "voxel_pooling_inference_forward_wrapper");
}
