
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> sdf_renderer_cuda_forward(torch::Tensor sdf,
        torch::Tensor feature,
                                                     torch::Tensor voxel_origin,
                                                     torch::Tensor side_lengths,
                                                     torch::Tensor rays
                                                     // torch::Tensor depth
);

std::vector<torch::Tensor> sdf_renderer_cuda_backward(torch::Tensor grad_depth,
        torch::Tensor grad_feature,
        torch::Tensor grad_boundary,
                                                     torch::Tensor sdf,
                                                     torch::Tensor feature,
                                                     torch::Tensor col_idx,
                                                     torch::Tensor coeff,
                                                     torch::Tensor ndot,
                                                     torch::Tensor ndot_f,
                                                     torch::Tensor valid
                                                     // torch::Tensor depth
);

// C++ interface

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> sdf_renderer_forward(torch::Tensor sdf,
torch::Tensor feature,
                                                torch::Tensor voxel_origin,
                                                torch::Tensor side_lengths,
                                                torch::Tensor rays
                                                // torch::Tensor depth
) {

  CHECK_INPUT(sdf);
  CHECK_INPUT(feature);
  CHECK_INPUT(voxel_origin);
  CHECK_INPUT(side_lengths);
  CHECK_INPUT(rays);
  // CHECK_INPUT(depth);

  return sdf_renderer_cuda_forward(sdf, feature, voxel_origin, side_lengths, rays);
}

std::vector<torch::Tensor> sdf_renderer_backward(
    torch::Tensor grad_depth,
    torch::Tensor grad_feature,
    torch::Tensor grad_boundary,
    torch::Tensor sdf,
    torch::Tensor feature,
    torch::Tensor col_idx,
    torch::Tensor coeff,
    torch::Tensor ndot,
    torch::Tensor ndot_f,
    torch::Tensor valid
    ) {
  CHECK_INPUT(grad_depth);
  CHECK_INPUT(grad_feature);
  CHECK_INPUT(grad_boundary);
  CHECK_INPUT(sdf);
  CHECK_INPUT(feature);
  CHECK_INPUT(col_idx);
  CHECK_INPUT(coeff);
  CHECK_INPUT(ndot);
  CHECK_INPUT(ndot_f);
  CHECK_INPUT(valid);

  return sdf_renderer_cuda_backward(
          grad_depth,
          grad_feature,
          grad_boundary,
          sdf,
          feature,
          col_idx, 
          coeff,
          ndot,
          ndot_f,
          valid
      );
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sdf_renderer_forward, "SDF Renderer forward (CUDA)");
  m.def("backward", &sdf_renderer_backward, "SDF Renderer forward (CUDA)");
}
