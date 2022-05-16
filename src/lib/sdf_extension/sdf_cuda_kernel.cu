
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <THC/THCAtomics.cuh>

#define min(a, b) (a > b) ? b : a
#define max(a, b) (a > b) ? a : b

template <typename scalar_t>
__device__ __forceinline__ float trilinear_interpolation(
    scalar_t *sdf, scalar_t *feature, scalar_t *min_corner, scalar_t *max_corner, float *position,
    size_t d, size_t h, size_t w, bool *valid, int *col_idx, float *coeff,
    float *normal, float *normal_f, float *feature_cur, size_t FDIM) {
  valid[0] = true;
  auto voxel_size_x = (max_corner[0] - min_corner[0]) / float(d - 1);
  auto voxel_size_y = (max_corner[1] - min_corner[1]) / float(h - 1);
  auto voxel_size_z = (max_corner[2] - min_corner[2]) / float(w - 1);

  auto x = (position[0] - min_corner[0]) / voxel_size_x;
  auto y = (position[1] - min_corner[1]) / voxel_size_y;
  auto z = (position[2] - min_corner[2]) / voxel_size_z;

  if ((x >= d - 1) || (y >= h - 1) || (z >= w - 1) || (x < 0) || (y < 0) ||
      (z < 0)

  ) {
    valid[0] = false;
    return 0.0;
  }

  int x0 = floor(x);
  int y0 = floor(y);
  int z0 = floor(z);
  auto dx = x - x0;
  auto dy = y - y0;
  auto dz = z - z0;

  coeff[0] = (1 - dx) * (1 - dy) * (1 - dz);
  coeff[1] = (dx) * (1 - dy) * (1 - dz);
  coeff[2] = (dx) * (1 - dy) * (dz);
  coeff[3] = (1 - dx) * (1 - dy) * (dz);
  coeff[4] = (1 - dx) * (dy) * (dz);
  coeff[5] = (1 - dx) * (dy) * (1 - dz);
  coeff[6] = (dx) * (dy) * (1 - dz);
  coeff[7] = (dx) * (dy) * (dz);

  col_idx[0] = (x0)*h * w + (y0)*w + (z0);
  col_idx[1] = (x0 + 1) * h * w + (y0)*w + (z0);
  col_idx[2] = (x0 + 1) * h * w + (y0)*w + (z0 + 1);
  col_idx[3] = (x0)*h * w + (y0)*w + (z0 + 1);
  col_idx[4] = (x0)*h * w + (y0 + 1) * w + (z0 + 1);
  col_idx[5] = (x0)*h * w + (y0 + 1) * w + (z0);
  col_idx[6] = (x0 + 1) * h * w + (y0 + 1) * w + (z0);
  col_idx[7] = (x0 + 1) * h * w + (y0 + 1) * w + (z0 + 1);

  auto v0 = sdf[col_idx[0]];
  auto v1 = sdf[col_idx[1]];
  auto v2 = sdf[col_idx[2]];
  auto v3 = sdf[col_idx[3]];
  auto v4 = sdf[col_idx[4]];
  auto v5 = sdf[col_idx[5]];
  auto v6 = sdf[col_idx[6]];
  auto v7 = sdf[col_idx[7]];

  auto value = v0 * coeff[0] + v1 * coeff[1] + v2 * coeff[2] + v3 * coeff[3] +
               v4 * coeff[4] + v5 * coeff[5] + v6 * coeff[6] + v7 * coeff[7];

  // precompute df/dx, need for backward pass
  normal[0] = ((v1 - v0) * (1 - dy) * (1 - dz) + (v2 - v3) * (1 - dy) * dz +
               (v6 - v5) * dy * (1 - dz) + (v7 - v4) * dz * dy) /
              voxel_size_x;
  normal[1] = ((v5 - v0) * (1 - dx) * (1 - dz) + (v6 - v1) * dx * (1 - dz) +
               (v4 - v3) * (1 - dx) * dz + (v7 - v2) * dx * dz) /
              voxel_size_y;
  normal[2] = ((v3 - v0) * (1 - dx) * (1 - dy) + (v2 - v1) * dx * (1 - dy) +
               (v7 - v6) * dx * dy + (v4 - v5) * (1 - dx) * dy) /
              voxel_size_z;

  for(int i=0;i<FDIM;i++){
      v0 = feature[col_idx[0]*FDIM+i];
      v1 = feature[col_idx[1]*FDIM+i];
      v2 = feature[col_idx[2]*FDIM+i];
      v3 = feature[col_idx[3]*FDIM+i];
      v4 = feature[col_idx[4]*FDIM+i];
      v5 = feature[col_idx[5]*FDIM+i];
      v6 = feature[col_idx[6]*FDIM+i];
      v7 = feature[col_idx[7]*FDIM+i];
    feature_cur[i] = v0 * coeff[0] + v1 * coeff[1] + v2 * coeff[2] + v3 * coeff[3] +
               v4 * coeff[4] + v5 * coeff[5] + v6 * coeff[6] + v7 * coeff[7];

      normal_f[i*FDIM] = ((v1 - v0) * (1 - dy) * (1 - dz) + (v2 - v3) * (1 - dy) * dz +
                   (v6 - v5) * dy * (1 - dz) + (v7 - v4) * dz * dy) /
                  voxel_size_x;
      normal_f[i*FDIM+1] = ((v5 - v0) * (1 - dx) * (1 - dz) + (v6 - v1) * dx * (1 - dz) +
                   (v4 - v3) * (1 - dx) * dz + (v7 - v2) * dx * dz) /
                  voxel_size_y;
      normal_f[i*FDIM+2] = ((v3 - v0) * (1 - dx) * (1 - dy) + (v2 - v1) * dx * (1 - dy) +
                   (v7 - v6) * dx * dy + (v4 - v5) * (1 - dx) * dy) /
                  voxel_size_z;

  }

  return value;
}

template <typename scalar_t>
__device__ __forceinline__ float ray_box_intersect(scalar_t *ray,
                                                   scalar_t *min_corner,
                                                   scalar_t *max_corner) {
  float dirfrac[3];

  const scalar_t INVALID_DEPTH = -1;
  float depth, t;

  for (int i = 0; i < 3; i++) {
    if (ray[i + 3] == 0) {
      dirfrac[i] = 1e16;
    } else {
      dirfrac[i] = 1.0 / ray[i + 3];
    }
  }

  auto t1 = (min_corner[0] - ray[0]) * dirfrac[0];
  auto t2 = (max_corner[0] - ray[0]) * dirfrac[0];
  auto t3 = (min_corner[1] - ray[1]) * dirfrac[1];
  auto t4 = (max_corner[1] - ray[1]) * dirfrac[1];
  auto t5 = (min_corner[2] - ray[2]) * dirfrac[2];
  auto t6 = (max_corner[2] - ray[2]) * dirfrac[2];

  //auto tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
  auto a0=min(t1,t2);
  auto a1=min(t3,t4);
  auto a2=min(t5,t6);
  auto tmin=max(a0,a1);
  tmin=max(tmin,a2);
  //auto tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
  auto a3=max(t1,t2);
  auto a4=max(t3,t4);
  auto a5=max(t5,t6);
  auto tmax=min(a3,a4);
  tmax=min(tmax,a5);

  if (tmax < 0) {
    t = tmax;
    return INVALID_DEPTH;
  }

  // if tmin > tmax, ray doesn't intersect AABB
  if (tmin > tmax) {
    t = tmax;
    return INVALID_DEPTH;
  }
  // this is true when the point is inside the box
  t = tmin;
  if (tmin < 0) {
    return INVALID_DEPTH;
  }
  depth = t;

  return depth;
}

template <typename scalar_t>
__global__ void sdf_cuda_forward_kernel(
    const scalar_t *__restrict__ sdf, 
    const scalar_t *__restrict__ feature, 
    const scalar_t *__restrict__ min_corner,
    const scalar_t *__restrict__ max_corner, 
    const scalar_t *__restrict__ rays,
    scalar_t *__restrict__ depth, 
    scalar_t *__restrict__ feature_map, 
    int *__restrict__ col_idx,
    scalar_t *__restrict__ coeff, 
    scalar_t *__restrict__ ndot,
    scalar_t *__restrict__ ndot_f,
    int *__restrict__ valid, 
    size_t b, 
    size_t n_ray, 
    size_t d, 
    size_t h,
    size_t w,
    size_t FDIM
    ) {

  const int batch_idx = blockIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;

  if (column >= n_ray || batch_idx >= b) {
    return;
  }

  const scalar_t INVALID_DEPTH = -1;

  auto sdf_idx = (batch_idx * d * h * w);
  auto feature_idx = (batch_idx * d * h * w * FDIM);
  auto ray_idx = (batch_idx * n_ray + column) * 6;
  auto corner_idx = (batch_idx)*3;
  auto depth_idx = (batch_idx * n_ray + column);
  auto feature_map_idx = (batch_idx * n_ray+ column)*FDIM;
  auto d_cur = ray_box_intersect(&rays[ray_idx], &min_corner[corner_idx],
                                 &max_corner[corner_idx]);
  d_cur += 0.01;

  if (d_cur == float(INVALID_DEPTH)) {
    depth[depth_idx] = INVALID_DEPTH;
    return;
  } else {
    int MAX_MARCHING_STEPS = 300;
    float EPSILON = 0;
    float MINIMUM_STEP_SIZE = 1*1e-3;
    float MAXIMUM_STEP_SIZE = 1*1e-2;
    float position[3];

    //MUST MATCH FDIM
    float feature_cur[3];
    float normal_cur_f[3*3];

    bool valid_cur;
    int col_idx_cur[8];
    float coeff_cur[8];
    float normal_cur[3];
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
      position[0] = rays[ray_idx] + d_cur * rays[ray_idx + 3];
      position[1] = rays[ray_idx + 1] + d_cur * rays[ray_idx + 4];
      position[2] = rays[ray_idx + 2] + d_cur * rays[ray_idx + 5];

      auto sdf_val = trilinear_interpolation(
          &sdf[sdf_idx], &feature[feature_idx], &min_corner[corner_idx], &max_corner[corner_idx],
          position, d, h, w, &valid_cur, col_idx_cur, coeff_cur, normal_cur, normal_cur_f, feature_cur, FDIM);

      if (!valid_cur) {
        depth[depth_idx] = INVALID_DEPTH;

        return;
      } else {
        if (sdf_val < EPSILON) {
          memcpy(&col_idx[depth_idx * 8], col_idx_cur, 8*sizeof(int));
          //memcpy(&col_idx[depth_idx * 8], col_idx_cur, 8);
          memcpy(&coeff[depth_idx * 8], coeff_cur, 8*sizeof(float));
          //memcpy(&coeff[depth_idx * 8], coeff_cur, 8);
          auto norm = sqrt(normal_cur[0] * normal_cur[0] +
                      normal_cur[1] * normal_cur[1] +
                      normal_cur[2] * normal_cur[2]);
          if (norm != 0) {
            ndot[depth_idx] = (normal_cur[0] * rays[ray_idx + 3] +
                               normal_cur[1] * rays[ray_idx + 4] +
                               normal_cur[2] * rays[ray_idx + 5]) /
                              norm;
          }
          else{
              ndot[depth_idx] = 0;
          }

          for(int j=0;j<FDIM;j++){
                norm = sqrt(normal_cur_f[j*3+0] * normal_cur_f[j*3+0] +
                          normal_cur_f[j*3+1] * normal_cur_f[j*3+1] +
                          normal_cur_f[j*3+2] * normal_cur_f[j*3+2]);
              if (norm != 0) {
                ndot_f[depth_idx*FDIM+j] = (normal_cur_f[j*3+0] * rays[ray_idx + 3] +
                                   normal_cur_f[j*3+1] * rays[ray_idx + 4] +
                                   normal_cur_f[j*3+2] * rays[ray_idx + 5]) /
                                  norm;
              }
              else{
                  ndot_f[depth_idx*FDIM+j] = 0;
              }

          }

          valid[depth_idx] = 1;
          depth[depth_idx] = d_cur;
          memcpy(&feature_map[feature_map_idx], feature_cur, sizeof(float)*FDIM);
          return;
        }
          auto step_size = max(sdf_val, MINIMUM_STEP_SIZE);
          step_size = min(step_size, MAXIMUM_STEP_SIZE);
        //d_cur += max(sdf_val, MINIMUM_STEP_SIZE);
          d_cur += step_size;
      }
    }

    depth[depth_idx] = INVALID_DEPTH;
  }
}

std::vector<torch::Tensor> sdf_renderer_cuda_forward(torch::Tensor sdf,torch::Tensor feature,
                                                     torch::Tensor voxel_origin,
                                                     torch::Tensor side_lengths,
                                                     torch::Tensor rays) {
  const auto b = sdf.size(0);
  const auto d = sdf.size(1);
  const auto h = sdf.size(2);
  const auto w = sdf.size(3);
  const auto c = feature.size(4);
  const auto n_ray = rays.size(1);
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

  auto depth = torch::zeros({b, n_ray}, options);
  auto feature_map = torch::zeros({b, n_ray, c}, options);
  auto col_idx = torch::zeros({b, n_ray, 8}, options_int);
  auto coeff = torch::zeros({b, n_ray, 8}, options);
  auto ndot = torch::zeros({b, n_ray}, options);
  auto ndot_f = torch::zeros({b, n_ray, c}, options);
  auto normal = torch::zeros({b, n_ray, 3}, options);
  auto valid = torch::zeros({b, n_ray}, options_int);
  auto min_corner = voxel_origin;
  auto max_corner = voxel_origin + side_lengths;

  const int threads = 128;
  const dim3 blocks((n_ray + threads - 1) / threads, b);

  AT_DISPATCH_FLOATING_TYPES(
      depth.type(), "sdf_forward_cuda", ([&] {
        sdf_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            sdf.data<scalar_t>(), 
            feature.data<scalar_t>(), 
            min_corner.data<scalar_t>(),
            max_corner.data<scalar_t>(), 
            rays.data<scalar_t>(),
            depth.data<scalar_t>(), 
            feature_map.data<scalar_t>(), 
            col_idx.data<int>(), 
            coeff.data<scalar_t>(),
            ndot.data<scalar_t>(), 
            ndot_f.data<scalar_t>(), 
            valid.data<int>(), 
            b, 
            n_ray, 
            d, 
            h, 
            w,
            c
            );
      }));
  return {depth, feature_map, col_idx, coeff, ndot, ndot_f, valid};
}

template <typename scalar_t>
__global__ void sdf_cuda_backward_kernel(
    const scalar_t *__restrict__ grad_depth, 
    const scalar_t *__restrict__ grad_feature, 
    const scalar_t *__restrict__ grad_boundary, 
    const scalar_t *__restrict__ sdf,
    const scalar_t *__restrict__ feature,
    const int *__restrict__ col_idx, const scalar_t *__restrict__ coeff,
    const int *__restrict__ valid, 
    const scalar_t *__restrict__ ndot,
    const scalar_t *__restrict__ ndot_f,
    scalar_t *__restrict__ d_sdf, 
    scalar_t *__restrict__ d_feature, 
    size_t b, size_t sdf_t, size_t feature_t, size_t n_ray, size_t FDIM) {
  const int batch_idx = blockIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  auto depth_idx = (batch_idx * n_ray + column);

  if (column >= n_ray || batch_idx >= b) {
    return;
  }

  if (valid[depth_idx] == 1) {
    auto col_start = (depth_idx)*8;
    auto sdf_start = batch_idx * sdf_t;
    auto feature_start = batch_idx * feature_t;
    scalar_t scale=0;
    scalar_t d_sdf_tp=0;
    scalar_t ndot_tp = abs(ndot[depth_idx]);
    ndot_tp = max(ndot_tp, 1e-1);
    ndot_tp = 1.0/ndot_tp;
    for(int i=0;i<FDIM;i++){
        scale += grad_feature[depth_idx*FDIM+i]*
            ndot_f[depth_idx*FDIM+i];
    }

    for (int i = 0; i < 8; i++) {
        for(int j=0;j<FDIM;j++){
        //gradient that comes from feature propagate to both feature and xyz
        atomicAdd(
              &d_feature[feature_start + col_idx[col_start + i]*FDIM+j],
             grad_feature[depth_idx*FDIM+j] * coeff[col_start + i]);
        }

        if(1){
            d_sdf_tp=ndot_tp * scale * coeff[col_start + i];

            // gradient that comes from xyz backpropagate only to xyz
            d_sdf_tp += 
              ndot_tp * grad_depth[depth_idx] * coeff[col_start + i];

          atomicAdd(
              &d_sdf[sdf_start + col_idx[col_start + i]], d_sdf_tp);
        }
        else{

          atomicAdd(
              &d_sdf[sdf_start + col_idx[col_start + i]], 
              grad_boundary[depth_idx] * coeff[col_start + i]
);


    }
  }
}
}

std::vector<torch::Tensor> sdf_renderer_cuda_backward(
    torch::Tensor grad_depth, 
    torch::Tensor grad_feature, 
    torch::Tensor grad_boundary, 
    torch::Tensor sdf, torch::Tensor feature, torch::Tensor col_idx,
    torch::Tensor coeff, torch::Tensor ndot, torch::Tensor ndot_f, torch::Tensor valid
    // torch::Tensor depth
) {
  const auto b = sdf.size(0);
  const auto d = sdf.size(1);
  const auto h = sdf.size(2);
  const auto w = sdf.size(3);
  const auto FDIM = feature.size(4);
  const auto n_ray = col_idx.size(1);
  const auto sdf_t = d * h * w;
  const auto feature_t = d * h * w * FDIM;

  auto d_sdf = torch::zeros_like(sdf);
  auto d_feature = torch::zeros_like(feature);

  const int threads = 128;
  const dim3 blocks((n_ray + threads - 1) / threads, b);

  AT_DISPATCH_FLOATING_TYPES(
      sdf.type(), "sdf_backward_cuda", ([&] {
        sdf_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_depth.data<scalar_t>(), 
            grad_feature.data<scalar_t>(), 
            grad_boundary.data<scalar_t>(), 
            sdf.data<scalar_t>(),
            feature.data<scalar_t>(),

            col_idx.data<int>(), coeff.data<scalar_t>(), valid.data<int>(),
            ndot.data<scalar_t>(), 
            ndot_f.data<scalar_t>(), 
            d_sdf.data<scalar_t>(), 
            d_feature.data<scalar_t>(), 
            b, sdf_t, feature_t, n_ray, FDIM);
      }));
  return {d_sdf, d_feature};
}
