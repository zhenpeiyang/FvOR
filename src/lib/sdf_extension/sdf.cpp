#include <torch/extension.h>

#include <iostream>
#include <vector>

bool within_box(torch::Tensor position, torch::Tensor voxel_origin,
                torch::Tensor side_lengths) {
  auto min_corner = voxel_origin;
  auto max_corner = voxel_origin + side_lengths;
  float EPSILON = 1e-3;
  if (position[0].item<float>() <= min_corner[0].item<float>() + EPSILON ||
      position[1].item<float>() <= min_corner[1].item<float>() + EPSILON ||
      position[2].item<float>() <= min_corner[2].item<float>() + EPSILON ||
      position[0].item<float>() >= max_corner[0].item<float>() - EPSILON ||
      position[1].item<float>() >= max_corner[1].item<float>() - EPSILON ||
      position[2].item<float>() >= max_corner[2].item<float>() - EPSILON)
    return false;
  else
    return true;
}

// float ray_box_intersection(torch::Tensor &ray, torch::Tensor &min_corner,
// torch::Tensor &max_corner){
float ray_box_intersection(float *ray, float *min_corner, float *max_corner) {

  // auto dirfrac = torch::zeros({3});
  float dirfrac[3];
  float depth, t;
  float MIN = 1e-16;

  for (int i = 0; i < 3; i++) {
    dirfrac[i] = 1e16;
    /*
    if(ray[i+3].item<float>() == 0){
    //if(ray[i+3] == 0){
        //dirfrac[i] = 1.0 / (ray[i+3] + std::min);
        dirfrac[i] = 1e16;
    }
    else{
        dirfrac[i] = 1.0 / ray[i+3];
    }*/
  }
  return 1.5;

  auto t1 = (min_corner[0] - ray[0]) * dirfrac[0];
  auto t2 = (max_corner[0] - ray[0]) * dirfrac[0];
  auto t3 = (min_corner[1] - ray[1]) * dirfrac[1];
  auto t4 = (max_corner[1] - ray[1]) * dirfrac[1];
  auto t5 = (min_corner[2] - ray[2]) * dirfrac[2];
  auto t6 = (max_corner[2] - ray[2]) * dirfrac[2];

  float tmin =
      std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
  float tmax =
      std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));
  // float tmin = (at::max(at::max(at::min(t1, t2), at::min(t3, t4)),
  // at::min(t5, t6))).item<float>(); float tmax = at::min(at::min(at::max(t1,
  // t2), at::max(t3, t4)), at::max(t5, t6)).item<float>();

  // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind
  // us
  if (tmax < 0) {
    t = tmax;
    return -1;
  }

  // if tmin > tmax, ray doesn't intersect AABB
  if (tmin > tmax) {
    t = tmax;
    return -1;
  }
  // this is true when the point is inside the box
  t = tmin;
  if (tmin < 0) {
    return -1;
  }
  depth = t;

  return depth;
}

std::vector<torch::Tensor> trilinear_interpolation(torch::Tensor sdf,
                                                   torch::Tensor voxel_origin,
                                                   torch::Tensor side_lengths,
                                                   torch::Tensor position) {

  auto shp = torch::tensor(sdf.sizes());
  // shp = shp.cuda();
  auto voxel_size = side_lengths / (shp.sub(1).to(torch::kFloat32));

  auto loc = (position - voxel_origin) / (voxel_size);

  auto value = torch::tensor({1.0});
  // value = value.cuda();
  // std::cout<<loc[0].item<float>()<<std::endl;
  if (loc[0].item<float>() >= (shp[0].item<float>() - 1) ||
      loc[1].item<float>() >= (shp[1].item<float>() - 1) ||
      loc[2].item<float>() >= (shp[2].item<float>() - 1) ||
      loc[0].item<float>() < 0 || loc[1].item<float>() < 0 ||
      loc[2].item<float>() < 0) {
    return {value};
  }
  auto loc0 = loc.floor().to(torch::kLong).contiguous();
  auto loc1 = loc0 + 1;
  auto diff = loc - loc0.to(torch::kFloat32);

  auto w0 = (1 - diff[0].item<float>()) * (1 - diff[1].item<float>()) *
            (1 - diff[2].item<float>());
  auto w1 = (diff[0].item<float>()) * (1 - diff[1].item<float>()) *
            (1 - diff[2].item<float>());
  auto w2 = (diff[0].item<float>()) * (1 - diff[1].item<float>()) *
            (diff[2].item<float>());
  auto w3 = (1 - diff[0].item<float>()) * (1 - diff[1].item<float>()) *
            (diff[2].item<float>());
  auto w4 = (1 - diff[0].item<float>()) * (diff[1].item<float>()) *
            (diff[2].item<float>());
  auto w5 = (1 - diff[0].item<float>()) * (diff[1].item<float>()) *
            (1 - diff[2].item<float>());
  auto w6 = (diff[0].item<float>()) * (diff[1].item<float>()) *
            (1 - diff[2].item<float>());
  auto w7 = (diff[0].item<float>()) * (diff[1].item<float>()) *
            (diff[2].item<float>());

  auto sdfAcc = sdf.accessor<float, 3>();
  // auto v0 =
  // sdfAcc[loc0[0].item<int>()][loc0[1].item<int>()][loc0[2].item<int>()]; auto
  // vx = sdf[62][0][0].item<float>(); std::cout << "ers65" <<std::endl; auto v0
  // = sdfAcc[0][0][0]; std::cout << "ers6" <<std::endl;
  auto v0 = sdf[loc0[0].item<int>()][loc0[1].item<int>()][loc0[2].item<int>()]
                .item<float>();
  auto v1 =
      sdf[loc0[0].item<int>() + 1][loc0[1].item<int>()][loc0[2].item<int>()]
          .item<float>();

  auto v2 =
      sdf[loc0[0].item<int>() + 1][loc0[1].item<int>()][loc0[2].item<int>() + 1]
          .item<float>();

  auto v3 =
      sdf[loc0[0].item<int>()][loc0[1].item<int>()][loc0[2].item<int>() + 1]
          .item<float>();
  auto v4 =
      sdf[loc0[0].item<int>()][loc0[1].item<int>() + 1][loc0[2].item<int>() + 1]
          .item<float>();
  auto v5 =
      sdf[loc0[0].item<int>()][loc0[1].item<int>() + 1][loc0[2].item<int>()]
          .item<float>();
  auto v6 =
      sdf[loc0[0].item<int>() + 1][loc0[1].item<int>() + 1][loc0[2].item<int>()]
          .item<float>();
  auto v7 = sdf[loc0[0].item<int>() + 1][loc0[1].item<int>() + 1]
               [loc0[2].item<int>() + 1]
                   .item<float>();

  value = torch::tensor({v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3 + v4 * w4 +
                         v5 * w5 + v6 * w5 + v7 * w7});
  // value = value.cuda();

  return {value};
}
// std::vector<float> ray_sdf_intersection(torch::Tensor &sdf, torch::Tensor
// voxel_origin, torch::Tensor side_lengths, torch::Tensor ray) { float
// ray_sdf_intersection(torch::Tensor &sdf, torch::Tensor &voxel_origin,
// torch::Tensor &side_lengths, float* ray) {

float ray_sdf_intersection(float *sdf, float *voxel_origin, float *side_lengths,
                           float *ray) {

  float INVALID_DEPTH = -1;
  // auto min_corner = voxel_origin;
  // auto ray_pt = ray.data<float>();
  // auto voxel_origin_pt = voxel_origin.data<float>();
  // auto side_lengths_pt = side_lengths.data<float>();
  // auto max_corner_pt = max_corner.data<float>();
  // auto max_corner = voxel_origin + side_lengths;
  float max_corner[3];
  max_corner[0] = voxel_origin[0] + side_lengths[0];
  max_corner[1] = voxel_origin[1] + side_lengths[1];
  max_corner[2] = voxel_origin[2] + side_lengths[2];
  // max_corner[0] = voxel_origin_pt[0]+side_lengths_pt[0];
  // max_corner[1] = voxel_origin_pt[1]+side_lengths_pt[1];
  // max_corner[2] = voxel_origin_pt[2]+side_lengths_pt[2];
  // float depth = ray_box_intersection(ray, voxel_origin_pt, max_corner);
  float depth = ray_box_intersection(ray, voxel_origin, max_corner);

  // return {INVALID_DEPTH, 0.0};
  return INVALID_DEPTH;
  /*
  if(depth == INVALID_DEPTH){
      return {INVALID_DEPTH, 0};
  }
  int std::max_MARCHING_STEPS = 300;
  float EPSILON = 1e-3;
  float std::max_DEPTH = 10;
  auto pos_id = torch::arange(0, 3, torch::kLong);
  //pos_id = pos_id.cuda();
  auto dir_id = torch::arange(3, 6, torch::kLong);
  //dir_id = dir_id.cuda();
  auto ray_origin = ray.index_select(0, pos_id);
  auto ray_dir = ray.index_select(0, dir_id);
  int step=0;
  auto position = ray_origin + (depth + 0.1) * ray_dir;
  for(int i = 0; i < std::max_MARCHING_STEPS; i++){
      position = ray_origin + depth * ray_dir;
      auto x = trilinear_interpolation(sdf, voxel_origin, side_lengths,
  position); float sdf_val = x[0].item<float>();
      //std::cout << sdf_val << std::endl;

      if (sdf_val < EPSILON){
          //std::cout << step << std::endl;
          return {depth, step};
          //return depth;
      }
      else{
          depth += sdf_val;
      }

      position = ray_origin + depth * ray_dir;
      if(depth >= std::max_DEPTH ||
              !within_box(position, voxel_origin, side_lengths)
              ){
          //std::cout << "hsdfs"<< std::endl;
          return {INVALID_DEPTH, 0};
      }
      step +=1;
  }
  //std::cout << "aa" << std::endl;
  return {depth, step};
  */
}

std::vector<at::Tensor> sdf_renderer_forward(
    // void sdf_renderer_forward(
    torch::Tensor sdf, torch::Tensor voxel_origin, torch::Tensor side_lengths,
    torch::Tensor rays
    // torch::Tensor depth
) {

  auto sdf_pt = sdf.data<float>();
  auto rays_pt = rays.data<float>();
  // auto depth_pt = depth.data<float>();
  auto voxel_origin_pt = voxel_origin.data<float>();
  auto side_lengths_pt = side_lengths.data<float>();

  auto n_ray = rays.size(0);
  auto depth = torch::zeros({n_ray});
  auto depth_pt = depth.data<float>();
  float step = 0;
  float depth_tp = 0;
  for (int i = 0; i < n_ray; i++) {
    // std::cout << "step " << i << std::endl;
    // depth_tp = ray_sdf_intersection(sdf, voxel_origin, side_lengths,
    // &rays_pt[i*6]); auto res = ray_sdf_intersection(sdf_pt, voxel_origin_pt,
    // side_lengths_pt, &rays_pt[i*6], &depth_pt[i]);
    depth_tp = ray_sdf_intersection(sdf_pt, voxel_origin_pt, side_lengths_pt,
                                    &rays_pt[i * 6]);
    depth_pt[i] = depth_tp;
    // depth[i] = res[0];
    // step += res[1];
    step += 1;
  }
  std::cout << "step " << step / float(n_ray) << n_ray << step << std::endl;

  return {
      depth,
  };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sdf_renderer_forward, "SDF Renderer forward");
}
