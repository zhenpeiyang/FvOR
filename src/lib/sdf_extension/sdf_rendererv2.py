import torch
import numpy as np
import sdf_cuda
import open3d as o3d


def compute_sdf_regularization_loss(sdf, lengths):
    # sdf: [b, d, h, w]
    # lengths: [b, 3]
    b = len(sdf)
    voxel_size = lengths.clone()
    voxel_size[:, 0] /= float(sdf.shape[1]-1)
    voxel_size[:, 1] /= float(sdf.shape[2]-1)
    voxel_size[:, 2] /= float(sdf.shape[3]-1)
    dx = (sdf[:, 1:, :-1, :-1] - sdf[:, :-1, :-1, :-1]) / voxel_size[:, 0].reshape(b, 1, 1, 1)
    dy = (sdf[:, :-1, 1:, :-1] - sdf[:, :-1, :-1, :-1]) / voxel_size[:, 1].reshape(b, 1, 1, 1)
    dz = (sdf[:, :-1, :-1, 1:] - sdf[:, :-1, :-1, :-1]) / voxel_size[:, 2].reshape(b, 1, 1, 1)

    debug = False
    if debug:
        xs=range(sdf.shape[1])
        grid = np.stack(np.meshgrid(xs,xs,xs),-1)
        err = (torch.sqrt(dx**2 + dy**2 + dz**2) - 1).abs().data.cpu().numpy()
        write_ply('test.ply', grid[1:,1:,1:].reshape(-1,3),color=err.reshape(-1,1)*np.array([1,0,0])[None,:])

    loss1 = ((dx**2 + dy**2 + dz**2 - 1)**2).mean()

    dx2 = (dx[:, 1:, :-1, :-1] - dx[:, :-1, :-1, :-1]) / voxel_size[:, 0].reshape(b, 1, 1, 1)
    dy2 = (dy[:, :-1, 1:, :-1] - dy[:, :-1, :-1, :-1]) / voxel_size[:, 1].reshape(b, 1, 1, 1)
    dz2 = (dz[:, :-1, :-1, 1:] - dz[:, :-1, :-1, :-1]) / voxel_size[:, 2].reshape(b, 1, 1, 1)

    loss2 = (dx2**2 + dy2**2 + dz2**2).mean()

    return loss1, loss2

class SDFRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sdf, feature, voxel_origin, lengths, rays):
        sdf = sdf.contiguous()
        feature = feature.contiguous()
        voxel_origin = voxel_origin.contiguous()
        lengths = lengths.contiguous()
        rays = rays.contiguous()
        depth, feature_map, cols, coeff, ndot, ndot_f, valid = sdf_cuda.forward(
                sdf,
                feature,
                voxel_origin,
                lengths,
                rays
                )
        boundary_map = torch.zeros(10,1).float().cuda()
        variables = [sdf, feature, cols, coeff, ndot, ndot_f, valid]
        ctx.save_for_backward(*variables)
        return depth, feature_map, boundary_map

    @staticmethod
    def backward(ctx, grad_depth, grad_feature_map, grad_boundary_map):
        grad_boundary_map = grad_boundary_map.view(grad_boundary_map.shape[0], -1)
        d_sdf, d_feature= sdf_cuda.backward(
                grad_depth.contiguous(),
                grad_feature_map.contiguous(),
                grad_boundary_map.contiguous(),
                *ctx.saved_tensors
                )
        return d_sdf, d_feature, None, None, None

