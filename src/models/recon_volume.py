import torch
import torch.nn as nn
from .modules import *
from .diff_pnp import *
from .backbone import FeatureNet
import numpy as np
from utils import get_unit_grid, create_gif, freeze_param, cuda_time
from torch import distributed as dist
import importlib
from functools import partial
import open3d as o3d
import cv2
import torchvision
from pykdtree.kdtree import KDTree
from .embedder import *
from .rend_util import rot_to_quat, quat_to_rot, get_camera_params
from camera_utils import get_cameras_accuracy
from collections import defaultdict
import os
import sys
from utils import *
from lib.sdf_extension.sdf_rendererv2 import *
import glob


class CostRegNet(nn.Module):
    def __init__(self,
                 channel_in=32,
                 channel_out=1,
                 down_sample_times=1,
                 kernel_size=3,
                 nhidden=64,
                 norm='bn',
                 position_encoding=True,
                 grid_dim=64,
                 track_running_stats=True):
        super(CostRegNet, self).__init__()
        self.track_running_stats = track_running_stats
        self.position_encoding = position_encoding
        self.down_sample_times = down_sample_times
        self.conv0 = self._make_block(channel_in,
                                      nhidden // 2,
                                      stride=1,
                                      kernel_size=1)
        channels = [
            nhidden // 2, nhidden, nhidden, nhidden * 2, nhidden * 2,
            nhidden * 4, nhidden * 8
        ]
        kernels = [3, 3, 3, 3, 3, 1]
        norms = ['bn', 'bn', 'bn', 'bn', 'bn', None]
        for i in range(down_sample_times):
            setattr(
                self, f'down_block_{i}',
                nn.Sequential(
                    self._make_block(
                        channels[i],
                        channels[i + 1],
                        stride=2,
                        kernel_size=kernels[i],
                        norm=norms[i],
                    ),
                    self._make_block(
                        channels[i + 1],
                        channels[i + 1],
                        stride=1,
                        kernel_size=kernels[i],
                        norm=norms[i],
                    )))

        for i in range(down_sample_times - 1, -1, -1):
            setattr(
                self, f'up_block_{i}',
                self._make_block(
                    channels[i + 1],
                    channels[i],
                    stride=2,
                    kernel_size=kernels[i],
                    downsample=False,
                    norm=norms[i],
                ))

        self.last_conv = nn.Conv3d(channels[0],
                                   channel_out,
                                   3,
                                   stride=1,
                                   padding=1)
        if self.position_encoding:
            embed_fn, channel_in = get_embedder(2)
            self.register_buffer(
                'grid',
                embed_fn(
                    get_unit_grid(1, grid_dim,
                                  half_size=0.5).view(1, -1, 3)).view(
                                      1, grid_dim, grid_dim, grid_dim,
                                      -1).permute(0, 4, 1, 2, 3))

    def _make_block(self,
                    channel_in,
                    channel_out,
                    kernel_size=3,
                    stride=2,
                    norm='bn',
                    downsample=True):
        pad = {3: 1, 1: 0}
        return ConvBlock(channel_in,
                         channel_out,
                         stride=stride,
                         kernel_size=kernel_size,
                         pad=pad[kernel_size],
                         norm=norm,
                         downsample=downsample,
                         track_running_stats=self.track_running_stats)

    def _position_encoding(self, volume):
        if len(volume) != len(self.grid):
            self.grid = self.grid[:1].repeat(len(volume), 1, 1, 1, 1)
        volume = torch.cat((volume, self.grid), 1)
        return volume

    def forward(self, x):
        """Run 3D Convolutions on volumetric feature map.
        Args:
            x (Tensor [b, c, d, d, d]): Input volumetric feature.
        Returns:
            (Tensor [b, c, d, d, d]): Output volumetric feature.
        """
        if self.position_encoding:
            x = self._position_encoding(x)
        features = dict()
        x = self.conv0(x)
        features[0] = x
        for i in range(self.down_sample_times):
            x = self.__getattr__(f'down_block_{i}')(x)
            features[i + 1] = x
        for i in range(self.down_sample_times - 1, -1, -1):
            x = features[i] + self.__getattr__(f'up_block_{i}')(x)
        return self.last_conv(x)


class SDFDecoder(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_out=[1],
    ):
        super().__init__()
        embed_fn, pos_embed_ch = get_embedder(2)
        self.embed_fn = embed_fn
        tmp = channel_in * 2 + pos_embed_ch
        self.fc_code = nn.Sequential(nn.Linear(tmp,
                                               512), nn.Softplus(beta=100),
                                     nn.Linear(512,
                                               512), nn.Softplus(beta=100),
                                     nn.Linear(512, 512),
                                     nn.Softplus(beta=100), nn.Linear(512, 64),
                                     nn.Softplus(beta=100), nn.Linear(64, 1))

    def forward(self, pts, data):
        """Decoder SDF value from volumetric and image features..
        Args:
            pts (Tensor [b, n, 3]): Query points.
            data['volume'] (Tensor [b, c, d, d, d]): Volumetric feature grid.
            data['feature_map'] (Tensor [b, m, c, h, w]): Feature map for each view.
            data['corners'] (Tensor [b, c, d, d, d]): The min_corner and max_corner for which data['volume'] was built.
            data['proj'] (Tensor [b, c, 4, 4]): The projection matrix of each view..
        Returns:
            ret['sdf_pred'] (Tensor [b, n]): Predicted SDF value for each query point.
        """
        volume, corners, proj, feature_map = data['volume'], data[
            'corners'], data['proj'], data['feature_map']
        ret = dict()
        b, num_point = pts.shape[:2]
        xyz = pts[:, :, :3]

        pos_embed = self.embed_fn(xyz).transpose(-2, -1)

        ### 3D Volumetric Feature
        dim = volume.shape[2]
        xyz_norm = (xyz - corners[:, 0].reshape(-1, 1, 3)) / (
            corners[:, 1] - corners[:, 0]).view(-1, 1, 3) * 2 - 1
        conv_f = grid_sample3d(volume,
                               xyz_norm[:, :, None,
                                        None, :]).squeeze(-1).squeeze(-1)

        ### 2D Feature
        uvd = xyz.unsqueeze(1) @ proj[:, :, :3, :3].transpose(
            2, 3) + proj[:, :, :3, 3].unsqueeze(2)
        Z = uvd[:, :, :, 2]
        b, n, c, h, w = feature_map.shape
        uv = uvd[:, :, :, :2] / uvd[:, :, :, 2:3].clamp(1e-16, None)
        uv[:, :, :, 0] = (uv[:, :, :, 0] / (w - 1) * 2 - 1)
        uv[:, :, :, 1] = (uv[:, :, :, 1] / (h - 1) * 2 - 1)
        valid_uv = (uv[:,:,:,0].abs() <= 1) & \
                (uv[:,:,:,1].abs() <= 1) & \
                            (Z > 0)

        X_mask = ((uv[:, :, :, 0] > 1) + (uv[:, :, :, 0] < -1)).detach()
        uv[:, :, :, 0][X_mask] = 2
        Y_mask = ((uv[:, :, :, 1] > 1) + (uv[:, :, :, 1] < -1)).detach()
        uv[:, :, :, 1][Y_mask] = 2
        pix_f = grid_sample(feature_map.view(b * n, -1, h, w),
                            uv.view(b * n, -1, 1, 2)).view(b, n, c, -1)
        pix_f = pix_f.mean(dim=1)

        code = torch.zeros(pix_f.shape[0], 0, pix_f.shape[2]).to(pix_f)
        code = torch.cat((code, conv_f), dim=1)
        code = torch.cat((code, pix_f), dim=1)
        code = torch.cat((code, pos_embed), dim=1)
        code = code.permute(0, 2, 1)
        ret['sdf_pred'] = self.fc_code(code)[:, :, 0]
        ret['sdf_pred_mask'] = torch.ones_like(
            ret['sdf_pred']).bool().squeeze(-1)
        return ret


class model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pixel_ch = config.BACKBONE.PIXEL_FEATURE_DIM
        self.grid_dim = config.COST_REG_NET.GRID_DIM
        channel_in = 5
        self.feature = FeatureNet(
            'resnet18',
            channel_in=channel_in,
            channel_out=self.pixel_ch,
        )
        if self.config.TRAINER.TRAIN_SCALE:
            self.feature2 = FeatureNet(
                'resnet18',
                channel_in=channel_in,
                channel_out=self.pixel_ch,
            )
            self.conv_scale = Encoder3D(
                channel_in=96,
                channel_out=self.config.TRAINER.BBOX_SCALE_NUM,
                down_sample_times=6,
                kernel_size=3,
                nhidden=64)

        self.embed_fn, channel_in = get_embedder(2)
        if self.config.COST_REG_NET.VOLUME_REDUCTION == 'mean+variance':
            cost_reg_ch = 2 * self.pixel_ch
        else:
            cost_reg_ch = self.pixel_ch

        self.cost_regularization = CostRegNet(
            channel_in=cost_reg_ch + channel_in,
            channel_out=self.pixel_ch,
            down_sample_times=config.COST_REG_NET.NUM_COST_REG_LAYER,
            nhidden=64,
            norm='bn',
            track_running_stats=config.TRAINER.TRACK_RUNNING_STATS,
            position_encoding=config.COST_REG_NET.POSITION_ENCODING,
            grid_dim=self.grid_dim,
        )
        self.sdf_decoder = SDFDecoder(channel_in=self.pixel_ch, )
        idv, idu = torch.meshgrid(
            torch.arange(self.config.DATASET.IMAGE_HEIGHT),
            torch.arange(self.config.DATASET.IMAGE_WIDTH))
        self.image_grid = torch.stack((idu, idv), -1).reshape(-1, 2)

        if self.config.TRAINER.TRAIN_MASK:
            self.mask_head = build_head(self.pixel_ch, 1)

        if not self.config.TRAINER.TRAIN_SHAPE:
            freeze_param(self.feature)
            freeze_param(self.sdf_decoder)
            freeze_param(self.cost_regularization)

        if self.config.TRAINER.TRAIN_POSE:
            self.flownet = FeatureNet(
                'resnet18',
                channel_in=1,
                channel_out=64,
                no_bn_act_last_layer=True,
                normalize_feature_map=self.config.POSE_REFINE.
                FLOWNET_NORMALIZE_FEATURE_MAP)
            self.pnp = PnP()

    def dump_vis(self,
                 data,
                 dump_dir,
                 prefix,
                 metrics=None,
                 mesh_name='mesh_final'):
        tbd_figures = {}
        for i in range(len(data[mesh_name])):
            identifier = data['bag_names'][i]
            if metrics is not None:
                identifier = identifier + f"_{metrics[i]:.3f}"
            mesh = data[mesh_name][i]
            grid_img = torchvision.utils.make_grid(
                data['images'][i, :, :3, :, :])
            cv2.imwrite(f"{dump_dir}/{prefix}_{identifier}.png",
                        grid_img.data.cpu().numpy().transpose(1, 2, 0) * 255)
            if mesh is not None:
                write_ply(f"{dump_dir}/{prefix}_{identifier}_gt.ply",
                          data['surface_point'][i].data.cpu().numpy())
                o3d.io.write_triangle_mesh(
                    f"{dump_dir}/{prefix}_{identifier}.ply", mesh)
            if self.config.TRAINER.TRAIN_MASK:
                b, n, h, w = data['object_masks_pred'].shape
                mask_pred = (data['object_masks_pred'][i].view(
                    -1, w).data.cpu().numpy() > 0).astype('float32')
                mask_gt = data['object_masks'][i].view(-1,
                                                       w).data.cpu().numpy()
                cv2.imwrite(f"{dump_dir}/{prefix}_{identifier}_mask.png",
                            np.concatenate((mask_pred, mask_gt), 1) * 255)
            pass
        return tbd_figures

    def make_sdf_query_fn(self, data):
        def func(xyz):
            sdf_pred = self.sdf_decoder(xyz.unsqueeze(0), data)['sdf_pred']
            return sdf_pred[0, :]

        return func

    def make_sdf_query_grad_fn(self, data):
        def func(x):
            with torch.set_grad_enabled(True):
                x.requires_grad_(True)
                y = self.sdf_decoder(
                    x,
                    data,
                )['sdf_pred']
                d_output = torch.ones_like(y,
                                           requires_grad=False,
                                           device=y.device)
                gradients = torch.autograd.grad(outputs=y,
                                                inputs=x,
                                                grad_outputs=d_output,
                                                create_graph=True,
                                                retain_graph=True,
                                                only_inputs=True)[0]
            return gradients

        return func

    def transform_mesh(self, mesh_pred, mesh_name, data):
        if '%s_align_T' % mesh_name in data and '%s_align_scale' % mesh_name in data and mesh_pred is not None:
            mesh_pred.transform(
                np.linalg.inv(data['%s_align_T' %
                                   mesh_name].data.cpu().numpy()[0]))
            mesh_pred.scale(1. / data['%s_align_scale' % mesh_name],
                            center=(0, 0, 0))
        else:
            print('No Alignment Matrix Found')
        return mesh_pred

    def extract_mesh(self,
                     data,
                     mesh_fn='mesh_pred',
                     level=2,
                     th=None,
                     verbose=False,
                     align=False):
        assert (len(data['images']) == 1)
        sdf_query_fn = self.make_sdf_query_fn(data)
        th = 0.0 if th is None else th
        sign = 1
        corners = data['corners'][0]
        data[mesh_fn] = []
        data[mesh_fn].append(
            extract_mesh(sdf_query_fn,
                         corners,
                         threshold=th,
                         sign=sign,
                         level=level,
                         post_processing=False))
        if align:
            self.rigid_align_to_GT(data, mesh_name=mesh_fn, verbose=verbose)

    def compute_loss_mask(self, data, loss_scalars):
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss_mask = criterion(data['object_masks_pred'],
                              data['object_masks']) * 1e-2
        loss_scalars.update({
            'loss_mask': loss_mask.clone().detach().cpu(),
        })
        return loss_mask

    def compute_loss_sdf(self, data, loss_scalars):
        sdf_pred = data['sdf_pred']
        sdf_gt = data['sdf'][..., 3]
        clamp = self.config.TRAINER.SDF_CLAMP
        if clamp is not None:
            sdf_gt = sdf_gt.clamp(-clamp, clamp)
        sdf_pred_mask = data['sdf_pred_mask']
        loss_sdf = (sdf_pred - sdf_gt)[sdf_pred_mask].abs().mean()
        loss_scalars.update({
            'loss_sdf': loss_sdf.clone().detach().cpu(),
        })
        return loss_sdf

    def compute_loss_sdf_grad(self, data, loss_scalars):
        if self.config.TRAINER.NORMALIZE_NORMAL:
            normal_pred = F.normalize(data['sdf_grad_pred'], dim=-1)
        else:
            normal_pred = data['sdf_grad_pred']
        loss_eikonal = ((normal_pred - data['surface_point_normal'])**
                        2).mean() * self.config.TRAINER.EIKONAL_LOSS
        err_n = (torch.acos(
            (normal_pred * data['surface_point_normal']).sum(-1).clamp(-1, 1))
                 / np.pi * 180).mean()
        loss_scalars.update({
            'loss_eikonal': loss_eikonal.clone().detach().cpu(),
            'err_n': err_n.clone().detach().cpu(),
        })
        return loss_eikonal

    def compute_loss_pose(self, data, loss_scalars):
        loss_dummy = data['f0'].mean() * 0
        if data['poses_cur'] is not None and torch.all(
                torch.isfinite(data['poses_cur'])).item():
            loss_pose_r = (data['poses_cur'][:, :, :3, :3] -
                           data['poses'][:, :, :3, :3]).pow(2).mean() * 1e2
            loss_pose_t = (data['poses_cur'][:, :, :3, 3] -
                           data['poses'][:, :, :3, 3]).pow(2).mean() * 1e2
            loss_pose = loss_pose_r + loss_pose_t
            loss_dummy += loss_pose
            loss_scalars.update({
                'loss_pose':
                loss_pose.clone().detach().cpu(),
                'loss_pose_r':
                loss_pose_r.clone().detach().cpu(),
                'loss_pose_t':
                loss_pose_t.clone().detach().cpu(),
            })
        else:
            loss_scalars.update({
                'loss_pose': torch.zeros(1).float()[0],
            })
        return loss_dummy

    def compute_loss_bbox_scale(self, data, loss_scalars):
        bbox_delta = (self.config.TRAINER.BBOX_SCALE_MAX -
                      self.config.TRAINER.BBOX_SCALE_MIN) / (
                          self.config.TRAINER.BBOX_SCALE_NUM - 1)
        bbox_scale_gt = (
            (data['bbox_scale_gt'] - self.config.TRAINER.BBOX_SCALE_MIN) /
            bbox_delta).round().clamp(0, self.config.TRAINER.BBOX_SCALE_NUM -
                                      1).long()
        crit = nn.CrossEntropyLoss()
        loss = crit(data['scale_logits'], bbox_scale_gt)
        loss_scalars.update({
            'loss_bbox_scale': loss.clone().detach().cpu(),
        })
        return loss

    def loss(self, data, mode='train'):
        loss, loss_scalars = 0, {}
        if self.config.TRAINER.TRAIN_SCALE:
            loss += self.compute_loss_bbox_scale(data, loss_scalars)

        if self.config.TRAINER.TRAIN_MASK:
            loss += self.compute_loss_mask(data, loss_scalars)

        if self.config.TRAINER.TRAIN_SHAPE:
            loss += self.compute_loss_sdf(data, loss_scalars)
            if self.config.TRAINER.EIKONAL_LOSS > 0:
                loss += self.compute_loss_sdf_grad(data, loss_scalars)

        if self.config.TRAINER.TRAIN_POSE:
            loss += self.compute_loss_pose(data, loss_scalars)

        loss_scalars.update({
            'loss': loss.clone().detach().cpu(),
        })
        data.update({"loss": loss, "loss_scalars": loss_scalars})

    def extract_feature(self, data):
        assert ('images' in data)
        x = data['images']
        features = []
        b, n, c, h, w = x.shape
        features = self.feature(x.view(b * n, -1, h, w))
        features = features.view(b, n, *features.shape[1:])

        data['feature_map'] = features

    def make_grid(self, bbox_scale, b, m):
        """
        Args:
            bbox_scale Tensor [b]
        Returns:
            Tensor [b,m,dim,dim,dim,3]
        """
        dim = self.grid_dim
        xs = torch.linspace(-0.5, 0.5, dim)
        zs, ys, xs = torch.meshgrid(xs, xs, xs)
        grid = torch.stack((xs, ys, zs), -1)
        grid = grid.view(1, 1, dim, dim, dim, 3).to(bbox_scale.device).float()
        grid = grid.repeat(b, m, 1, 1, 1, 1) * bbox_scale.view(
            b, 1, 1, 1, 1, 1)
        return grid

    def build_volume(self, data):
        """Build volumetric feature volume based on current camera pose.
        Args:
            data['bbox_scale'] Tensor [b]: The scale of object (predicted or heuristic defined).
            data['proj'] Tensor [b, m, 4, 4]: The projection matix for each view.
            data['feature_map'] Tensor [b, m, 4, c, h, w]: The feaature map for each view.
        Returns:
            data['volume'] Tensor [b, c, d, d, d]: The volumetric feature grid.
            data['grid'] Tensor [b, m, d, d, d, 3]: The volumetric grid which was used for building volumetric feature.
            data['corners'] Tensor [b, 2, 3]: The corners for the grid.
        """
        proj = data['proj']
        feature_map = data['feature_map']
        b, n, c, h, w = feature_map.shape
        dim = self.grid_dim
        device = feature_map.device
        scale = data['bbox_scale'].float()
        data['grid'] = self.make_grid(scale, b, n)
        data['corners'] = torch.FloatTensor([[-0.5, -0.5, -0.5], [
            0.5, 0.5, 0.5
        ]]).to(device).view(1, 2, 3) * scale.float().view(b, 1, 1)
        uv, z = self.make_uv_grid(proj, data['grid'], w, h)
        volume = F.grid_sample(feature_map.view(b * n, -1, h, w),
                               uv.view(b * n, -1, 1, 2),
                               padding_mode='zeros',
                               align_corners=True).view(
                                   b, n, c, dim, dim, dim)
        volume = self.volume_reduction(volume)
        volume = self.cost_regularization(volume)
        data['volume'] = volume

    def volume_reduction(self, volume):
        if self.config.COST_REG_NET.VOLUME_REDUCTION == 'mean':
            return volume.mean(1)
        elif self.config.COST_REG_NET.VOLUME_REDUCTION == 'variance':
            b, n = volume.shape[:2]
            volume_sum = volume.sum(1)
            volume_sq_sum = (volume**2).sum(1)
            volume_variance = volume_sq_sum.div_(n).sub_(
                volume_sum.div_(n).pow_(2))
            volume_in = volume_variance
            del volume_sum, volume_sq_sum, volume_variance
            return volume_in
        elif self.config.COST_REG_NET.VOLUME_REDUCTION == 'mean+variance':
            b, n = volume.shape[:2]
            volume_sum = volume.sum(1)
            volume_sq_sum = (volume**2).sum(1)
            volume_variance = volume_sq_sum.div_(n).sub_(
                volume_sum.div_(n).pow_(2))
            volume_in = volume_variance
            del volume_sq_sum, volume_variance
            return torch.cat((volume_in, volume_sum), 1)

    def make_uv_grid(self, proj, grid, w, h):
        b, n = grid.shape[:2]
        uvd = grid.view(b, n, -1, 3) @ proj[:, :, :3, :3].transpose(
            2, 3) + proj[:, :, :3, 3].unsqueeze(2)
        Z = uvd[:, :, :, 2]
        uv = uvd[:, :, :, :2] / uvd[:, :, :, 2:3]
        uv[:, :, :, 0] = (uv[:, :, :, 0] / (w - 1) * 2 - 1)
        uv[:, :, :, 1] = (uv[:, :, :, 1] / (h - 1) * 2 - 1)
        valid_uv = (uv[:,:,:,0].abs() <= 1) & \
                (uv[:,:,:,1].abs() <= 1) & \
                            (Z > 0)

        X_mask = ((uv[:, :, :, 0] > 1) + (uv[:, :, :, 0] < -1)).detach()
        uv[:, :, :, 0][X_mask] = 2
        Y_mask = ((uv[:, :, :, 1] > 1) + (uv[:, :, :, 1] < -1)).detach()
        uv[:, :, :, 1][Y_mask] = 2
        return uv, Z

    def render_volume(self, poses, data):
        """Render implicit shape representation into depth map and 3D point clouds.
        Args:
            poses (Tensor [b, n, 4, 4]): The current camera pose for each view.
            data['sdf_vol'] (Tensor [b, 1, d, d, d]): Current implicit shape representation. Note that the implicit grid must be compatible with data['corners'].
            data['corners'] (Tensor [b, 2, 3]): The corners of current implicit shape representation.
            data['intriniscs'] (Tensor [b, n, 4, 4]): The camera intrinsic for each view.
        Returns:
            data['depth_render'] (Tensor [b, m, h, w]): The rendered depth map for each camear pose.
            data['pixel_uv_3d'] (Tensor [b, m, h*w, 3]): The rendered point cloud for each camera pose
        """
        b, n, _, h, w = data['images'].shape
        dim = self.grid_dim
        poses_cam2world = self.inverse(poses)
        uv = self.image_grid.unsqueeze(0).repeat(b * n, 1, 1).cuda()
        ray_dirs, cam_loc = get_camera_params(
            uv, poses_cam2world.view(-1, 4, 4),
            data['intrinsics'].view(-1, 4, 4))

        cam_loc = cam_loc.view(b, n, 1, 3).repeat(1, 1, uv.shape[1],
                                                  1).view(b * n, -1, 3)
        rays = torch.cat((cam_loc, ray_dirs), -1)
        rays = rays[:, :, [2, 1, 0, 5, 4, 3]]
        depth, feature_map, boundary = SDFRender.apply(
            data['sdf_vol'][:, 0],
            torch.zeros([b, dim, dim, dim, 3]).cuda(), data['corners'][:, 0],
            data['corners'][:, 1] - data['corners'][:, 0], rays.view(b, -1, 6))
        points = rays[:, :, :3] + rays[:, :, 3:6] * depth.view(b * n, -1, 1)
        points = points[:, :, [2, 1, 0]]
        data['pixel_uv_3d'] = points.view(b, n, -1, 3)
        depth[depth == -1] = 0
        depth = depth.view(b, n, h, w)
        data['depth_render'] = depth

    def inverse(self, pose):
        R = pose[:, :, :3, :3]
        t = pose[:, :, :3, 3:4]
        R_new = R.transpose(2, 3)
        t_new = -R_new @ t
        pose_new = torch.zeros_like(pose)
        pose_new[:, :, :3, :3] = R_new
        pose_new[:, :, :3, 3:4] = t_new
        pose_new[:, :, 3, 3] = 1
        return pose_new

    def visualize(self, data, postfix='', write=False, dump_dir='./'):
        depth_render = data['depth_render']
        b, n, h, w = depth_render.shape
        images = []
        for j in range(n):
            img = data['images'][0, j, :3].data.cpu().numpy().transpose(
                1, 2, 0) * 255
            tp = depth_render[0, j].view(h, w)
            tp[tp == -1] = 0
            mask = (tp == 0).data.cpu().numpy()
            tp = (tp.data.cpu().numpy() > 0).astype('float32') * 255
            blend = tp[:, :, None] * np.array([1, 1, 1]).reshape(
                1, 1, 3).astype('float32')
            blend = cv2.addWeighted(img, 0.3, blend, 0.7, 0)
            blend[mask] = img[mask]
            images.append(tp)
            if write:
                cv2.imwrite(f"{dump_dir}/opt%d_%s_blend.png" % (j, postfix),
                            blend)
                cv2.imwrite(f"{dump_dir}/opt%d_%s_mask.png" % (j, postfix), tp)
                cv2.imwrite(f"{dump_dir}/opt%d_%s_img.png" % (j, postfix), img)
        return images

    def update_pose(self, data, verbose=False):
        b, n, h, w = data['depth_render'].shape
        inp0 = 1 - data['images'][:, :, :3].mean(2).view(b * n, 1, h, w)
        inp1 = (data['depth_render'] > 0).float().view(b * n, 1, h, w)
        data['f0'] = self.flownet(inp0)
        data['f1'] = self.flownet(inp1)
        weight = (data['depth_render'] > 0).view(b, n, -1)
        outputs = self.pnp(
            data['pixel_uv_3d'].detach(),
            data['f1'].view(b, n, -1, h * w).transpose(-2, -1),
            data['f0'].view(b, n, -1, h, w),
            weight,
            data['intrinsics'],
            data['poses_cur'],
            data['poses'], {},
            T_init=data['poses_init'],
            use_double=self.config.POSE_REFINE.USE_DOUBLE,
            clamp=self.config.POSE_REFINE.CLAMP_POSE_UPDATE,
            lambda_reg=self.config.POSE_REFINE.LAMBDA_REG,
            lambda_damping=self.config.POSE_REFINE.LAMBDA_DAMPING,
            verbose=verbose)
        data.update(
            dict(poses_cur=outputs['pose_pnp'], energy=outputs['energy']))

    def joint_update_pose_and_shape(self,
                                    data,
                                    write=False,
                                    dump_dir='./',
                                    verbose=False):
        b, n, c, h, w = data['images'].shape
        dim = self.grid_dim
        data['poses_cur'] = data['poses_noise@PoseModule'].clone()
        data['poses_init'] = data['poses_cur'].clone()
        with torch.no_grad():
            data['proj'] = data['intrinsics'] @ data['poses_cur']
            self.build_volume(data)
            self.extract_mesh(data, mesh_fn='mesh_init', th=0.0)
            if data['mesh_init'][0] is not None:
                o3d.io.write_triangle_mesh(f"{dump_dir}/mesh_init.ply",
                                           data['mesh_init'][0])
        mesh_fn = 'mesh_step_%d' % 0
        align = self.config.TRAINER.RIGID_ALIGN_TO_GT
        data['poses_cur_step_%d' % 0] = data['poses_cur'].clone()
        self.extract_mesh(data,
                          mesh_fn=mesh_fn,
                          th=0.0,
                          verbose=verbose,
                          align=align)

        with torch.no_grad():
            for step in range(self.config.POSE_REFINE.JOINT_STEP):
                poses_last = data['poses_cur'].clone()
                if verbose:
                    print('step ', step)

                ### Decode sdf volume
                data['sdf_vol'] = self.sdf_decoder(
                    data['grid'][:, 0].contiguous().view(1, -1, 3),
                    data)['sdf_pred'].view(b, 1, dim, dim, dim)

                ### Update pose
                for i in range(self.config.POSE_REFINE.POSE_UPDATE_INNER_ITER):
                    self.render_volume(data['poses_cur'], data)
                    data['step_%d_%d' % (step + 1, i)] = self.visualize(
                        data,
                        postfix='joint_%d_step%d' % (step + 1, i),
                        dump_dir=dump_dir,
                        write=write)
                    self.update_pose(data, verbose=verbose)
                    if i == 0:
                        energy_init = data['energy']
                energy_cur = data['energy']
                if energy_cur > energy_init:
                    ### Drop this update
                    if self.config.POSE_REFINE.DROP_FOR_INCREASING_ENERGY:
                        print(
                            'drop this update because increasing energy value')
                        data['poses_cur'] = poses_last
                data['energy_step_%d' % (step + 1)] = energy_cur
                data['poses_cur_step_%d' %
                     (step + 1)] = data['poses_cur'].clone()

                ### Build shape volume
                data['proj'] = data['intrinsics'] @ data['poses_cur']
                self.build_volume(data)

                ### Decode mesh
                mesh_fn = 'mesh_step_%d' % (step + 1)
                align = self.config.TRAINER.EVAL_PER_JOINT_STEP and self.config.TRAINER.RIGID_ALIGN_TO_GT
                self.extract_mesh(data,
                                  mesh_fn=mesh_fn,
                                  th=0.0,
                                  verbose=verbose,
                                  align=align)

                if write:
                    if data[mesh_fn][0] is not None:
                        o3d.io.write_triangle_mesh(
                            f"{dump_dir}/mesh_step{step+1}.ply",
                            data[mesh_fn][0])

            self.extract_mesh(data,
                              mesh_fn='mesh_final',
                              verbose=verbose,
                              align=self.config.TRAINER.RIGID_ALIGN_TO_GT)
            if data['mesh_final'][0] is not None:
                if write:
                    if data['mesh_final'][0] is not None:
                        o3d.io.write_triangle_mesh(
                            f"{dump_dir}/mesh_final.ply",
                            data['mesh_final'][0])
            else:
                print('mesh is None')
        if write:
            pngs = sorted(glob.glob(f"{dump_dir}/*blend*png"))
            if len(pngs):
                create_gif(img_list=pngs,
                           save_file=f"{dump_dir}.gif",
                           duration=70,
                           loop=0)
            cmd = f"rm -r {dump_dir}/*png"
            os.system(cmd)

    def get_cameras_accuracy(self, poses, poses_gt):
        pred_Rs = poses[:, :3, :3].double().cpu()
        pred_ts = poses[:, :3, 3].double().cpu()
        gt_Rs = poses_gt[:, :3, :3].double().cpu()
        gt_ts = poses_gt[:, :3, 3].double().cpu()
        try:
            result = get_cameras_accuracy(pred_Rs,
                                          gt_Rs,
                                          pred_ts,
                                          gt_ts,
                                          verbose=False)
            return result
        except Exception as e:
            print(e)
            return None

    def compute_metric_sdf(self, sdf_pred, sdf_gt, metrics):
        sdf_pred = sdf_pred.data.cpu().numpy()
        sdf_gt = sdf_gt.data.cpu().numpy()
        clamp = self.config.TRAINER.SDF_CLAMP
        if clamp is not None:
            sdf_gt = sdf_gt.clip(-clamp, clamp)
        metrics['err_sdf'] = np.abs(sdf_pred - sdf_gt).mean(-1)

    def compute_metric_pseudo_IoU(self, sdf_pred, sdf_gt, metrics):
        intersection = (sdf_pred < 0) * (sdf_gt < 0)
        union = ((sdf_pred < 0) + (sdf_gt < 0)) > 0
        metrics['PseudoIoU'].append(
            ((sdf_pred < 0) * (sdf_gt < 0)).sum(-1) /
            (((sdf_pred < 0) + (sdf_gt < 0)) > 0).sum(-1))

    def compute_metric_scale(self, mesh_pred, point_gt, metrics):
        verts_pred = np.array(mesh_pred.vertices)
        verts_pred -= verts_pred.mean(0)
        scale_pred = np.std(verts_pred)
        verts_gt = point_gt.data.cpu().numpy()
        verts_gt -= verts_gt.mean(0)
        scale_gt = np.std(verts_gt)
        scale_err = np.abs(scale_pred / scale_gt - 1.0)
        metrics['scale_err'].append(scale_err)

    def compute_metric_bbox_scale(self, scale_pred, scale_gt, metrics):
        scale_err = (scale_pred - scale_gt).abs().data.cpu().numpy().tolist()
        abs_rel_scale_err = ((scale_pred - scale_gt).abs() /
                             scale_gt).data.cpu().numpy().tolist()
        metrics['bbox_scale_abs_err'] = scale_err
        metrics['bbox_scale_abs_rel_err'] = abs_rel_scale_err

    def compute_metric_pose(self, pose_name, data, metrics):
        result0 = self.get_cameras_accuracy(data['poses_noise@PoseModule'][0],
                                            data['poses'][0])
        result1 = self.get_cameras_accuracy(data[pose_name][0],
                                            data['poses'][0])
        if result0 is not None and result1 is not None:
            err_R_percentage = result1['R_errors'].mean(
            ) / result0['R_errors'].mean()
            err_t_percentage = result1['t_errors'].mean(
            ) / result0['t_errors'].mean()
            metrics['err_R_percentage'].append(err_R_percentage)
            metrics['err_t_percentage'].append(err_t_percentage)
            metrics['err_t'].append(np.mean(result1['t_errors']))
            metrics['err_R'].append(np.mean(result1['R_errors']))
        else:
            metrics['err_R_percentage'].append(1.0)
            metrics['err_t_percentage'].append(1.0)
            metrics['err_t'].append(np.mean(result0['t_errors']))
            metrics['err_R'].append(np.mean(result0['R_errors']))

        b, n = data['poses'].shape[:2]
        T_aligned = torch.eye(4).view(1, 1, 4, 4).repeat(b, n, 1,
                                                         1).float().cuda()
        T_aligned[:, :, :3, :3] = torch.from_numpy(result1['R_fixed']).view(
            b, n, 3, 3).float().cuda()
        T_aligned[:, :, :3, 3] = torch.from_numpy(result1['t_fixed']).view(
            b, n, 3).float().cuda()
        data['%s_aligned' % pose_name] = T_aligned

    def compute_metric_pixel(self, pose_name, data, metrics):
        metrics['avg_noise_r'].append(
            data['poses_noise@PoseModule_stat']['noise_r'].mean().item())
        metrics['avg_noise_t'].append(
            data['poses_noise@PoseModule_stat']['noise_t'].mean().item())
        metrics['avg_noise_pixel'].append(
            self.compute_noise_magnitude(
                data, pose_name='poses_noise@PoseModule').item())

        metrics['avg_noise_pixel_cur'].append(
            self.compute_noise_magnitude(data,
                                         pose_name='%s_aligned' %
                                         pose_name).item())

    def compute_metrics(self,
                        data,
                        mesh_name,
                        mode='train',
                        pose_name='poses_cur'):
        metrics = defaultdict(list)

        ### Shape metrics
        if self.config.TRAINER.TRAIN_SHAPE:
            if self.config.TRAINER.TRAIN_SCALE:
                self.compute_metric_bbox_scale(data['bbox_scale_pred'],
                                               data['bbox_scale_gt'], metrics)
            if not self.config.TRAINER.USE_OCC:
                self.compute_metric_sdf(data['sdf_pred'], data['sdf'][..., 3],
                                        metrics)

            xyz_all, sdf_gt = self.make_evaluation_data(data)
            sdf_query_fn = self.make_sdf_query_fn(data)
            sdf_pred = sdf_query_fn(xyz_all).data.cpu().numpy()
            self.compute_metric_pseudo_IoU(sdf_pred, sdf_gt, metrics)

            if mode == 'test':
                n = len(data[mesh_name])
                assert (n == 1 and not self.config.TRAINER.USE_OCC)
                try:
                    mesh_pred = data[mesh_name][0]
                    if self.config.TRAINER.RIGID_ALIGN_TO_GT:
                        mesh_pred = self.transform_mesh(
                            mesh_pred, mesh_name, data)
                    self.compute_metric_scale(mesh_pred,
                                              data['surface_point'][0],
                                              metrics)
                    mesh_pred = o3d_mesh_to_trimesh(mesh_pred)
                    out_dict = eval_mesh(
                        mesh_pred, data['surface_point'][0].data.cpu().numpy(),
                        data['surface_point_normal'][0].data.cpu().numpy(),
                        xyz_all.data.cpu().numpy(),
                        (sdf_gt < 0).astype('float'))
                    metrics.update({
                        'chamfer-L1': [out_dict['chamfer-L1']],
                        'chamfer-L2': [out_dict['chamfer-L2']],
                        'IoU': [out_dict['iou']],
                        'Normal-Consistency': [out_dict['normals']]
                    })
                except:
                    metrics.update({
                        'scale_err': [1.0],
                        'chamfer-L1': [np.nan],
                        'chamfer-L2': [np.nan],
                        'IoU': [0],
                        'Normal-Consistency': [-1]
                    })
        ### Pose metrics
        if mode == 'test':
            if self.config.TRAINER.CAMERA_POSE != 'gt':
                self.compute_metric_pose(pose_name, data, metrics)
                self.compute_metric_pixel(pose_name, data, metrics)
        return metrics

    def make_evaluation_data(self, data):
        if self.config.DATASET.name == 'shapenet':
            xyz_all = data['sdf_all'][0, :, :3]
            sdf_gt = data['sdf_all'][0, :, 3].data.cpu().numpy()
        elif self.config.DATASET.name == 'hm3d_abo':
            xyz_all = self.make_grid(data['bbox_scale_gt'].float(), 1,
                                     1)[0, 0].view(-1, 3)
            tree = KDTree(data['sdf_all'][0, :, :3].data.cpu().numpy())
            dst, idx = tree.query(xyz_all.data.cpu().numpy())
            sdf_gt = data['sdf_all'][0, :, 3].data.cpu().numpy()[idx]
        else:
            raise Exception('Unknown Dataset')
        return xyz_all, sdf_gt

    def compute_noise_magnitude(self,
                                data,
                                pose_name='poses_noise@PoseModule'):
        xyz = data['surface_point'].unsqueeze(
            1) @ data['poses'][:, :, :3, :3].transpose(
                -2, -1) + data['poses'][:, :, :3, 3].unsqueeze(-2)
        xyz = xyz @ data['intrinsics'][:, :, :3, :3].transpose(-2, -1)
        xyz = xyz[:, :, :, :2] / xyz[:, :, :, 2:3]

        xyz_noise = data['surface_point'].unsqueeze(
            1) @ data[pose_name][:, :, :3, :3].transpose(
                -2, -1) + data[pose_name][:, :, :3, 3].unsqueeze(-2)
        xyz_noise = xyz_noise @ data['intrinsics'][:, :, :3, :3].transpose(
            -2, -1)
        xyz_noise = xyz_noise[:, :, :, :2] / xyz_noise[:, :, :, 2:3]
        pixel_noise = torch.norm(xyz - xyz_noise, dim=-1).mean()
        return pixel_noise

    @torch.enable_grad()
    def rigid_align_to_GT(self, data, mesh_name='mesh_final', verbose=False):
        b, n, c, h, w = data['images'].shape
        xyz = data['surface_point']
        sdf_query_fn = self.make_sdf_query_fn(data)
        N = 200
        LR = self.config.POST_PROCESSING.JOINT_LR
        pose_vecs = torch.nn.Embedding(b, 7, sparse=True).cuda()
        pose_vecs.weight.data.zero_()
        pose_vecs.weight.data[:, 0] = 1  # identity matrix
        assert (b == 1)
        ## initial scale estimate
        box_scale0 = torch.norm((xyz[0].max(0)[0] - xyz[0].min(0)[0])).item()
        if self.config.POST_PROCESSING.OPTIMIZE_SCALE:
            if data[mesh_name][0] is not None:
                xyz1 = np.array(data[mesh_name][0].vertices)
                box_scale1 = np.linalg.norm((xyz1.max(0) - xyz1.min(0)))
                scale_init = box_scale1 / box_scale0
                scale_init = np.clip(scale_init, 0.8, 1.2)
            else:
                scale_init = 1.0
            scale = scale_init * torch.ones(b).float().cuda()
            scale.requires_grad = True
            optimizer = torch.optim.Adam(
                list(pose_vecs.parameters()) + [scale], LR)
        else:
            optimizer = torch.optim.Adam(list(pose_vecs.parameters()), LR)
        for i in range(N):
            optimizer.zero_grad()
            R = quat_to_rot(pose_vecs.weight[:, :4])
            t = pose_vecs.weight[:, 4:]
            if self.config.POST_PROCESSING.OPTIMIZE_SCALE:
                xyz_t = xyz * scale.view(b, 1, 1)
            else:
                xyz_t = xyz
            xyz_t = xyz_t @ R.transpose(-2, -1) + t.unsqueeze(-2)
            sdf_pred = self.sdf_decoder(xyz_t, data)['sdf_pred']
            loss = sdf_pred.abs().mean()
            loss.backward()
            optimizer.step()
            if i % 50 == 0 and verbose:
                print(i, loss.item())
        R = quat_to_rot(pose_vecs.weight[:, :4])
        t = pose_vecs.weight[:, 4:]
        T = torch.eye(4).view(1, 4, 4).repeat(b, 1, 1).float().cuda()
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        data['%s_align_T' % mesh_name] = T
        if self.config.POST_PROCESSING.OPTIMIZE_SCALE:
            data['%s_align_scale' % mesh_name] = scale[0].item()
        else:
            data['%s_align_scale' % mesh_name] = 1.0

    def forward(self, data, mode='train'):
        b, n, c, h, w = data['images'].shape
        dim = self.grid_dim

        ### Extract feature
        self.extract_feature(data)

        ### Build projection matrice
        if self.config.TRAINER.CAMERA_POSE == 'gt':
            data['poses_cur'] = data['poses']
        else:
            data['poses_cur'] = data['poses_noise@ShapeModule']
        data['proj'] = data['intrinsics'] @ data['poses_cur'].detach()

        ### Build volumetrics representation
        self.build_volume(data)

        ### Decode pose
        if self.config.TRAINER.TRAIN_POSE:
            with torch.no_grad():
                data['sdf_vol'] = self.sdf_decoder(
                    data['grid'][:, 0].contiguous().view(b, -1, 3),
                    data)['sdf_pred'].view(b, 1, dim, dim, dim)
            data['poses_cur'] = data['poses_noise@PoseModule']
            data['poses_init'] = data['poses_cur'].clone()
            self.render_volume(data['poses_cur'], data)
            self.update_pose(data)

        ### Decode shape
        if self.config.TRAINER.TRAIN_SHAPE:
            data.update(self.sdf_decoder(data['sdf'][..., :3], data))
            if self.config.TRAINER.EIKONAL_LOSS > 0:
                sdf_query_grad_fn = self.make_sdf_query_grad_fn(data)
                data['sdf_grad_pred'] = sdf_query_grad_fn(
                    data['surface_point'])

        ### Decode mask
        if self.config.TRAINER.TRAIN_MASK:
            b, n, c, h, w = data['feature_map'].shape
            data['object_masks_pred'] = self.mask_head(
                data['feature_map'].view(b * n, c, h, w)).view(b, n, h, w)

        return data
