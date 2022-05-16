import torch
import cv2
import torch.nn as nn
from einops.einops import rearrange
from .position_encoding import PositionEncodingSine
from .transformer import LocalFeatureTransformer
from .modules import *
import numpy as np
from utils import (
    axisangle_to_rotmat,
    Quaternion2rot,
    rot2Quaternion,
    lower_config,
    opencv_ransacPnP,
)
from collections import defaultdict
from .resnet_fpn import ResNetFPN_8_2v2


class model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = ResNetFPN_8_2v2(config.POSE_INIT.BACKBONE)
        self.pos_encoding = PositionEncodingSine(
            config.POSE_INIT.D_COARSE,
            frame_embed=self.config.POSE_INIT.FRAME_EMBED)
        self.d_coarse = config.POSE_INIT.D_COARSE
        self.loftr_coarse = LocalFeatureTransformer(
            lower_config(config.POSE_INIT.CROSS_ATTENTION))
        if self.config.TRAINER.TRAIN_MASK:
            self.mask_head = build_head(64, 1)
        if self.config.TRAINER.TRAIN_SCALE:
            self.conv_scale1 = build_head(256, 64, nhidden=64)
            self.conv_scale2 = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, self.config.TRAINER.BBOX_SCALE_NUM))

        self.conv_scene_coord = nn.Sequential(
            conv3x3(config.POSE_INIT.D_FINE, 64), nn.ReLU(), conv3x3(64, 64),
            nn.ReLU(), conv1x1(64, 3))
        self.conv1 = conv3x3(256, 64)

    def get_pose_loss(self, data, loss_scalars):
        scene_coord_gt = data['scene_coords_half']
        scene_coord_pred = data['scene_coord_pred']
        mask = data['masks_half']
        b, n, _, h, w = scene_coord_gt.shape
        weight = mask.float() * 10 + (1 - mask.float())
        loss = ((data['scene_coord_pred'] - scene_coord_gt).norm(dim=2) *
                weight).sum() / (weight.sum() + 1e-16)

        loss_scalars.update({'loss_scene_coord': loss.clone().detach().cpu()})
        return loss

    def compute_loss_mask(self, data, loss_scalars):
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        object_masks_half = F.interpolate(data['object_masks'],
                                          scale_factor=0.5)
        loss_mask = criterion(data['object_masks_pred'],
                              object_masks_half) * 1e-2
        loss_scalars.update({
            'loss_mask': loss_mask.clone().detach().cpu(),
        })
        return loss_mask

    def compute_loss_bbox_scale(self, data, loss_scalars):
        bbox_scale_min = self.config.TRAINER.BBOX_SCALE_MIN
        bbox_scale_max = self.config.TRAINER.BBOX_SCALE_MAX
        num_bin = self.config.TRAINER.BBOX_SCALE_NUM
        bin_size = (bbox_scale_max - bbox_scale_min) / (num_bin - 1)
        scale_gt = torch.round(
            (data['bbox_scale_gt'] - bbox_scale_min) / bin_size).clamp(
                0, num_bin - 1).long()
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        scale_pred = data['bbox_scale_pred']
        loss_bbox_scale = criterion(scale_pred, scale_gt) * 1e-2
        loss_scalars.update({
            'loss_bbox_scale':
            loss_bbox_scale.clone().detach().cpu(),
        })
        return loss_bbox_scale

    def loss(self, data, mode='train'):
        loss_scalars = {}

        loss = self.get_pose_loss(data, loss_scalars)
        if self.config.TRAINER.TRAIN_MASK:
            loss += self.compute_loss_mask(data, loss_scalars)
        if self.config.TRAINER.TRAIN_SCALE:
            loss += self.compute_loss_bbox_scale(data, loss_scalars)
        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})

    def compute_metrics(self, data, mode='test'):
        metrics = dict()
        metrics.update(self.compute_cam_pose_ransac(data, data['poses']))
        metrics.update(self.evaluate_scene_coord(data))
        if self.config.TRAINER.TRAIN_SCALE:
            metrics.update(self.evaluate_bbox(data))
        metrics.update(
            self.evaluate_camera_pose(data['T_pred'], data['poses'],
                                      data['surface_point'],
                                      data['intrinsics']))
        return metrics

    def evaluate_bbox(self, data):
        """Evaluate bounding box prediction.
        Args: 
            data['bbox_scale_gt'] (Tensor [b]) The ground truth scale of object bounding box.
            data['bbox_scale_pred'] (Tensor [b]) The predicted object bounding box logits.
        Return:
            metrics['bbox_scale_abs_err']: The absolute scale error.
            metrics['bbox_scale_abs_rel_err']: The absolute relative scale error.
        """
        bbox_scale_min = self.config.TRAINER.BBOX_SCALE_MIN
        bbox_scale_max = self.config.TRAINER.BBOX_SCALE_MAX
        num_bin = self.config.TRAINER.BBOX_SCALE_NUM
        bin_size = (bbox_scale_max - bbox_scale_min) / (num_bin - 1)
        scale_gt = data['bbox_scale_gt']
        scales = torch.linspace(bbox_scale_min, bbox_scale_max,
                                num_bin).to(scale_gt)
        index = torch.argmax(data['bbox_scale_pred'], -1)
        scale_pred = scales[index]
        data['scale_pred'] = scale_pred
        bbox_scale_abs_err = (scale_pred -
                              scale_gt).abs().data.cpu().numpy().tolist()
        bbox_scale_abs_rel_err = ((scale_pred - scale_gt).abs() /
                                  scale_gt).data.cpu().numpy().tolist()
        return dict(bbox_scale_abs_err=bbox_scale_abs_err,
                    bbox_scale_abs_rel_err=bbox_scale_abs_rel_err)

    def evaluate_camera_pose(self, T_pred, T_gt, surface_point, intrinsics):
        """Evaluate camera poses prediction.
        Args: 
            T_pred (Tensor [b, m, 4, 4]) The predicted camera poses.
            T_gt (Tensor [b, m, 4, 4]) The ground truth camera poses.
            surface_point (Tensor [b, n, 3]) The surface point cloud of the underlying object.
            intrinsics (Tensor [b, n, 4, 4]) The camera intrinsics for each view.
        Return:
            metrics['pose_err_R']: The rotation error for each data. 
            metrics['pose_err_t']: The translation error for each data. 
            metrics['pixel_err']: The mean pixel-wise reprojection error. 
        """
        metrics = defaultdict(list)
        b, n = T_pred.shape[:2]
        for bi in range(b):
            err_R, err_t = 0, 0
            for i in range(n):
                R_gt = T_gt[bi, i][:3, :3]
                t_gt = T_gt[bi, i][:3, 3]
                R_pred = T_pred[bi, i, :3, :3]
                t_pred = T_pred[bi, i, :3, 3]
                t_err = torch.norm(t_pred - t_gt).item()
                r_err = angular_distance_np(R_pred.data.cpu().numpy(),
                                            R_gt.data.cpu().numpy())[0]
                err_R += r_err
                err_t += t_err
            err_R /= n
            err_t /= n
            metrics['pose_err_R'].append(err_R)
            metrics['pose_err_t'].append(err_t)

        xyz = surface_point.unsqueeze(1) @ T_gt[:, :, :3, :3].transpose(
            -2, -1) + T_gt[:, :, :3, 3].unsqueeze(-2)
        xyz = xyz @ intrinsics[:, :, :3, :3].transpose(-2, -1)
        xyz = xyz[:, :, :, :2] / xyz[:, :, :, 2:3]

        xyz_noise = surface_point.unsqueeze(
            1) @ T_pred[:, :, :3, :3].transpose(
                -2, -1) + T_pred[:, :, :3, 3].unsqueeze(-2)
        xyz_noise = xyz_noise @ intrinsics[:, :, :3, :3].transpose(-2, -1)
        xyz_noise = xyz_noise[:, :, :, :2] / xyz_noise[:, :, :, 2:3]
        pixel_err = torch.norm(xyz - xyz_noise,
                               dim=-1).mean([-2,
                                             -1]).data.cpu().numpy().tolist()
        metrics['pixel_err'] = pixel_err
        return metrics

    def evaluate_scene_coord(self, data):
        """Evaluate scene coordinate prediction.
        Args: 
            data['scene_coords_half'] (Tensor [b, n, 3, h, w]) The ground truth scene coordinate at half of input resolution.
            data['scene_coords_pred'] (Tensor [b, n, 3, h, w]) The predicted scene coordinate at half of input resolution.
            data['masks_half'] (Tensor [b, n, 1, h, w]) The ground truth scene coordinate mask. (For ShapeNet, the background pixel does not have ground truth scene coordinate)
        Return:
            metrics['scene_coord_err']: scene coordinate L1 error for each example.
            metrics['scene_coord_err_masked']: scene coordinate L1 error for each example if only counted valid region.
        """
        metrics = defaultdict(list)
        scene_coord_gt = data['scene_coords_half']
        mask = data['masks_half']
        err = torch.norm(data['scene_coord_pred'] - scene_coord_gt, dim=2)
        err_masked = (err * mask).sum(
            dim=[1, 2, 3]) / (mask.sum(dim=[1, 2, 3]) + 1e-16)
        metrics['scene_coord_err_masked'] = [x.item() for x in err_masked]
        return metrics

    def dump_vis(self, data, dump_dir, prefix, metrics=None):
        if 'test' in prefix:
            np.save(f"{dump_dir}/{data['bag_names'][0]}.npy",
                    data['T_pred'][0].data.cpu().numpy())
        tbd_figures = {}
        return tbd_figures

    def make_prediction(self, feats_c, feats_f, data):
        b, n, _, hf, wf = feats_f.shape
        feats_c = self.conv1(F.interpolate(feats_c, scale_factor=4))
        feats_f = feats_f + feats_c.view(b, n, -1, hf, wf)
        self.decode_scene_coord(feats_f, data)

    def cross_attention(self, feats_c):
        b, n, _, hc, wc = feats_c.shape
        feats_c_new = []
        for i in range(n):
            feats_c_new.append(
                rearrange(self.pos_encoding(feats_c[:, i]),
                          'n c h w -> n (h w) c'))
        feats_c = torch.stack(feats_c_new, dim=1)
        feats_c = self.loftr_coarse(feats_c, None)
        feats_c = feats_c.view(b * n, hc, wc,
                               self.d_coarse).permute(0, 3, 1, 2).contiguous()
        return feats_c

    def compute_cam_pose_ransac(self, data, poses_gt):
        """Compute camera pose from scene coordiate prediction using RANSAC.
        Args:
            data['scene_coord_pred'] (Tensor [b, m, 3, h, w]): Predicted scene coordinate for each view.
            data['object_masks_pred'] (Tensor [b, m, h, w]): Predicted object mask for each view. This is only useful for ShapeNet dataset as in this case the non-object scene coordiante is not well defined..
            data['intrinsics'] (Tensor [b, n, 4, 4]) The camera intrinsics for each view.
        Returns:
            data['T_pred'] (Tensor [b, m, 4, 4]): camera to world matrix for each view.
        """
        scene_coord_pred = data['scene_coord_pred']
        b, n, _, h, w = scene_coord_pred.shape
        assert (b == 1)
        metrics = defaultdict(list)
        data['T_pred'] = torch.eye(4).view(1, 1, 4,
                                           4).repeat(b, n, 1,
                                                     1).float().cuda()
        for bi in range(b):
            err_R, err_t = 0, 0
            for i in range(n):
                out_pose = torch.zeros((4, 4))
                K = data['intrinsics'][bi, i][:3, :3]
                if self.config.DATASET.name == 'shapenet':
                    mask = (data['object_masks_pred'][bi, i] > 0)
                elif self.config.DATASET.name == 'hm3d_abo':
                    mask = torch.ones(h, w).bool().cuda()
                K_out = K.data.cpu().numpy()
                K_out[:2, :] /= 2
                if mask.sum() < 4:
                    out_pose = np.eye(4)
                    print('Blank Mask')
                else:
                    out_pose = opencv_ransacPnP(
                        scene_coord_pred[bi,
                                         i].cpu().numpy().transpose(1, 2, 0),
                        K_out,
                        mask.data.cpu().numpy())
                out_pose = torch.from_numpy(out_pose).float()

                R_pred = out_pose[:3, :3]
                t_pred = out_pose[:3, 3]
                R_gt = poses_gt[bi, i].inverse()[:3, :3]
                t_gt = poses_gt[bi, i].inverse()[:3, 3]
                t_err = torch.norm(t_pred - t_gt.cpu()).item()
                r_err = angular_distance_np(R_pred.data.cpu().numpy(),
                                            R_gt.data.cpu().numpy())[0]
                data['T_pred'][bi, i, :3, :3] = R_pred.cuda().t()
                data['T_pred'][bi, i, :3, 3] = -(R_pred.t() @ t_pred).cuda()
                err_R += r_err
                err_t += t_err
            err_R /= n
            err_t /= n
            metrics['pose_err_R'].append(err_R)
            metrics['pose_err_t'].append(err_t)
        return metrics

    def decode_object_mask(self, feature_map, data):
        b, n, c, h, w = feature_map.shape
        data['object_masks_pred'] = self.mask_head(
            feature_map.view(b * n, c, h, w)).view(b, n, h, w)

    def decode_scene_coord(self, feats, data):
        # feats: [b, c, h, w]
        b, n, c, h, w = feats.shape
        scene_coord = self.conv_scene_coord(feats.view(b * n, c, h, w))
        data['scene_coord_pred'] = scene_coord.view(b, n, -1, h, w)

    def forward(self, data, mode='train'):
        b, n, c, h, w = data['images'].shape

        ### Extract feature maps
        feats_c, feats_f = self.backbone(data['images'].reshape(-1, c, h, w))
        feats_c = feats_c.reshape(b, n, -1, feats_c.shape[-2],
                                  feats_c.shape[-1])
        feats_f = feats_f.reshape(b, n, -1, feats_f.shape[-2],
                                  feats_f.shape[-1])

        feats_c = self.cross_attention(feats_c)

        ### Decode pose
        self.make_prediction(feats_c, feats_f, data)

        ### Decode mask
        if self.config.TRAINER.TRAIN_MASK:
            self.decode_object_mask(feats_f, data)
        ### Decode Object Bounding Box
        if self.config.TRAINER.TRAIN_SCALE:
            data['bbox_scale_pred'] = self.conv_scale2(
                self.conv_scale1(feats_c).mean([-2, -1]).view(b, n,
                                                              -1).mean(1))
