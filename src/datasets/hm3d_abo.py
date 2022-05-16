from os import path as osp
import cv2
from typing import Dict
from unicodedata import name
import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv
from utils import *
import os
from camera_utils import Lie, add_noise_to_pose, add_noise_to_world_to_cam_pose
import scipy
import visdom
from scipy.spatial.transform import Rotation as R


class HM3DABODataset(utils.data.Dataset):
    def __init__(self, npz_name, args, config, mode='train', **kwargs):
        super().__init__()

        self.lie = Lie()
        self.debug = args.debug
        self.config = config
        self.center_crop = args.center_crop
        self.root_dir = config.DATASET.DATA_ROOT
        self.mode = mode
        npz_path = os.path.join(
            config.DATASET.NPZ_ROOT,
            npz_name + '.npy')
        self.data_names = np.load(npz_path, allow_pickle=True)
        self.modelID = npz_path.split('/')[-1].split('.')[0]
        if mode == 'train':
            np.random.shuffle(self.data_names)
        elif mode == 'test':
            if config.TRAINER.LIMIT_TEST_NUMBER != -1:
                np.random.shuffle(self.data_names)
                self.data_names = self.data_names[:config.TRAINER.LIMIT_TEST_NUMBER]
        self.img_wh = (config.DATASET.IMAGE_WIDTH, config.DATASET.IMAGE_HEIGHT)

    def __len__(self):
        if self.debug:
            return 1
        else:
            return len(self.data_names)

    def read_image(self, path, resize=(640, 480), augment_fn=None):
        """
        Args:
            resize (tuple): align image to depthmap, in (w, h).
            augment_fn (callable, optional): augments images with pre-defined visual effects
        Returns:
            image (torch.tensor): (1, h, w)
            mask (torch.tensor): (h, w)
            scale (torch.tensor): [w/w_new, h/h_new]        
        """
        # read and resize image
        try:
            image = cv2.imread(path)
            scales = (resize[0] / float(image.shape[1]),
                      resize[1] / float(image.shape[0]))
        except:
            print(path)
        if self.center_crop:
            image = self.center_crop_helper(image)
        image = cv2.resize(image, resize) / 255.0
        h, w = image.shape[:2]
        idv, idu = np.meshgrid(np.linspace(0, 1, h),
                               np.linspace(0, 1, w),
                               indexing='ij')
        image = np.concatenate((image, idu[:, :, None], idv[:, :, None]), -1)

        # (h, w) -> (1, h, w) and normalized
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return image, scales

    def _read_abs_pose(self, scene_name, frame):
        pth = osp.join(self.root_dir, 'scenes', scene_name, 'pose', f'{frame}.txt')
        return np.loadtxt(pth).astype('float32')  # w2c

    def read_depth(self, fname, resize=None):
        depth = cv2.imread(fname, cv2.IMREAD_UNCHANGED).astype('float32')
        depth[depth == np.inf] = 0
        depth[depth > 100] = 0
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        if resize is not None:
            depth = cv2.resize(depth, resize, interpolation=cv2.INTER_NEAREST)
        return depth

    def parse_object_id(self, modelID):
        return modelID.split('_')[1]

    def load_point_cloud(self, modelID):
        object_id = self.parse_object_id(modelID)
        fn = f"{self.root_dir}/abo_assets/{object_id}.point.npz"
        gt = np.load(fn)
        points = (gt['points'] * gt['scale'] +
                                 gt['loc']).astype('float32')
        normals = gt['normals'].astype('float32')
        if self.mode == 'train':
            N = self.config.TRAINER.NUM_3D_POINTS
            ind = np.random.choice(points.shape[0], N)
            points = points[ind]
            normals = normals[ind]

        ### Compute the 3D Bounding Box
        x1, y1, z1 = np.max(points, axis=0)
        x0, y0, z0 = np.min(points, axis=0)
        BoundingBox = np.array([[x0, y0, z0], [x1, y0,
                                                       z0], [x1, y1, z0],
                                        [x0, y1, z0], [x0, y0,
                                                       z1], [x1, y0, z1],
                                        [x1, y1, z1], [x0, y1, z1]])
        return points, normals, BoundingBox

    def load_sdf(self, modelID):
        N = self.config.TRAINER.NUM_3D_POINTS
        object_id = self.parse_object_id(modelID)
        fn = f"{self.root_dir}/abo_assets/{object_id}.sdf.npz"
        gt = np.load(fn)
        xyz_all = gt['points'] * gt['scale'] + gt['loc']
        sdf_all = gt['sdf'] * gt['scale']
        ind = np.random.choice(len(xyz_all), N)
        xyz = xyz_all[ind]
        sdf = sdf_all[ind]
        sdf_all = np.concatenate((xyz_all, sdf_all[:,None]), 1).astype('float32')
        sdf = np.concatenate((xyz, sdf[:,None]), 1).astype('float32')
        return sdf_all, sdf


    def load_geometry(self, modelID, data):
        objectID = modelID.split('_')[1]
        N = self.config.TRAINER.NUM_3D_POINTS

        data['surface_point'], data['surface_point_normal'], data['BoundingBox'] = self.load_point_cloud(modelID)

        data['sdf_all'], data['sdf'] = self.load_sdf(modelID)

        corner_min = data['surface_point'].min(0)
        corner_max = data['surface_point'].max(0)
        data['bbox_corners'] = np.stack((corner_min, corner_max))
        data['bbox_scale_gt'] = float(np.max(corner_max - corner_min))
        if self.config.TRAINER.BBOX == 'gt':
            data['bbox_scale'] = float(np.max(corner_max - corner_min)) * 1.1
        elif self.config.TRAINER.BBOX == 'gt+perturb':
            scaling = 1 + (np.random.randn(1) * 0.3).clip(-0.25, 0.25)
            scaling = scaling * 1.5
            data['bbox_scale'] = float(
                np.max(corner_max - corner_min) * scaling)
        elif self.config.TRAINER.BBOX == 'fixed':
            data['bbox_scale'] = 2.5
        elif self.config.TRAINER.BBOX == 'predicted':
            data['bbox_scale'] = np.load(
                f"results_pose_init/{data['bag_names']}.npy",
                allow_pickle=True).item()['scale_pred']
            data['bbox_scale'] = data['bbox_scale'] * 1.5
        data['sdf'][:,3] = data['sdf'][:,3] / data['bbox_scale']
        data['sdf_all'][:,3] = data['sdf_all'][:,3] / data['bbox_scale']

    def load_masks(self, modelID, frames, data):
        n = len(frames)
        masks = []
        for i in range(n):
            stem_name = f"{frames[i]:06d}"
            img_name = osp.join(self.root_dir, 'scenes', modelID, 'mask',
                                f'{stem_name}.png')
            mask = (cv2.resize(cv2.imread(img_name),
                               self.img_wh,
                               interpolation=cv2.INTER_NEAREST)[:, :, 0] >
                    0).astype('float32')
            masks.append(mask)
        masks = np.stack(masks)
        data['object_masks'] = masks

    def load_images(self, modelID, frames, data):
        n = len(frames)
        images = []
        for i in range(n):
            stem_name = f"{frames[i]:06d}"
            img_name = osp.join(self.root_dir, 'scenes', modelID, 'rgb',
                                f'{stem_name}.jpg')
            image, scales = self.read_image(img_name,
                                           resize=self.img_wh,
                                           augment_fn=None)
            images.append(image)
        data['scales'] = scales
        images = np.stack(images)
        data['images'] = images

    def inverse(self, pose):
        R = pose[..., :3, :3]
        t = pose[..., :3, 3:4]
        R_new = np.swapaxes(R, -2, -1)
        t_new = -R_new @ t
        pose_new = np.zeros_like(pose)
        pose_new[..., :3, :3] = R_new
        pose_new[..., :3, 3:4] = t_new
        pose_new[..., 3, 3] = 1
        return pose_new

    def load_object_pose(self, modelID, data):
        """Object to world and world to object pose
        """
        data['T_obj_to_world'] = np.loadtxt(f"{self.root_dir}/scenes/{modelID}/obj_pose.txt").astype('float32')
        data['T_world_to_obj'] = self.inverse(data['T_obj_to_world'])

    def load_camera_pose(self, modelID, frames, data):
        """Camera to world and world to Camera pose
        """
        n = len(frames)
        poses = []
        for i in range(n):
            stem_name = f"{frames[i]:06d}"
            pose = self._read_abs_pose(modelID, stem_name)  # w2c
            poses.append(pose)
        poses = np.stack(poses)
        data['T_world_to_cam'] = poses
        data['T_cam_to_world'] = self.inverse(poses)

    def load_poses(self, modelID, frames, data):
        """Load Camera to Estimated Object coordinate system pose and vice versa.
        """
        self.load_camera_pose(modelID, frames, data)

        self.load_object_pose(modelID, data)

        data['poses'] = data['T_world_to_cam'] @ data[
            'T_obj_to_world']
        data['poses_inv'] = self.inverse(data['poses'])

        del data['T_world_to_cam'], data['T_cam_to_world'], data[
            'T_obj_to_world'], data['T_world_to_obj']

    def load_depths(self, modelID, frames, data):
        n = len(frames)
        depths = []
        for i in range(n):
            stem_name = f"{frames[i]:06d}"
            depth = self.read_depth(osp.join(self.root_dir, 'scenes', modelID, 'depth',
                                             f'{stem_name}.exr'),
                                    resize=self.img_wh)
            depths.append(depth)
        depths = np.stack(depths)

        depth_range_mask = (depths > 0) & (depths < 5.0)
        masks = depth_range_mask
        depths = depths * masks
        data['masks'] = masks
        data['depths'] = depths

    def load_intrinsics(self, modelID, frames, data):
        n = len(frames)
        scales = data['scales']
        K = np.eye(4)
        K[:3, :3] = np.loadtxt(
            f"{self.root_dir}/scenes/{modelID}/intrinsic.txt").astype('float32')
        intrinsics = []
        for i in range(n):
            K_cur = K.copy()
            K_cur[:2, :] *= np.array([scales[0], scales[1]])[:, None]
            intrinsics.append(K_cur)
        intrinsics = np.stack(intrinsics)
        data['intrinsics'] = intrinsics.astype('float32')
        data['intrinsics_inv'] = np.linalg.inv(intrinsics).astype('float32')

    def compute_scene_coords(self, data):
        intrinsics = []
        depths = data['depths']
        n = len(depths)

        poses = data['poses']

        scene_coords = []
        scene_coords_half = []
        local_scene_coords_half = []
        for i in range(n):
            depth = depths[i]
            pose = poses[i]
            mask = depth > 0
            pose = np.linalg.inv(pose)
            ## generate scene coordinate GT
            inv_K = data['intrinsics_inv'][i][:3, :3]
            local_scene_coord = backproject_depth(depth, inv_K, mask=False)
            scene_coord = transform4x4(local_scene_coord,
                                       pose).reshape(depth.shape[0],
                                                     depth.shape[1],
                                                     3).astype('float32')
            local_scene_coord = local_scene_coord.reshape(
                depth.shape[0], depth.shape[1], 3).astype('float32')
            scene_coord[~mask] = 0
            scene_coords_half.append(
                cv2.resize(scene_coord,
                           (self.img_wh[0] // 2, self.img_wh[1] // 2),
                           interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1))
            local_scene_coords_half.append(
                cv2.resize(local_scene_coord,
                           (self.img_wh[0] // 2, self.img_wh[1] // 2),
                           interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1))
            scene_coords.append(scene_coord.transpose(2, 0, 1))
        scene_coords = np.stack(scene_coords)
        scene_coords_half = np.stack(scene_coords_half)
        local_scene_coords_half = np.stack(local_scene_coords_half)
        masks_half = ((scene_coords_half)**2).sum(1) > 0
        data.update({
            'masks_half': masks_half,
            'scene_coords': scene_coords,  # (h, w)
            'scene_coords_half': scene_coords_half,  # (h, w)
            'local_scene_coords_half': local_scene_coords_half,  # (h, w)
        })

    def load_noisy_pose(self, data):
        if self.config.TRAINER.CAMERA_POSE == 'gt+noise':
            data['poses_noise@ShapeModule'], data['poses_noise@ShapeModule_stat'] = add_noise_to_world_to_cam_pose(
                self.lie,
                data['poses'],
                noise_std=self.config.TRAINER.NOISE_POSE_STD,
                fix_first=False)
            data['poses_noise@PoseModule'], data['poses_noise@PoseModule_stat'] = add_noise_to_world_to_cam_pose(
                self.lie,
                data['poses'],
                noise_std=self.config.TRAINER.NOISE_POSE_STD * self.config.TRAINER.NOISE_POSE_MULTIPLIER ,
                fix_first=False)
        elif self.config.TRAINER.CAMERA_POSE == 'predicted':
            data['poses_noise@PoseModule'] = np.load(
                f"results_pose_init/{data['bag_names']}.npy",
                allow_pickle=True).item()['pred']
            data['poses_noise@ShapeModule'] = data['poses_noise@PoseModule']
            err_r = angular_distance_np(data['poses_noise@PoseModule'][:,:3,:3],data['poses'][:,:3,:3])
            err_t = np.linalg.norm(data['poses_noise@PoseModule'][:,:3,3]-data['poses'][:,:3,3], axis=-1)
            data['poses_noise@PoseModule_stat']= {'noise_r':err_r, 'noise_t': err_t}
        elif self.config.TRAINER.CAMERA_POSE == 'gt':
            data['poses_noise@ShapeModule'] = data['poses']
            data['poses_noise@PoseModule'] = data['poses']
            err_r = angular_distance_np(data['poses_noise@PoseModule'][:,:3,:3],data['poses'][:,:3,:3])
            err_t = np.linalg.norm(data['poses_noise@PoseModule'][:,:3,3]-data['poses'][:,:3,3], axis=-1)
            data['poses_noise@PoseModule_stat']= {'noise_r':err_r, 'noise_t': err_t}

    def compute_quaternion(self, poses_gt):
        n = len(poses_gt)
        Q = []
        for i in range(n):
            Q.append(rot2Quaternion_np(poses_gt[i][:3,:3]))
        Q = np.stack(Q)
        return np.concatenate((Q, poses_gt[:,:3,3]), -1)

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        data = dict()
        data_name = self.data_names[idx]
        if self.mode == 'train':
            modelID = self.modelID
        else:
            modelID = data_name['modelID']
        frames = data_name['frames'][:self.config.TRAINER.NUM_FRAME]

        data.update(dict(scene_id=modelID,bag_id=idx,bag_names=modelID + '-' + '-'.join([f"{x:06d}" for x in frames]),group='HM3D_ABO')
            #'group':group
        )

        self.load_images(modelID, frames, data)

        self.load_intrinsics(modelID, frames, data)

        self.load_poses(modelID, frames, data)

        self.load_geometry(modelID, data)

        self.load_masks(modelID, frames, data)

        self.load_noisy_pose(data)

        if self.config.DATASET.LOAD_DEPTH:
            self.load_depths(modelID, frames, data)
            self.compute_scene_coords(data)

        return data
