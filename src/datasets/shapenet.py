from os import path as osp
import glob
import cv2
import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv
from utils import *
import os
from camera_utils import load_K_Rt_from_P, Lie, add_noise_to_pose, add_noise_to_world_to_cam_pose
import scipy


class ShapeNetDataset(utils.data.Dataset):
    def __init__(self, npz_name, args, config, mode='train', **kwargs):
        super().__init__()
        self.lie = Lie()
        self.debug = args.debug
        self.config = config
        self.root_dir = config.DATASET.DATA_ROOT
        self.root_dir2 = config.DATASET.DATA_ROOT.replace(
            'FvOR_ShapeNet', 'ShapeNet')
        self.mode = mode
        npz_path = os.path.join(config.DATASET.NPZ_ROOT, npz_name + '.npz')
        self.data_names = np.load(npz_path, allow_pickle=True)['data']
        self.modelID = npz_path.split('/')[-1].split('.')[0]
        if mode == 'test' and len(self.config.DATASET.SHAPENET_CATEGORY_TEST):
            self.data_names = list(
                filter(
                    lambda x: x['category'] in self.config.DATASET.
                    SHAPENET_CATEGORY_TEST, self.data_names))
        if mode == 'train':
            np.random.shuffle(self.data_names)
        elif mode == 'val':
            self.data_names = np.random.RandomState(666).permutation(
                self.data_names)[:500]
        elif mode == 'test':
            if config.TRAINER.LIMIT_TEST_NUMBER != -1:
                np.random.shuffle(self.data_names)
                self.data_names = self.data_names[:config.TRAINER.LIMIT_TEST_NUMBER]

        np.random.shuffle(self.data_names)
        self.img_wh = (config.DATASET.IMAGE_WIDTH, config.DATASET.IMAGE_HEIGHT)
        self.K = np.eye(4)
        self.K[:3, :3] = np.array([[149.84375, 0, 68.5], [0, 149.84375, 68.5],
                                   [0, 0, 1]])
        self.DEPTH_MAX = 100

    def __len__(self):
        if self.debug:
            return 1
        else:
            return len(self.data_names)

    def read_image(self, path, resize=(640, 480), augment_fn=None):
        image = cv2.imread(path)
        try:
            scales = (resize[0] / float(image.shape[1]),
                      resize[1] / float(image.shape[0]))
        except:
            print('path ', path)
        image = cv2.resize(image, resize) / 255.0
        h, w = image.shape[:2]
        idv, idu = np.meshgrid(np.linspace(0, 1, h),
                               np.linspace(0, 1, w),
                               indexing='ij')
        image = np.concatenate((image, idu[:, :, None], idv[:, :, None]), -1)
        # (h, w) -> (1, h, w) and normalized
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return image, scales

    def load_images(self, modelID, frames, data):
        n = len(frames)
        images = []
        category, model = modelID
        for i in range(n):
            stem_name = f"{frames[i]:03d}"
            img_name = osp.join(self.root_dir2, category, model,
                                'img_choy2016', f'{stem_name}.jpg')
            image, scales = self.read_image(img_name,
                                            resize=self.img_wh,
                                            augment_fn=None)
            images.append(image)
        data['scales'] = scales
        images = np.stack(images)
        data['images'] = images

    def read_depth(self, fname, resize=None):
        depth = cv2.imread(fname, cv2.IMREAD_UNCHANGED).astype('float32')
        depth[depth == np.inf] = 0
        depth[depth > self.DEPTH_MAX] = 0
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        if resize is not None:
            depth = cv2.resize(depth, resize, interpolation=cv2.INTER_NEAREST)
        return depth

    def load_depths(self, modelID, frames, data):
        n = len(frames)
        depths = []
        category, model = modelID

        for i in range(n):
            if os.path.exists(
                    osp.join(self.root_dir, category, model, 'depth',
                             f'00.exr')):
                stem_name = f"{frames[i]:02d}"
            else:
                stem_name = f"{frames[i]+1:02d}"
            depth = self.read_depth(osp.join(self.root_dir, category, model,
                                             'depth', f'{stem_name}.exr'),
                                    resize=self.img_wh)
            depths.append(depth)

        depths = np.stack(depths)
        data['depths'] = depths
        data['masks'] = (depths > 0)
        data['object_masks'] = data['masks'].astype('float32')

    def _read_abs_pose(self, modelID, frame):
        category, model = modelID
        pth = osp.join(self.root_dir, category, model, 'pose', f'{frame}.txt')
        return np.loadtxt(pth).astype('float32')  # w2c

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)

        return np.matmul(pose1, inv(pose0))  # (4, 4)

    def load_poses(self, modelID, frames, data):
        n = len(frames)
        poses = []
        category, model = modelID
        ret = np.load(
            f"{self.root_dir2}/{category}/{model}/img_choy2016/cameras.npz")
        for i in range(n):
            pose = np.eye(4)
            pose[:3, :] = ret['world_mat_%d' % frames[i]]
            poses.append(pose)
        poses = np.stack(poses)
        data['T_world_to_cam'] = poses.astype('float32')
        data['T_cam_to_world'] = self.inverse(data['T_world_to_cam'])
        data['poses'] = data['T_world_to_cam']
        del data['T_world_to_cam'], data['T_cam_to_world']

    def load_noisy_pose(self, data):
        if self.config.TRAINER.CAMERA_POSE == 'gt+noise':
            data['poses_noise@ShapeModule'], data[
                'poses_noise@ShapeModule_stat'] = add_noise_to_world_to_cam_pose(
                    self.lie,
                    data['poses'],
                    noise_std=self.config.TRAINER.NOISE_POSE_STD,
                    fix_first=True)
            data['poses_noise@PoseModule'], data[
                'poses_noise@PoseModule_stat'] = add_noise_to_world_to_cam_pose(
                    self.lie,
                    data['poses'],
                    noise_std=self.config.TRAINER.NOISE_POSE_STD * self.config.TRAINER.NOISE_POSE_MULTIPLIER,
                    fix_first=True)
        elif self.config.TRAINER.CAMERA_POSE == 'predicted':
            data['poses_noise@PoseModule'] = np.load(
                f"results_pose_init/{data['bag_names']}.npy",
                allow_pickle=True).item()['pred']
            data['poses_noise@ShapeModule'] = data['poses_noise@PoseModule']
            err_r = angular_distance_np(
                data['poses_noise@PoseModule'][:, :3, :3],
                data['poses'][:, :3, :3])
            err_t = np.linalg.norm(data['poses_noise@PoseModule'][:, :3, 3] -
                                   data['poses'][:, :3, 3],
                                   axis=-1)
            data['poses_noise@PoseModule_stat'] = {
                'noise_r': err_r,
                'noise_t': err_t
            }
        elif self.config.TRAINER.CAMERA_POSE == 'gt':
            data['poses_noise@ShapeModule'] = data['poses']
            data['poses_noise@PoseModule'] = data['poses']
            err_r = angular_distance_np(
                data['poses_noise@PoseModule'][:, :3, :3],
                data['poses'][:, :3, :3])
            err_t = np.linalg.norm(data['poses_noise@PoseModule'][:, :3, 3] -
                                   data['poses'][:, :3, 3],
                                   axis=-1)
            data['poses_noise@PoseModule_stat'] = {
                'noise_r': err_r,
                'noise_t': err_t
            }

    def compute_quaternion(self, poses_gt):
        n = len(poses_gt)
        Q = []
        for i in range(n):
            Q.append(rot2Quaternion_np(poses_gt[i][:3, :3]))
        Q = np.stack(Q)
        Q = np.concatenate((Q, poses_gt[:, :3, 3]), -1)
        return Q

    def load_point_cloud(self, modelID):
        category, model = modelID
        base_dir = f"{self.root_dir2}/{category}/{model}/"
        gt = np.load(f"{base_dir}/pointcloud.npz")
        points = (gt['points'] * gt['scale'] + gt['loc']).astype('float32')
        normals = gt['normals'].astype('float32')
        if self.mode == 'train':
            N = self.config.TRAINER.NUM_3D_POINTS
            ind2 = np.random.choice(len(points), N)
            points = points[ind2]
            normals = normals[ind2]
        x1, y1, z1 = np.max(points, axis=0)
        x0, y0, z0 = np.min(points, axis=0)
        BoundingBox = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0],
                                [x0, y1, z0], [x0, y0, z1], [x1, y0, z1],
                                [x1, y1, z1], [x0, y1, z1]])

        return points, normals, BoundingBox

    def load_sdf(self, modelID):
        category, model = modelID
        base_dir = f"{self.root_dir2}/{category}/{model}/"
        N = self.config.TRAINER.NUM_3D_POINTS
        if self.config.TRAINER.USE_OCC:
            gt = np.load(f"{base_dir}/points.npz")
            loc = gt['loc']
            xyz_all = gt['points'] * gt['scale'] + loc
            sdf_all = np.unpackbits(gt['occupancies'])
            ind = np.random.choice(len(xyz_all), N)
            xyz = xyz_all[ind]
            sdf = sdf_all[ind]
        else:
            gt = np.load(f"{base_dir}/points.npz")
            sdf = np.load(
                f"{base_dir.replace('ShapeNet', 'FvOR_ShapeNet')}/sdf.npz")
            xyz_all = gt['points'] * gt['scale'] + gt['loc']
            sdf_all = sdf['sdf'] * gt['scale']
            ind = np.random.choice(len(xyz_all), N)
            xyz = xyz_all[ind]
            sdf = sdf_all[ind]

        sdf_all = np.concatenate((xyz_all, sdf_all[:, None]),
                                 1).astype('float32')
        sdf = np.concatenate((xyz, sdf[:, None]), 1).astype('float32')
        return sdf_all, sdf

    def load_geometry(self, modelID, data):
        category, model = modelID

        data['surface_point'], data['surface_point_normal'], data[
            'BoundingBox'] = self.load_point_cloud(modelID)

        data['sdf_all'], data['sdf'] = self.load_sdf(modelID)

        data['bbox_scale'] = float(1.0)
        data['bbox_scale_gt'] = float(1.0)

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

    def load_intrinsics(self, modelID, frames, data):
        n = len(frames)
        scales = data['scales']
        intrinsics = []
        for i in range(n):
            K_cur = self.K.copy()
            K_cur[:2, :] *= np.array([scales[0], scales[1]])[:, None]
            intrinsics.append(K_cur)
        intrinsics = np.stack(intrinsics)
        data['intrinsics'] = intrinsics.astype('float32')
        data['intrinsics_inv'] = np.linalg.inv(intrinsics).astype('float32')

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

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        data = dict()
        data_name = self.data_names[idx]
        modelID = (data_name['category'], data_name['model'])
        frames = data_name['frame_ids'][:self.config.TRAINER.NUM_FRAME]
        data.update({
            'scene_id':
            modelID,
            'bag_id':
            idx,
            'bag_names':
            '%s_%s' % (modelID[0], modelID[1]) + '-' +
            '-'.join([f"{x:06d}" for x in frames]),
            'group':
            data_name['category'],
        })

        self.load_images(modelID, frames, data)

        self.load_intrinsics(modelID, frames, data)

        self.load_poses(modelID, frames, data)

        self.load_noisy_pose(data)

        self.load_geometry(modelID, data)

        if self.config.DATASET.LOAD_DEPTH:
            self.load_depths(modelID, frames, data)

            self.compute_scene_coords(data)

        return data
