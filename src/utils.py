from __future__ import absolute_import, division, print_function
from collections import defaultdict
import os
import random
import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
from torch.autograd import Variable
import time
import torch.nn.functional as F
import re
from torch._six import container_abcs, string_classes, int_classes
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess
import open3d as o3d
import skimage.measure
from pykdtree.kdtree import KDTree
import trimesh
from sklearn.metrics import f1_score
from collections import deque
import collections
from yacs.config import CfgNode
import logging
import shlex
import subprocess
from lib.libmesh import check_mesh_contains
from lib import libmcubes
from PIL import Image, ImageFont, ImageDraw
from collections import OrderedDict
import fnmatch
import matplotlib.pyplot as plt
import contextlib
import joblib
from typing import Union
from loguru import _Logger, logger
from itertools import chain
from yacs.config import CfgNode as CN
from pytorch_lightning.utilities import rank_zero_only

MARKERS = ["o", "X", "D", "^", "<", "v", ">"]



def update_dict_recursive(cfg, k, v):
    if '.' in k:
        head, tail = k.split('.', 1)
        if head in cfg:
            update_dict_recursive(cfg[head], tail, v)
        else:
            return
    else:
        cfg[k] = v



def update_cfg_with_args(cfg, args):
    if hasattr(args, 'noise_std'):
        if args.noise_std is not None:
            cfg.TRAINER.NOISE_POSE_STD = args.noise_std
    if hasattr(args, 'rigid_align_to_gt'):
        if args.rigid_align_to_gt:
            cfg.TRAINER.RIGID_ALIGN_TO_GT = True

    if hasattr(args, 'limit_test_number'):
        if args.limit_test_number is not None:
            cfg.TRAINER.LIMIT_TEST_NUMBER = args.limit_test_number
    if hasattr(args, 'lambda_reg'):
        if args.lambda_reg is not None:
            cfg.POSE_REFINE.LAMBDA_REG = args.lambda_reg
    if hasattr(args, 'lambda_damping'):
        if args.lambda_damping is not None:
            cfg.POSE_REFINE.LAMBDA_DAMPING = args.lambda_damping
    if hasattr(args, 'joint_step'):
        if args.joint_step is not None:
            cfg.POSE_REFINE.JOINT_STEP = args.joint_step
    if hasattr(args, 'flownet_normalize_feature_map'):
        if args.flownet_normalize_feature_map:
            cfg.POSE_REFINE.FLOWNET_NORMALIZE_FEATURE_MAP = args.flownet_normalize_feature_map
    if hasattr(args, 'eval_per_joint_step'):
        if args.eval_per_joint_step:
            cfg.TRAINER.EVAL_PER_JOINT_STEP = True
    if hasattr(args, 'use_predicted_pose') and hasattr(
            args, 'use_gt_pose') and hasattr(args, 'use_noisy_pose'):
        assert (np.sum([
            args.use_predicted_pose, args.use_gt_pose, args.use_noisy_pose
        ]) <= 1)
        if args.use_gt_pose:
            cfg.TRAINER.CAMERA_POSE = 'gt'
        elif args.use_noisy_pose:
            cfg.TRAINER.CAMERA_POSE = 'gt+noise'
        elif args.use_predicted_pose:
            cfg.TRAINER.CAMERA_POSE = 'predicted'
    if hasattr(args, 'limit_test_number'):
        if args.limit_test_number is not None:
            cfg.TRAINER.LIMIT_TEST_NUMBER = args.limit_test_number
    if hasattr(args, 'train_scale'):
        if args.train_scale is not None:
            cfg.TRAINER.TRAIN_SCALE = args.train_scale



def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 10000
    split = []
    for i, indx in enumerate(
            torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['ray_dirs'] = torch.index_select(model_input['ray_dirs'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1,
                                                 indx)
        split.append(data)
    return split


def launch_tensorboard_process(logdir):
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logger = logging.getLogger('run.tensorboard_util')
    logger.info("Launching TensorBoard process")
    command = "tensorboard --logdir %s --port %s" % \
              (logdir, os.environ.get("TENSORBOARD_PORT"))
    subprocess.Popen(shlex.split(command),
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     env=os.environ.copy())


def deep_update(source, overrides):
    """
    https://github.com/rbgirshick/yacs/issues/29
    """
    for key, value in overrides.items():
        print(key)
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            returned = CfgNode(returned)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


class AverageMeter(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20000):
        self.values = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.values.append(value)
        self.counts.append(count)
        self.sum += value
        self.count += count

    @property
    def avg(self):
        if np.sum(self.counts) == 0:
            return 0
        return np.sum(self.values) / np.sum(self.counts)

    @property
    def median(self):
        return np.median(self.values)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        return self.sum / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            count = 1
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    count = v.numel()
                    v = v.sum().item()
            elif isinstance(v, np.float32) or isinstance(v, np.float64):
                v = float(v)
            assert isinstance(v, (float, int))
            self.meters[k].update(v, count)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append("{}: {:.4f} ({:.4f})".format(
                name, meter.avg, meter.global_avg))
        return self.delimiter.join(metric_str)

    @property
    def summary_str(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append("{}: {:.4f}".format(name, meter.global_avg))
        return self.delimiter.join(metric_str)


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:
    Returns:
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        'scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(backend=backend,
                            init_method='tcp://127.0.0.1:%d' % tcp_port,
                            rank=local_rank,
                            world_size=num_gpus)
    rank = dist.get_rank()
    return num_gpus, rank


def v(var, cuda=True, volatile=False):
    if type(var) == torch.Tensor or type(var) == torch.DoubleTensor:
        res = Variable(var.float(), volatile=volatile)
    elif type(var) == np.ndarray:
        res = Variable(torch.from_numpy(var).float(), volatile=volatile)
    if cuda:
        res = res.cuda()
    return res


def npy(var):
    return var.data.cpu().numpy()


def cuda_time():
    torch.cuda.synchronize()
    return time.time()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parameters_count(net, name):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('total parameters for %s: %.3f M' % (name, params / 1e6))


def write_ply(fn, point, normal=None, color=None):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(point)
    if color is not None:
        ply.colors = o3d.utility.Vector3dVector(color)
    if normal is not None:
        ply.normals = o3d.utility.Vector3dVector(normal)
    o3d.io.write_point_cloud(fn, ply)


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def get_unit_grid(batch_size, dim, half_size=0.5):
    xs = np.linspace(-half_size, half_size, dim)
    #grid = np.stack(np.meshgrid(xs, xs, xs, indexing='ij'), -1).reshape(-1, 3)
    zs, ys, xs = np.meshgrid(xs, xs, xs, indexing='ij')
    grid = np.stack((xs, ys, zs), -1).reshape(-1, 3)
    grid = torch.from_numpy(grid).float().unsqueeze(0).view(
        1, dim, dim, dim, -1).repeat(batch_size, 1, 1, 1, 1).contiguous()
    return grid


def convert_sdf_samples_to_trianglemesh_onet(occ_hat, threshold, corners):
    n_x, n_y, n_z = occ_hat.shape
    box_size = corners[1] - corners[0]
    # Make sure that mesh is watertight
    t0 = time.time()
    occ_hat_padded = np.pad(occ_hat, 1, 'constant', constant_values=1e6)
    vertices, triangles = libmcubes.marching_cubes(occ_hat_padded, threshold)
    # Strange behaviour in libmcubes: vertices are shifted by 0.5
    vertices -= 0.5
    # Undo padding
    vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
    vertices = box_size * (vertices - 0.5)
    return vertices, triangles


def convert_sdf_samples_to_trianglemesh(
    sdf,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
    threshold=0.0,
    need_normal=False,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        sdf,
        level=threshold,
        spacing=[voxel_size[0], voxel_size[1], voxel_size[2]])

    if need_normal:
        d, h, w = sdf.shape[:3]
        x = verts[:, 0] * (d - 1)
        y = verts[:, 1] * (h - 1)
        z = verts[:, 2] * (w - 1)
        x = x.clip(0, d - 2)
        y = y.clip(0, h - 2)
        z = z.clip(0, w - 2)
        x0 = np.floor(x).astype('int')
        y0 = np.floor(y).astype('int')
        z0 = np.floor(z).astype('int')
        dx = x - x0
        dy = y - y0
        dz = z - z0
        col_idx = np.zeros([len(x), 8])
        normal = np.zeros([len(x), 3])
        col_idx[:, 0] = (x0) * h * w + (y0) * w + (z0)
        col_idx[:, 1] = (x0 + 1) * h * w + (y0) * w + (z0)
        col_idx[:, 2] = (x0 + 1) * h * w + (y0) * w + (z0 + 1)
        col_idx[:, 3] = (x0) * h * w + (y0) * w + (z0 + 1)
        col_idx[:, 4] = (x0) * h * w + (y0 + 1) * w + (z0 + 1)
        col_idx[:, 5] = (x0) * h * w + (y0 + 1) * w + (z0)
        col_idx[:, 6] = (x0 + 1) * h * w + (y0 + 1) * w + (z0)
        col_idx[:, 7] = (x0 + 1) * h * w + (y0 + 1) * w + (z0 + 1)

        sdf = sdf.flatten()
        col_idx = col_idx.astype('int')
        v0 = sdf[col_idx[:, 0]]
        v1 = sdf[col_idx[:, 1]]
        v2 = sdf[col_idx[:, 2]]
        v3 = sdf[col_idx[:, 3]]
        v4 = sdf[col_idx[:, 4]]
        v5 = sdf[col_idx[:, 5]]
        v6 = sdf[col_idx[:, 6]]
        v7 = sdf[col_idx[:, 7]]
        normal[:,0] = ((v1 - v0) * (1 - dy) * (1 - dz) + (v2 - v3) * (1 - dy) * dz +\
                   (v6 - v5) * dy * (1 - dz) + (v7 - v4) * dz * dy) / voxel_size
        normal[:,1] = ((v5 - v0) * (1 - dx) * (1 - dz) + (v6 - v1) * dx * (1 - dz) +\
                   (v4 - v3) * (1 - dx) * dz + (v7 - v2) * dx * dz) / voxel_size
        normal[:,2] = ((v3 - v0) * (1 - dx) * (1 - dy) + (v2 - v1) * dx * (1 - dy) +\
                   (v7 - v6) * dx * dy + (v4 - v5) * (1 - dx) * dy) / voxel_size

        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file
    if need_normal:
        return mesh_points, faces, normal
    else:
        return mesh_points, faces


def f_score(predicted, target):
    score = f1_score(target,
                     predicted,
                     pos_label=1,
                     average='binary',
                     sample_weight=None)
    return score


def eval_pointcloud(pointcloud,
                    pointcloud_tgt,
                    normals=None,
                    normals_tgt=None):
    ''' Evaluates a point cloud.
    Args:
        pointcloud (numpy array): predicted point cloud
        pointcloud_tgt (numpy array): target point cloud
        normals (numpy array): predicted normals
        normals_tgt (numpy array): target normals
    '''
    # Return maximum losses if pointcloud is empty
    if pointcloud.shape[0] == 0:
        logger.warn('Empty pointcloud / mesh detected!')
        out_dict = EMPTY_PCL_DICT.copy()
        if normals is not None and normals_tgt is not None:
            out_dict.update(EMPTY_PCL_DICT_NORMALS)
        return out_dict

    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, completeness_normals = distance_p2p(pointcloud_tgt,
                                                      normals_tgt, pointcloud,
                                                      normals)
    completeness2 = completeness**2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(pointcloud, normals,
                                              pointcloud_tgt, normals_tgt)
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (0.5 * completeness_normals + 0.5 * accuracy_normals)
    chamferL1 = 0.5 * (completeness + accuracy)

    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals': normals_correctness,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer-L2': chamferL2,
        'chamfer-L1': chamferL1,
    }

    return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0],
                                       dtype=np.float32)
    return dist, normals_dot_product


def get_local_split(items: list, world_size: int, rank: int, seed: int):
    """ The local rank only loads a split of the dataset. """
    n_items = len(items)
    items_permute = np.random.RandomState(seed).permutation(items)
    if n_items % world_size == 0:
        padded_items = items_permute
    else:
        padding = np.random.RandomState(seed).choice(items,
                                                     world_size -
                                                     (n_items % world_size),
                                                     replace=True)
        padded_items = np.concatenate([items_permute, padding])
        assert len(padded_items) % world_size == 0, \
            f'len(padded_items): {len(padded_items)}; world_size: {world_size}; len(padding): {len(padding)}'
    n_per_rank = len(padded_items) // world_size
    local_items = padded_items[n_per_rank * rank:n_per_rank * (rank + 1)]
    return local_items


def o3d_mesh(mesh_v, mesh_f, mesh_c=None):
    trimesh = o3d.geometry.TriangleMesh()
    trimesh.vertices = o3d.utility.Vector3dVector(mesh_v)
    trimesh.triangles = o3d.utility.Vector3iVector(mesh_f)
    if mesh_c is not None:
        trimesh.vertex_colors = o3d.utility.Vector3dVector(mesh_c)
    return trimesh


def o3d_mesh_to_trimesh(mesh):
    out_mesh = trimesh.Trimesh(vertices=np.array(mesh.vertices),
                               faces=np.array(mesh.triangles))
    return out_mesh


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def calculate_fscore(pred_point, gt_point, th=0.01):
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    gen_points_kd_tree = KDTree(pred_point)

    gt_points_kd_tree = KDTree(gt_point)

    threshold_list = [
        CUBE_SIDE_LEN / 200, CUBE_SIDE_LEN / 100, CUBE_SIDE_LEN / 50,
        CUBE_SIDE_LEN / 20, CUBE_SIDE_LEN / 10, CUBE_SIDE_LEN / 5
    ]

    ret = {}
    for th in threshold_list:
        import ipdb
        ipdb.set_trace()
        d1, one_vertex_ids = gen_points_kd_tree.query(gt_point)
        d2, one_vertex_ids = gt_points_kd_tree.query(pred_point)

        if len(d1) and len(d2):
            recall = float(sum(d < th for d in d2)) / float(len(d2))
            precision = float(sum(d < th for d in d1)) / float(len(d1))

            if recall + precision > 0:
                fscore = 2 * recall * precision / (recall + precision)
            else:
                fscore = 0
        else:
            fscore = 0
            precision = 0
            recall = 0

        ret['fscore_%.3f' % th] = fscore
        ret['recall_%.3f' % th] = recall
        ret['precision_%.3f' % th] = precision

    return ret


def compute_chamfer(gt_points, pred_point, num_mesh_samples=100000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """
    # one direction
    gen_points_kd_tree = KDTree(pred_point)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(one_distances)
    gt_to_gen_chamfer2 = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_point)
    gen_to_gt_chamfer = np.mean(two_distances)
    gen_to_gt_chamfer2 = np.mean(np.square(two_distances))

    chamferL1 = 0.5 * (gt_to_gen_chamfer + gen_to_gt_chamfer)
    chamferL2 = 0.5 * (gt_to_gen_chamfer2 + gen_to_gt_chamfer2)
    return chamferL1, chamferL2


def compute_trimesh_chamfer(gt_points, gen_mesh, num_mesh_samples=100000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = np.array(
        gen_mesh.sample_points_uniformly(num_mesh_samples).points).astype(
            'float32')

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(one_distances)
    gt_to_gen_chamfer2 = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(two_distances)
    gen_to_gt_chamfer2 = np.mean(np.square(two_distances))

    chamferL1 = 0.5 * (gt_to_gen_chamfer + gen_to_gt_chamfer)
    chamferL2 = 0.5 * (gt_to_gen_chamfer2 + gen_to_gt_chamfer2)
    return chamferL1, chamferL2


def backproject_depth_th(depth, inv_K, mask=False, device='cuda'):
    h, w = depth.shape
    idu, idv = np.meshgrid(range(w), range(h))
    grid = np.stack((idu.flatten(), idv.flatten(), np.ones([w * h])))
    grid = torch.from_numpy(grid).float().to(device)
    x = torch.matmul(inv_K[:3, :3], grid)
    x = x * depth.flatten()[None, :]
    x = x.t()
    if mask:
        x = x[depth.flatten() > 0]
    return x


def backproject_depth(depth, inv_K, mask=False):
    h, w = depth.shape
    idu, idv = np.meshgrid(range(w), range(h))
    grid = np.stack((idu.flatten(), idv.flatten(), np.ones([w * h])))
    x = np.matmul(inv_K[:3, :3], grid)
    x = x * depth.flatten()[None, :]
    x = x.T
    if mask:
        x = x[depth.flatten() > 0]
    return x


def transform4x4(pc, T):
    # T: [4,4]
    # pc: [n, 3]
    # return: [n, 3]
    return (np.matmul(T[:3, :3], pc.T) + T[:3, 3:4]).T


def transform3x3(pc, T):
    # T: [4,4]
    # pc: [n, 3]
    # return: [n, 3]
    return (np.matmul(T[:3, :3], pc.T)).T


def transform4x4_th(pc, T):
    # T: [4,4]
    # pc: [n, 3]
    # return: [n, 3]
    return (torch.matmul(T[:3, :3], pc.t()) + T[:3, 3:4]).t()


def global_align(pose_pred, pose_gt):
    #W = (pose_pred @ torch.inverse(pose_gt)).mean(dim=1)
    W = (pose_gt @ torch.inverse(pose_pred)).mean(dim=1)
    U, S, V = torch.svd(W[:, :3, :3])
    R = U @ V.transpose(-1, -2)
    #t = W[:,:3,3:4]
    T = torch.zeros(pose_pred.shape[0], 4, 4).cuda()
    t = -(R @ pose_pred[:, :, :3, 3:4] - pose_gt[:, :, :3, 3:4]).mean(dim=1)
    T[:, :3, :3] = R
    T[:, :3, 3:4] = t
    T[:, 3, 3] = 1
    return T


def pcloud_line(prev, cur, color=None):
    alpha = np.linspace(0, 1, 100)
    pcloud = prev[None, :] + alpha[:, None] * (cur - prev)[None, :]
    if color is None:
        pcolor = np.tile(np.array([0, 1, 0])[None, :], [pcloud.shape[0], 1])
    else:
        assert (len(color) == 3)
        pcolor = np.tile(np.array(color)[None, :], [pcloud.shape[0], 1])
    return pcloud, pcolor


def pcloud_point(x, color=None, normal=None, eps=1e-3, style='point'):

    if style == 'cross':

        pcloud = []
        pcolor = []
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
        pcloud.append(
            np.linspace(-0.2, 0.2, 50)[:, None] * v[None, :] + x.reshape(1, 3))
        v2 = np.random.randn(3)
        v2 -= np.dot(v2, v) * v2
        v2 /= np.linalg.norm(v2)
        pcloud.append(
            np.linspace(-0.2, 0.2, 50)[:, None] * v2[None, :] +
            x.reshape(1, 3))
        pcloud = np.concatenate(pcloud)
    elif style == 'point':
        pcloud = np.tile(x[None, :], [100, 1])
        pcloud += (np.random.rand(*pcloud.shape) - 0.5) * eps
    if color is None:
        pcolor = np.tile(np.array([0, 1, 0])[None, :], [pcloud.shape[0], 1])
    else:
        assert (len(color) == 3)
        pcolor = np.tile(np.array(color)[None, :], [pcloud.shape[0], 1])
    if normal is None:
        pnormal = np.tile(np.array([0, 1, 0])[None, :], [pcloud.shape[0], 1])
    else:
        assert (len(normal) == 3)
        pnormal = np.tile(np.array(normal)[None, :], [pcloud.shape[0], 1])
    return pcloud, pcolor, pnormal


def pcloud_arrow(pos, ray, length=0.1, color=None):
    xyz, xyz_c = pcloud_line(pos, pos + ray * length, color)
    return xyz, xyz_c


def visCamera(poses_cam2world, color=None, fn='test.ply'):
    # poses: [n, 4, 4]
    xyz, xyz_c = [], []
    for i in range(len(poses_cam2world)):
        pos = poses_cam2world[i][:3, 3]
        lookat = poses_cam2world[i][:3, 2]
        tp1, tp2 = pcloud_arrow(pos, lookat, length=0.1, color=color)
        xyz.append(tp1)
        xyz_c.append(tp2)
        tp1, tp2, _ = pcloud_point(pos,
                                   color=np.array([1, 1, 0]),
                                   normal=None,
                                   eps=1e-2,
                                   style='point')
        xyz.append(tp1)
        xyz_c.append(tp2)
    xyz = np.concatenate(xyz)
    xyz_c = np.concatenate(xyz_c)
    if fn is not None:
        write_ply(fn, xyz, color=xyz_c)
    return xyz, xyz_c


def randomRotationbk(deflection=0.1, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3, ))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def randomRotation(epsilon):
    axis = (np.random.rand(3) - 0.5)
    axis /= np.linalg.norm(axis)
    dtheta = np.random.randn(1) * np.pi * epsilon
    K = np.array(
        [0, -axis[2], axis[1], axis[2], 0, -axis[0], -axis[1], axis[0],
         0]).reshape(3, 3)
    dR = np.eye(3) + np.sin(dtheta) * K + (1 - np.cos(dtheta)) * np.matmul(
        K, K)
    return dR


def randomRotationv2(max_angle=5.0):
    axis = (np.random.rand(3) - 0.5)
    axis /= np.linalg.norm(axis)
    dtheta = np.random.uniform(-1, 1) * np.deg2rad(max_angle)
    K = np.array(
        [0, -axis[2], axis[1], axis[2], 0, -axis[0], -axis[1], axis[0],
         0]).reshape(3, 3)
    dR = np.eye(3) + np.sin(dtheta) * K + (1 - np.cos(dtheta)) * np.matmul(
        K, K)
    return dR


def angular_distance_np(R_hat, R):
    # measure the angular distance between two rotation matrice
    # R1,R2: [n, 3, 3]
    if R_hat.shape == (3, 3):
        R_hat = R_hat[np.newaxis, :]
    if R.shape == (3, 3):
        R = R[np.newaxis, :]
    n = R.shape[0]
    trace_idx = [0, 4, 8]
    trace = np.matmul(R_hat,
                      R.transpose(0, 2, 1)).reshape(n, -1)[:, trace_idx].sum(1)
    metric = np.arccos(((trace - 1) / 2).clip(-1, 1)) / np.pi * 180.0
    return metric


def angular_distance(R_hat, R):
    b = R_hat.shape[0]
    err = 0
    for i in range(b):
        rotDiff = R_hat[i] @ R[i].transpose(1, 0)
        trace = torch.trace(rotDiff)
        err += torch.acos(
            ((trace - 1.0) / 2.0).clamp(-0.99, 0.99)) / np.pi * 180
    err /= b
    return err


def axisangle_to_rotmat(axisangle):
    # pose: [n, 6]
    # return R: [n, 4, 4]
    n = axisangle.size(0)
    v = axisangle[:, :3]
    #theta = pose[:, 3]
    epsilon = 0.000000001  # used to handle 0/0
    v_length = torch.sqrt(torch.sum(v * v, dim=1))
    vx = (v[:, 0] + epsilon) / (v_length + epsilon)
    vy = (v[:, 1] + epsilon) / (v_length + epsilon)
    vz = (v[:, 2] + epsilon) / (v_length + epsilon)
    zero_ = torch.zeros_like(vx)
    m = torch.stack([zero_, -vz, vy, vz, zero_, -vx, -vy, vx,
                     zero_]).transpose(0, 1).view(n, 3, 3)

    I3 = Variable(torch.eye(3).view(1, 3, 3).repeat(n, 1, 1).cuda())
    R = Variable(torch.eye(4).view(1, 4, 4).repeat(n, 1, 1).cuda())
    R[:, :3, :3] = I3 + torch.sin(v_length).view(n, 1, 1) * m + (
        1 - torch.cos(v_length)).view(n, 1, 1) * torch.bmm(m, m)
    R[:, :3, 3] = R[:, :3, 3] + axisangle[:, 3:]
    return R


def axisangle_to_rotmatv2(axis, angle, trans):
    # pose: [n, 6]
    # return R: [n, 4, 4]
    n = len(axis)
    vx = axis[:, 0]
    vy = axis[:, 1]
    vz = axis[:, 2]
    zero_ = torch.zeros_like(vx)
    m = torch.stack([zero_, -vz, vy, vz, zero_, -vx, -vy, vx,
                     zero_]).transpose(0, 1).view(n, 3, 3)

    I3 = Variable(torch.eye(3).view(1, 3, 3).repeat(n, 1, 1).cuda())
    R = Variable(torch.eye(4).view(1, 4, 4).repeat(n, 1, 1).cuda())
    R[:, :3, :3] = I3 + torch.sin(angle).view(
        n, 1, 1) * m + (1 - torch.cos(angle)).view(n, 1, 1) * torch.bmm(m, m)
    R[:, :3, 3] = R[:, :3, 3] + trans
    return R


def find_edge_th(t):
    edge = torch.cuda.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] !=
                                               t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] !=
                                               t[:, :, :-1, :])
    return edge.float()


def find_edge(t):
    edge = np.zeros_like(t).astype('bool')
    edge[:, 1:] = edge[:, 1:] | (t[:, 1:] != t[:, :-1])
    edge[:, :-1] = edge[:, :-1] | (t[:, 1:] != t[:, :-1])
    edge[1:, :] = edge[1:, :] | (t[1:, :] != t[:-1, :])
    edge[:-1, :] = edge[:-1, :] | (t[1:, :] != t[:-1, :])
    return edge.astype('float')


def pnp(points_3d,
        points_2d,
        camera_matrix,
        method=cv2.SOLVEPNP_ITERATIVE,
        pose_init=None):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[
        0], 'points 3D and points 2D must have same number of vertices'
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    if pose_init is not None:
        R = pose_init[:3, :3]
        tr = (np.trace(R) - 1) / 2
        theta = np.arccos(tr.clip(-1, 1))
        pho = np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        rvec_init = pho / (2 * np.sin(theta)) * theta
        tvec_init = pose_init[:3, 3:4]
        #print(cv2.Rodrigues(rvec_init)[0]-R)

        #_, R_exp, t,inlier = cv2.solvePnPRansac(points_3d,
        _, R_exp, t = cv2.solvePnP(points_3d,
                                   points_2d,
                                   camera_matrix,
                                   dist_coeffs,
                                   rvec=np.array(rvec_init),
                                   tvec=np.array(tvec_init),
                                   useExtrinsicGuess=True,
                                   flags=method)
        # , None, None, False, cv2.SOLVEPNP_UPNP)
    else:
        success, R_exp, t = cv2.solvePnP(points_3d,
                                         points_2d,
                                         camera_matrix,
                                         dist_coeffs,
                                         flags=method)
        if not success:
            import ipdb
            ipdb.set_trace()

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)


def ray_box_intersection(min_corner, max_corner, ray_o, ray_d):
    dirfrac = 1.0 / (ray_d + 1e-16)
    t1 = (min_corner[:, 0:1, None] - ray_o[:, :, 0, :]) * dirfrac[:, :, 0, :]
    t2 = (max_corner[:, 0:1, None] - ray_o[:, :, 0, :]) * dirfrac[:, :, 0, :]
    t3 = (min_corner[:, 0:1, None] - ray_o[:, :, 1, :]) * dirfrac[:, :, 1, :]
    t4 = (max_corner[:, 0:1, None] - ray_o[:, :, 1, :]) * dirfrac[:, :, 1, :]
    t5 = (min_corner[:, 0:1, None] - ray_o[:, :, 2, :]) * dirfrac[:, :, 2, :]
    t6 = (max_corner[:, 0:1, None] - ray_o[:, :, 2, :]) * dirfrac[:, :, 2, :]
    a0 = torch.min(t1, t2)
    a1 = torch.min(t3, t4)
    a2 = torch.min(t5, t6)
    tmin = torch.max(a0, a1)
    tmin = torch.max(tmin, a2)
    a3 = torch.max(t1, t2)
    a4 = torch.max(t3, t4)
    a5 = torch.max(t5, t6)
    tmax = torch.min(a3, a4)
    tmax = torch.min(tmax, a5)
    mask = (tmin < tmax) & (tmin > 0)
    return tmin, tmax, mask


def r2n2_az_el_t_to_T(az, el, t_pred):
    img_w = 137
    img_h = 137

    # Calculate intrinsic matrix.
    F_MM = 35
    SKEW = 0
    SENSOR_SIZE_MM = 32
    f_u = F_MM * img_w / SENSOR_SIZE_MM
    f_v = F_MM * img_h / SENSOR_SIZE_MM
    u_0 = img_w / 2
    v_0 = img_h / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))
    blender_T = np.array([
        [1, 0., 0],
        [0, 0, -1],
        [0, 1, 0.],
    ])
    CAM_ROT = np.matrix(((1.910685676922942e-15, 4.371138828673793e-08,
                          1.0), (1.0, -4.371138828673793e-08, -0.0),
                         (4.371138828673793e-08, 1.0, -4.371138828673793e-08)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(
        np.matrix(
            ((ca * ce, -sa, ca * se), (sa * ce, ca, sa * se), (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    #T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    #T_world2cam = R_camfix * T_world2cam
    T_world2cam = t_pred

    RT = np.hstack((R_world2cam, T_world2cam.reshape(3, 1)))

    RT = np.asarray(RT)
    M = RT[:, :3] @ blender_T
    c = RT[:, 3:]
    RT = np.concatenate([M, c], axis=1)
    RT = np.concatenate([RT, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
    return RT


def r2n2_az_el_t_to_T_th(az, el, t_pred):
    """
    az: [n]
    el: [n]
    t_pred:[n,3]
    """
    img_w = 137
    img_h = 137
    n = len(az)

    # Calculate intrinsic matrix.
    F_MM = 35
    SKEW = 0
    SENSOR_SIZE_MM = 32
    f_u = F_MM * img_w / SENSOR_SIZE_MM
    f_v = F_MM * img_h / SENSOR_SIZE_MM
    u_0 = img_w / 2
    v_0 = img_h / 2
    K = torch.FloatTensor([[f_u, SKEW, u_0], [0, f_v, v_0], [0, 0, 1]]).cuda()
    blender_T = torch.FloatTensor([
        [1, 0., 0],
        [0, 0, -1],
        [0, 1, 0.],
    ]).cuda()
    CAM_ROT = torch.FloatTensor(
        [[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
         [1.0, -4.371138828673793e-08, -0.0],
         [4.371138828673793e-08, 1.0, -4.371138828673793e-08]]).cuda()

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = torch.sin(-az / 180.0 * np.pi)
    ca = torch.cos(-az / 180.0 * np.pi)
    se = torch.sin(-el / 180.0 * np.pi)
    ce = torch.cos(-el / 180.0 * np.pi)
    o = torch.zeros_like(ce)
    R_world2obj = torch.stack(
        (ca * ce, -sa, ca * se, sa * ce, ca, sa * se, -se, o, ce),
        -1).view(n, 3, 3).transpose(1, 2)

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = CAM_ROT.t()
    R_world2cam = R_obj2cam.unsqueeze(0) @ R_world2obj

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = torch.FloatTensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).cuda()
    R_world2cam = R_camfix.unsqueeze(0) @ R_world2cam
    T_world2cam = t_pred

    RT = torch.cat((R_world2cam, T_world2cam.reshape(n, 3, 1)), dim=-1)

    RT[:, :3, :3] = RT[:, :3, :3] @ blender_T.unsqueeze(0)
    lastrow = torch.FloatTensor([0, 0, 0, 1]).float().cuda()
    RT = torch.cat([RT, lastrow.view(1, 1, 4).repeat(n, 1, 1)], dim=1)
    return RT


### For pose init
### Util functions
def angles_to_matrix(angles):
    """Compute the rotation matrix from euler angles for a mini-batch.
    This is a PyTorch implementation computed by myself for calculating
    R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    
    For the original numpy implementation in StarMap, you can refer to:
    https://github.com/xingyizhou/StarMap/blob/26223a6c766eab3c22cddae87c375150f84f804d/tools/EvalCls.py#L20
    """
    azi = angles[:, 0]
    ele = angles[:, 1]
    rol = angles[:, 2]
    element1 = (torch.cos(rol) * torch.cos(azi) -
                torch.sin(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element2 = (torch.sin(rol) * torch.cos(azi) +
                torch.cos(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(
                    1)  ### row2, col1
    element3 = (torch.sin(ele) * torch.sin(azi)).unsqueeze(1)  ### row3, col1
    element4 = (-torch.cos(rol) * torch.sin(azi) -
                torch.sin(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(
                    1)  ### row1, col3
    element5 = (-torch.sin(rol) * torch.sin(azi) +
                torch.cos(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element6 = (torch.sin(ele) * torch.cos(azi)).unsqueeze(1)
    element7 = (torch.sin(rol) * torch.sin(ele)).unsqueeze(1)
    element8 = (-torch.cos(rol) * torch.sin(ele)).unsqueeze(1)
    element9 = (torch.cos(ele)).unsqueeze(1)
    return torch.cat((element1, element2, element3, element4, element5,
                      element6, element7, element8, element9),
                     dim=1)


def rotation_err(preds, targets):
    """compute rotation error for viewpoint estimation"""
    preds = preds.float().clone()
    targets = targets.float().clone()

    # get elevation and inplane-rotation in the right format
    # R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    preds[:, 1] = preds[:, 1]  #- 180.
    #preds[:, 2] = preds[:, 2] - 180.
    targets[:, 1] = targets[:, 1]  # - 180.
    #targets[:, 2] = targets[:, 2] - 180.

    # change degrees to radians
    preds = preds * np.pi / 180.
    targets = targets * np.pi / 180.

    # get rotation matrix from euler angles
    R_pred = angles_to_matrix(preds)
    R_gt = angles_to_matrix(targets)

    # compute the angle distance between rotation matrix in degrees
    """
    transform = torch.from_numpy(np.array([[-1, 0,  0],
                      [0, 0, 1],
                      [0, 1, 0]])).cuda().float()
    R_pred = torch.matmul(transform, R_pred).transpose(1,0)
    """
    R_err = torch.acos(((torch.sum(R_pred * R_gt, 1)).clamp(-1., 3.) - 1.) / 2)
    R_err = R_err * 180. / np.pi

    return R_err


def rotation_acc(preds, targets, th=30.):
    R_err = rotation_err(preds, targets)
    return 100. * torch.mean((R_err <= th).float()), torch.mean(R_err).float()


def get_pred_from_cls_output(outputs):
    preds = []
    for n in range(0, len(outputs)):
        output = outputs[n]
        _, pred = output.topk(1, 1, True, True)
        preds.append(pred.view(-1))
    return preds


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def area_under_curve(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


def Quaternion2rot(Q):
    # Q: [n, 4]
    # assume Q: [qw, qx, qy, qz]
    R_00 = Q[:, 0]**2 + Q[:, 1]**2 - Q[:, 2]**2 - Q[:, 3]**2
    R_01 = 2 * (Q[:, 1] * Q[:, 2] - Q[:, 0] * Q[:, 3])
    R_02 = 2 * (Q[:, 0] * Q[:, 2] + Q[:, 1] * Q[:, 3])
    R_10 = 2 * (Q[:, 1] * Q[:, 2] + Q[:, 0] * Q[:, 3])
    R_11 = Q[:, 0]**2 - Q[:, 1]**2 + Q[:, 2]**2 - Q[:, 3]**2
    R_12 = 2 * (Q[:, 2] * Q[:, 3] - Q[:, 0] * Q[:, 1])
    R_20 = 2 * (Q[:, 1] * Q[:, 3] - Q[:, 0] * Q[:, 2])
    R_21 = 2 * (Q[:, 0] * Q[:, 1] + Q[:, 2] * Q[:, 3])
    R_22 = Q[:, 0]**2 - Q[:, 1]**2 - Q[:, 2]**2 + Q[:, 3]**2
    R = torch.stack((R_00, R_01, R_02, R_10, R_11, R_12, R_20, R_21, R_22),
                    1).view(Q.size(0), 3, 3)
    return R


def rot2Quaternion(rot):
    # rot: [3,3]
    device = rot.device
    assert (rot.shape == (3, 3))
    tr = torch.trace(rot)
    if tr > 0:
        S = torch.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (rot[2, 1] - rot[1, 2]) / S
        qy = (rot[0, 2] - rot[2, 0]) / S
        qz = (rot[1, 0] - rot[0, 1]) / S
    elif (rot[0, 0] > rot[1, 1]) and (rot[0, 0] > rot[2, 2]):
        S = torch.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        qw = (rot[2, 1] - rot[1, 2]) / S
        qx = 0.25 * S
        qy = (rot[0, 1] + rot[1, 0]) / S
        qz = (rot[0, 2] + rot[2, 0]) / S
    elif rot[1, 1] > rot[2, 2]:
        S = torch.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        qw = (rot[0, 2] - rot[2, 0]) / S
        qx = (rot[0, 1] + rot[1, 0]) / S
        qy = 0.25 * S
        qz = (rot[1, 2] + rot[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        qw = (rot[1, 0] - rot[0, 1]) / S
        qx = (rot[0, 2] + rot[2, 0]) / S
        qy = (rot[1, 2] + rot[2, 1]) / S
        qz = 0.25 * S

    return torch.FloatTensor([qw, qx, qy, qz]).to(device)


def rot2Quaternion_np(rot):
    # rot: [3,3]
    assert (rot.shape == (3, 3))
    tr = np.trace(rot)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (rot[2, 1] - rot[1, 2]) / S
        qy = (rot[0, 2] - rot[2, 0]) / S
        qz = (rot[1, 0] - rot[0, 1]) / S
    elif (rot[0, 0] > rot[1, 1]) and (rot[0, 0] > rot[2, 2]):
        S = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        qw = (rot[2, 1] - rot[1, 2]) / S
        qx = 0.25 * S
        qy = (rot[0, 1] + rot[1, 0]) / S
        qz = (rot[0, 2] + rot[2, 0]) / S
    elif rot[1, 1] > rot[2, 2]:
        S = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        qw = (rot[0, 2] - rot[2, 0]) / S
        qx = (rot[0, 1] + rot[1, 0]) / S
        qy = 0.25 * S
        qz = (rot[1, 2] + rot[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        qw = (rot[1, 0] - rot[0, 1]) / S
        qx = (rot[0, 2] + rot[2, 0]) / S
        qy = (rot[1, 2] + rot[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])


def axisangle_to_rotmat(axisangle):
    # pose: [n, 6]
    # return R: [n, 4, 4]
    n = axisangle.size(0)
    v = axisangle[:, :3]
    #theta = pose[:, 3]
    epsilon = 0.000000001  # used to handle 0/0
    v_length = torch.sqrt(torch.sum(v * v, dim=1))
    vx = (v[:, 0] + epsilon) / (v_length + epsilon)
    vy = (v[:, 1] + epsilon) / (v_length + epsilon)
    vz = (v[:, 2] + epsilon) / (v_length + epsilon)
    zero_ = torch.zeros_like(vx)
    m = torch.stack([zero_, -vz, vy, vz, zero_, -vx, -vy, vx,
                     zero_]).transpose(0, 1).view(n, 3, 3)

    I3 = Variable(torch.eye(3).view(1, 3, 3).repeat(n, 1, 1).cuda())
    R = Variable(torch.eye(4).view(1, 4, 4).repeat(n, 1, 1).cuda())
    R[:, :3, :3] = I3 + torch.sin(v_length).view(n, 1, 1) * m + (
        1 - torch.cos(v_length)).view(n, 1, 1) * torch.bmm(m, m)
    R[:, :3, 3] = R[:, :3, 3] + axisangle[:, 3:]
    return R


def eval_mesh(mesh, pointcloud_tgt, normals_tgt, points_iou, occ_tgt):
    ''' Evaluates a mesh.
    Args:
        mesh (trimesh): mesh which should be evaluated
        pointcloud_tgt (numpy array): target point cloud
        normals_tgt (numpy array): target normals
        points_iou (numpy_array): points tensor for IoU evaluation
        occ_tgt (numpy_array): GT occupancy values for IoU points
    '''
    if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        pointcloud, idx = mesh.sample(100000, return_index=True)
        pointcloud = pointcloud.astype(np.float32)
        normals = mesh.face_normals[idx]
    else:
        pointcloud = np.empty((0, 3))
        normals = np.empty((0, 3))
    out_dict = eval_pointcloud(pointcloud, pointcloud_tgt, normals,
                               normals_tgt)

    if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        occ = check_mesh_contains(mesh, points_iou)
        out_dict['iou'] = compute_iou(occ, occ_tgt)
    else:
        out_dict['iou'] = 0.

    return out_dict


def o3d_mesh_to_trimesh(mesh):
    out_mesh = trimesh.Trimesh(vertices=np.array(mesh.vertices),
                               faces=np.array(mesh.triangles))
    return out_mesh


def freeze_param(net):
    for p in net.parameters():
        p.requires_grad = False


def create_gif(img_list, save_file, duration, loop):
    assert len(img_list) > 1
    img, *imgs = [Image.open(f) for f in img_list]
    img.save(
        fp=save_file,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=duration,
        loop=loop,
    )


def build_optimizer(model, config):
    name = config.TRAINER.OPTIMIZER
    lr = config.TRAINER.TRUE_LR

    if name == "adam":
        return torch.optim.Adam(model.parameters(),
                                lr=lr,
                                weight_decay=config.TRAINER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(),
                                 lr=lr,
                                 weight_decay=config.TRAINER.ADAMW_DECAY)
    else:
        raise ValueError(
            f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config.TRAINER.SCHEDULER_INTERVAL}
    name = config.TRAINER.SCHEDULER

    if name == 'MultiStepLR':
        scheduler.update({
            'scheduler':
            MultiStepLR(optimizer,
                        config.TRAINER.MSLR_MILESTONES,
                        gamma=config.TRAINER.MSLR_GAMMA)
        })
    elif name == 'CosineAnnealing':
        scheduler.update({
            'scheduler':
            CosineAnnealingLR(optimizer, config.TRAINER.COSA_TMAX)
        })
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config.TRAINER.ELR_GAMMA)})
    else:
        raise NotImplementedError()

    return scheduler


def make_scene_coord_figures(data, config, mode='evaluation'):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence']  # 'confidence'
    figures = {mode: []}
    i = 1
    mask = data['masks_half'][0, i].data.cpu().numpy()
    _min = data['scene_coords_half'][0,
                                     i].view(3,
                                             -1).min(-1)[0].data.cpu().numpy()
    _max = data['scene_coords_half'][0,
                                     i].view(3,
                                             -1).max(-1)[0].data.cpu().numpy()
    scene_coord_pred = (
        (data['scene_coord_pred'][0, i].data.cpu().numpy().transpose(1, 2, 0) -
         _min[None, None, :]) / (_max[None, None, :] - _min[None, None, :]) *
        255).astype('uint8')
    scene_coord_gt = (
        (data['scene_coords_half'][0, i].data.cpu().numpy().transpose(1, 2, 0)
         - _min[None, None, :]) / (_max[None, None, :] - _min[None, None, :]) *
        255).astype('uint8')
    scene_coord_pred[~mask] = 0
    scene_coord_gt[~mask] = 0
    #if config['LOFTR']['PRED_CONF']:
    if 0:
        pred_conf = torch.exp(data['pred_log_conf'][0, i]).data.cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(10, 6), dpi=75)
        axes[0].imshow(scene_coord_pred)
        axes[1].imshow(scene_coord_gt)
        axes[2].imshow(pred_conf)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=75)
        axes[0].imshow(scene_coord_pred)
        axes[1].imshow(scene_coord_gt)
    figures[mode].append(fig)
    return figures


def opencv_ransacPnP(scene_coord, K, mask):
    h, w = scene_coord.shape[:2]
    N = min(1000, mask.sum())
    assert (N >= 4)
    idv, idu = np.meshgrid(range(h), range(w), indexing='ij')
    grid = np.stack((idu, idv), -1).reshape(-1, 2)
    grid = grid[mask.reshape(-1)]
    scene_coord = scene_coord.reshape(-1, 3)[mask.reshape(-1)]
    sel = np.random.choice(len(grid), N, replace=False)
    imgpts = grid[sel].astype('float32')
    pts3d = scene_coord.reshape(-1, 3)[sel]
    imgpts = imgpts.reshape(-1, 1, 2)
    pts3d = pts3d.reshape(-1, 1, 3)
    success, R_vec, t, inliers = cv2.solvePnPRansac(pts3d,
                                                    imgpts,
                                                    K,
                                                    np.zeros([4]),
                                                    iterationsCount=5000,
                                                    reprojectionError=10,
                                                    flags=cv2.SOLVEPNP_P3P)
    if success:
        inliers = inliers[:, 0]
        num_inliers = len(inliers)
        inlier_ratio = len(inliers) / len(pts3d)
        success &= num_inliers >= 15
        ret, R_vec, t = cv2.solvePnP(pts3d[inliers],
                                     imgpts[inliers],
                                     K,
                                     np.zeros([4]),
                                     rvec=R_vec,
                                     tvec=t,
                                     useExtrinsicGuess=True,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        assert ret
        query_T_w = np.eye(4)
        query_T_w[:3, :3] = cv2.Rodrigues(R_vec)[0]
        query_T_w[:3, 3] = t[:, 0]
        #return query_T_w
        w_T_query = np.linalg.inv(query_T_w)
        return w_T_query
    else:
        print('OpenCV PnP fail')
        return np.eye(4)


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def get_rank_zero_only_logger(logger: _Logger):
    if rank_zero_only.rank == 0:
        return logger
    else:
        for _level in logger._core.levels.keys():
            level = _level.lower()
            setattr(logger, level, lambda x: None)
        logger._log = lambda x: None
    return logger


def setup_gpus(gpus: Union[str, int]) -> int:
    """ A temporary fix for pytorch-lighting 1.3.x """
    gpus = str(gpus)
    gpu_ids = []

    if ',' not in gpus:
        n_gpus = int(gpus)
        return n_gpus if n_gpus != -1 else torch.cuda.device_count()
    else:
        gpu_ids = [i.strip() for i in gpus.split(',') if i != '']

    # setup environment variables
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_devices is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in gpu_ids)
        visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        logger.warning(
            f'[Temporary Fix] manually set CUDA_VISIBLE_DEVICES when specifying gpus to use: {visible_devices}'
        )
    else:
        logger.warning(
            '[Temporary Fix] CUDA_VISIBLE_DEVICES already set by user or the main process.'
        )
    return len(gpu_ids)


def flattenList(x):
    return list(chain(*x))


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    
    Usage:
        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))
            
    When iterating over a generator, directly use of tqdm is also a solutin (but monitor the task queuing, instead of finishing)
        ret_vals = Parallel(n_jobs=args.world_size)(
                    delayed(lambda x: _compute_cov_score(pid, *x))(param)
                        for param in tqdm(combinations(image_ids, 2),
                                          desc=f'Computing cov_score of [{pid}]',
                                          total=len(image_ids)*(len(image_ids)-1)/2))
    Src: https://stackoverflow.com/a/58936697
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


