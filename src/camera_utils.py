import cv2
import numpy as np
import torch
import cvxpy as cp
from utils import angular_distance_np


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)

    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """
    def so3_to_SO3(self, w):  # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I + A * wx + B * wx @ wx
        return R

    def SO3_to_so3(self, R, eps=1e-7):  # [...,3,3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
            ..., None, None] % np.pi  # ln(R) will explode if theta==pi
        lnR = 1 / (2 * self.taylor_A(theta) + 1e-8) * (
            R - R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    def se3_to_SE3(self, wu, shape='3x4'):  # [...,3]
        w, u = wu.split([3, 3], dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        wu.shape[:-1]
        if shape == '4x4':
            bottom = torch.FloatTensor([0, 0, 0, 1]).to(Rt).view(1, -1).repeat(
                np.prod(wu.shape[:-1]), 1).view(*wu.shape[:-1], 1, 4)
            Rt = torch.cat((Rt, bottom), -2)
        return Rt

    def SE3_to_se3(self, Rt, eps=1e-8):  # [...,3,4]
        R, t = Rt.split([3, 1], dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta**2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    def skew_symmetric(self, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([
            torch.stack([O, -w2, w1], dim=-1),
            torch.stack([w2, O, -w0], dim=-1),
            torch.stack([-w1, w0, O], dim=-1)
        ],
                         dim=-2)
        return wx

    def taylor_A(self, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            if i > 0: denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1)**i * x**(2 * i) / denom
        return ans

    def taylor_B(self, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1)**i * x**(2 * i) / denom
        return ans

    def taylor_C(self, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1)**i * x**(2 * i) / denom
        return ans


def add_noise_to_world_to_cam_pose(lie,
                                   pose,
                                   noise_std=0.05,
                                   fix_first=True,
                                   N=1):
    se3_noise = torch.randn(pose.shape[0], N, 6) * noise_std
    if fix_first:
        se3_noise[0] = 0
    pose_noise = lie.se3_to_SE3(se3_noise, shape='4x4').data.cpu().numpy()
    pose_noise = pose_noise[:, 0]
    pose_noise = pose_noise @ pose
    err_r = angular_distance_np(pose_noise[:, :3, :3], pose[:, :3, :3])
    err_t = np.linalg.norm(pose_noise[:, :3, 3] - pose[:, :3, 3], axis=-1)
    return pose_noise, {'noise_r': err_r, 'noise_t': err_t}


def add_noise_to_pose(lie, pose, noise_std=0.05, fix_first=True, N=1):
    ## random perturb the cameras
    pose0 = pose[0]
    pose_norm = np.linalg.inv(pose0)[None, :, :] @ pose

    #torch.manual_seed(300)
    se3_noise = torch.randn(pose.shape[0], N, 6) * noise_std
    if fix_first:
        se3_noise[0] = 0
    pose_noise = lie.se3_to_SE3(se3_noise, shape='4x4').data.cpu().numpy()
    if N == 1:
        pose_noise = pose_noise[:, 0]
        pose_noise = pose0 @ pose_noise @ pose_norm
        err_r = angular_distance_np(pose_noise[:, :3, :3], pose[:, :3, :3])
        err_t = np.linalg.norm(pose_noise[:, :3, 3] - pose[:, :3, 3], axis=-1)
        return pose_noise, {'noise_r': err_r, 'noise_t': err_t}
    else:
        pose_noise = pose0.reshape(1, 1, 4,
                                   4) @ pose_noise @ pose_norm[:, None, :, :]
        err_r = angular_distance_np(pose_noise[:, :3, :3], pose[:, :3, :3])
        err_t = np.linalg.norm(pose_noise[:, :3, 3] - pose[:, :3, 3], axis=-1)
        return pose_noise, {'noise_r': err_r, 'noise_t': err_t}
        return pose_noise


def compare_rotations(R1, R2):
    cos_err = (torch.bmm(R1, R2.transpose(
        1, 2))[:, torch.arange(3), torch.arange(3)].sum(dim=-1) - 1) / 2
    cos_err[cos_err > 1] = 1
    cos_err[cos_err < -1] = -1
    return cos_err.acos() * 180 / np.pi


def get_cameras_accuracy(pred_Rs, gt_Rs, pred_ts, gt_ts, verbose=True):
    ''' Align predicted pose to gt pose and print cameras accuracy'''

    # find rotation
    d = pred_Rs.shape[-1]
    n = pred_Rs.shape[0]

    Q = torch.addbmm(torch.zeros(d, d, dtype=torch.double), gt_Rs,
                     pred_Rs.transpose(1, 2))
    Uq, _, Vq = torch.svd(Q)
    sv = torch.ones(d, dtype=torch.double)
    sv[-1] = torch.det(Uq @ Vq.transpose(0, 1))
    R_opt = Uq @ torch.diag(sv) @ Vq.transpose(0, 1)
    R_fixed = torch.bmm(R_opt.repeat(n, 1, 1), pred_Rs)

    # find translation
    pred_ts = pred_ts @ R_opt.transpose(0, 1)
    c_opt = cp.Variable()
    t_opt = cp.Variable((1, d))

    constraints = []
    obj = cp.Minimize(
        cp.sum(
            cp.norm(gt_ts.numpy() - (c_opt * pred_ts.numpy() + np.ones(
                (n, 1), dtype=np.double) @ t_opt),
                    axis=1)))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    t_fixed = c_opt.value * pred_ts.numpy() + np.ones(
        (n, 1), dtype=np.double) * t_opt.value

    # Calculate transaltion error
    t_error = np.linalg.norm(t_fixed - gt_ts.numpy(), axis=-1)
    t_error = t_error
    t_error_mean = np.mean(t_error)
    t_error_medi = np.median(t_error)

    # Calculate rotation error
    R_error = compare_rotations(R_fixed, gt_Rs)

    R_error = R_error.numpy()
    R_error_mean = np.mean(R_error)
    R_error_medi = np.median(R_error)

    if verbose:
        print(
            'CAMERAS EVALUATION: R error mean = {0} ; t error mean = {1} ; R error median = {2} ; t error median = {3}'
            .format("%.4f" % R_error_mean, "%.4f" % t_error_mean,
                    "%.4f" % R_error_medi, "%.4f" % t_error_medi))

    # return alignment and aligned pose
    return {
        'R_opt': R_opt.numpy(),
        't_opt': t_opt.value,
        'c_opt': c_opt.value,
        'R_fixed': R_fixed.numpy(),
        't_fixed': t_fixed,
        'R_errors': R_error,
        't_errors': t_error
    }
