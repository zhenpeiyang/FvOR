from utils import angular_distance_np, npy
import torch.nn as nn
import torch.nn.functional as F
import torch
from camera_utils import Lie


class PnP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lie = Lie()

    def objective(self, x3, f3, f2, weight, K, R, t, T_init):
        b, n, c, h, w = f2.shape
        with torch.no_grad():
            b, n, N, _ = x3.shape
            y = R @ x3.permute(0, 1, 3, 2) + t
            f = K[:, :, :3, :3] @ y
            x2_hat, J0, _ = self.project(f)
            u2_hat, v2_hat = x2_hat[..., 0], x2_hat[..., 1]
            u2_hat = u2_hat / (w - 1) * 2. - 1
            v2_hat = v2_hat / (h - 1) * 2. - 1
            x2_hat_norm = torch.stack((u2_hat, v2_hat), -1)
            f2_interp = F.grid_sample(f2.view(b * n, c, h, w),
                                      x2_hat_norm.view(b * n, -1, 1, 2),
                                      align_corners=True,
                                      padding_mode='zeros').view(
                                          b, n, c, h, w)
            f2_interp = f2_interp.view(b, n, c, -1).transpose(-2, -1)
            loss = self.lambda_f * (weight.view(b, n, -1) *
                                    (f2_interp - f3).pow(2).sum(dim=-1)).sum()
            loss_reg_r = self.lambda_reg * (T_init[:, :, :3, :3] -
                                            R).pow(2).sum([-2, -1]).sum()
            loss_reg_t = self.lambda_reg * (T_init[:, :, :3, 3:4] -
                                            t).pow(2).sum([-2, -1]).sum()

            loss += loss_reg_r + loss_reg_t

        return loss

    def project(self, xyz):
        # x: [b, n, 3, N]
        b, n, _, N = xyz.shape
        x, y, z = torch.split(xyz, 1, dim=2)
        zinv = torch.where(z > 0.1, 1.0 / z, torch.zeros_like(z))
        uv = torch.stack((x * zinv, y * zinv), -1).squeeze(2)
        zinv2 = zinv**2
        o = torch.zeros_like(z)
        J = torch.stack((zinv, o, -x * zinv2, o, zinv, -y * zinv2),
                        -1).squeeze(2).view(b, n, N, 2, 3)
        mask = ((zinv > 0).sum(-1) > 0).squeeze(-1)
        return uv, J, mask

    def cross_product_matrix(self, x):
        # x: [b, n, 3, N]
        b, n, _, N = x.shape
        x0, x1, x2 = torch.split(x, 1, dim=2)

        o = torch.zeros_like(x0)
        m = torch.stack((o, -x2, x1, x2, o, -x0, -x1, x0, o),
                        -1).squeeze(2).view(b, n, N, 3, 3)
        return m

    def hat(self, a, dim=1):
        a1, a2, a3 = torch.split(a, 1, dim=dim)
        zz = torch.zeros_like(a1)
        ax = torch.stack([
            torch.cat([zz, -a3, a2], dim=dim),
            torch.cat([a3, zz, -a1], dim=dim),
            torch.cat([-a2, a1, zz], dim=dim)
        ],
                         dim=dim)
        return ax

    def feature_grad(self, f2, x2_hat):
        b, n, c, h, w = f2.shape
        u2_hat, v2_hat = x2_hat[..., 0], x2_hat[..., 1]
        u2_hat = u2_hat / (w - 1) * 2. - 1
        v2_hat = v2_hat / (h - 1) * 2. - 1
        grid = torch.stack((u2_hat, v2_hat), -1)
        grid_l = grid.clone()
        grid_l[..., 0] -= (2. / (w - 1))

        grid_r = grid.clone()
        grid_r[..., 0] += (2. / (w - 1))

        grid_b = grid.clone()
        grid_b[..., 1] += (2. / (h - 1))

        grid_t = grid.clone()
        grid_t[..., 1] -= (2. / (h - 1))

        def get_features(grid):
            return F.grid_sample(f2.view(b * n, c, h, w),
                                 grid.view(b * n, -1, 1, 2),
                                 align_corners=True,
                                 padding_mode='zeros').view(b, n, c, h, w)

        pts_feature_l = get_features(grid_l)
        pts_feature_r = get_features(grid_r)
        pts_feature_t = get_features(grid_t)
        pts_feature_b = get_features(grid_b)
        pts_feature_grad_x = 0.5 * (pts_feature_r - pts_feature_l)
        pts_feature_grad_y = 0.5 * (pts_feature_b - pts_feature_t)
        pts_feature_grad = torch.stack(
            (pts_feature_grad_x, pts_feature_grad_y), dim=-1)
        pts_feature_grad = pts_feature_grad.view(b, n, c, -1,
                                                 2).transpose(2, 3)
        return pts_feature_grad

    def forward(
        self,
        x3,
        f3,
        f2,
        weight,
        K,
        T0,
        T_gt,
        outputs,
        postfix='',
        verbose=False,
        T_init=None,
        use_double=False,
        clamp=False,
        lambda_reg=1.0e4,
        lambda_damping=1e2,
    ):
        """Optimize camera poses by aligning feature maps.
        Args:
            x3 (Tensor [b, m, n, 3]): 3D surface points obtained by rendering current implicit shape.
            f3 (Tensor [b, m, n, c]): The corresponding feature descriptor for x3.
            f2 (Tensor [b, m, c, h, w]): The feature map for each view.
            weight (Tensor [b, m, n]): The weight for each point in x3.
            K (Tensor [b, m, 4, 4]): The intrinsic matrix for each view.
            T0 (Tensor [b, m, 4, 4]): The initial camera pose for each view.
            T_gt (Tensor [b, m, 4, 4]): The ground truth camera pose for each view.
        Returns:
            dict: Optimized camera poses, and some intermediate results.
        """
        if use_double:
            x3 = x3.double()
            f3 = f3.double()
            f2 = f2.double()
            T0 = T0.double()
            K = K.double()
            T_gt = T_gt.double()
            weight = weight.double()

        R = T0[:, :, :3, :3]
        t = T0[:, :, :3, 3:4]
        K = K[:, :, :3, :3]
        x3_reshape = x3.transpose(-2, -1)
        self.lambda_reg = lambda_reg
        self.lambda_reg_r = lambda_reg * 1.
        self.lambda_f = 1.0
        b, n, c, h, w = f2.shape
        for i in range(1):
            loss = self.objective(x3, f3, f2, weight, K, R, t, T_init)
            if verbose:
                print('iter ', i, ' ', loss.item())
                print(
                    ' angular distance ',
                    angular_distance_np(npy(R.view(-1, 3, 3)),
                                        npy(T_gt[:, :, :3, :3].view(
                                            -1, 3, 3))).mean(), 'translation',
                    (t.view(-1, 3) -
                     T_gt[:, :, :3, 3].view(-1, 3)).norm(dim=-1).mean())
            b, n, N, _ = x3.shape
            y = R @ x3_reshape + t
            f = K @ y
            x2_hat, J0, mask = self.project(f)
            u2_hat, v2_hat = x2_hat[..., 0], x2_hat[..., 1]
            u2_hat = u2_hat / (w - 1) * 2. - 1
            v2_hat = v2_hat / (h - 1) * 2. - 1
            x2_hat_norm = torch.stack((u2_hat, v2_hat), -1)

            ycross = self.cross_product_matrix(y)
            I = torch.eye(3).view(1, 1, 1, 3, 3).repeat(b, n, N, 1, 1).cuda()
            J = J0 @ K.unsqueeze(2) @ (torch.cat((-ycross, I), -1))
            Jf = self.feature_grad(f2, x2_hat)
            J = Jf @ J
            H = torch.einsum('...aji,...ajk->...ik',
                             self.lambda_f * weight[..., None, None] * J, J)
            zero = torch.zeros_like(R)
            I = torch.eye(3).view(1, 1, 3, 3).repeat(b, n, 1, 1).float().to(J)
            J_reg = torch.cat((zero, -I), -1)
            H_reg = torch.einsum('...ji,...jk->...ik', self.lambda_reg * J_reg,
                                 J_reg)

            J_reg_r1 = self.cross_product_matrix(R[:, :, :3, 0:1]).squeeze(2)
            J_reg_r2 = self.cross_product_matrix(R[:, :, :3, 1:2]).squeeze(2)
            J_reg_r3 = self.cross_product_matrix(R[:, :, :3, 2:3]).squeeze(2)
            H_reg_r = torch.zeros_like(H_reg)
            H_reg_r[:, :, :3, :3] = torch.einsum(
                '...ji,...jk->...ik',
                self.lambda_reg_r * J_reg_r1, J_reg_r1) + torch.einsum(
                    '...ji,...jk->...ik',
                    self.lambda_reg_r * J_reg_r2, J_reg_r2) + torch.einsum(
                        '...ji,...jk->...ik', self.lambda_reg_r * J_reg_r3,
                        J_reg_r3)

            H = H + H_reg + H_reg_r
            f2_interp = F.grid_sample(f2.view(b * n, c, h, w),
                                      x2_hat_norm.view(b * n, -1, 1, 2),
                                      align_corners=True,
                                      padding_mode='zeros').view(
                                          b, n, c, h, w)
            f2_interp = f2_interp.view(b, n, c, -1).transpose(-2, -1)
            Jtg = torch.einsum('...aji,...ajk->...ik',
                               self.lambda_f * weight[..., None, None] * J,
                               (f2_interp - f3).unsqueeze(-1))
            g_reg = (T_init[:, :, :3, 3:4] - t)
            Jtg_reg = self.lambda_reg * J_reg.transpose(-2, -1) @ g_reg
            g_reg_r1 = (T_init[:, :, :3, 0:1] - R[:, :, :3, 0:1])
            g_reg_r2 = (T_init[:, :, :3, 1:2] - R[:, :, :3, 1:2])
            g_reg_r3 = (T_init[:, :, :3, 2:3] - R[:, :, :3, 2:3])

            Jtg_reg_r1 = self.lambda_reg_r * J_reg_r1.transpose(-2,
                                                                -1) @ g_reg_r1
            Jtg_reg_r2 = self.lambda_reg_r * J_reg_r2.transpose(-2,
                                                                -1) @ g_reg_r2
            Jtg_reg_r3 = self.lambda_reg_r * J_reg_r3.transpose(-2,
                                                                -1) @ g_reg_r3
            Jtg_reg_r = torch.zeros_like(Jtg_reg)
            Jtg_reg_r[:, :, :3, :] = Jtg_reg_r1 + Jtg_reg_r2 + Jtg_reg_r3

            Jtg = Jtg + Jtg_reg + Jtg_reg_r

            lambda2 = 0.0001 * 0
            H = H + (lambda_damping + lambda2 * H) * torch.eye(6)[
                None, None, :, :].repeat(b, n, 1, 1).cuda()
            try:
                chol = torch.cholesky(H.cpu()).cuda()
                delta = torch.cholesky_solve(-Jtg, chol).squeeze(-1)
            except:
                print('Cholesky fail')
                outputs['pose_pnp%s' % postfix] = T0
                outputs['energy'] = 0
                return outputs

            THRESH = 2e-1
            if clamp:
                norm_ = delta.norm(dim=-1)
                norm_clamp = norm_.clamp(None, 0.03)
                delta = delta / (
                    1e-12 + norm_.unsqueeze(-1)) * norm_clamp.unsqueeze(-1)
            T_inc = self.lie.se3_to_SE3(delta)
            ## avoid nan grad on all zero supports
            To = torch.eye(4).to(f3).cuda().view(1, 1, 4,
                                                 4).repeat(b, n, 1,
                                                           1)[:, :, :3, :]
            T_inc = torch.where(
                mask.view(b, n, 1, 1).repeat(1, 1, 3, 4), T_inc, To)
            R = T_inc[:, :, :3, :3] @ R
            t = T_inc[:, :, :3, :3] @ t + T_inc[:, :, :3, 3:4]
        T = torch.eye(4).to(f3).cuda().view(1, 1, 4, 4).repeat(b, n, 1, 1)
        T[:, :, :3, :3] = R
        T[:, :, :3, 3:4] = t
        if verbose:
            print(
                'final angular distance ',
                angular_distance_np(npy(R.view(-1, 3, 3)),
                                    npy(T_gt[:, :, :3, :3].view(-1, 3,
                                                                3))).mean(),
                'translation',
                (t.view(-1, 3) -
                 T_gt[:, :, :3, 3].view(-1, 3)).norm(dim=-1).mean())
        outputs['pose_pnp%s' % postfix] = T.float()
        outputs['energy'] = loss.item()
        return outputs
