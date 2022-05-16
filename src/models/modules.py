import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np
from lib.libmise import MISE
from utils import convert_sdf_samples_to_trianglemesh, o3d_mesh, write_ply, angular_distance_np, npy, convert_sdf_samples_to_trianglemesh_onet
from .embedder import *
from torch import distributed as dist


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        pad=1,
        downsample=True,
        norm='bn',
        conv_layer='Conv3d',
        deconv_layer='ConvTranspose3d',
        norm_layer='BatchNorm3d',
        activation='relu',
        track_running_stats=True,
    ):
        super(ConvBlock, self).__init__()
        self.norm = norm
        self.activation = activation
        if downsample:
            self.conv = getattr(nn, conv_layer)(in_channels,
                                                out_channels,
                                                kernel_size,
                                                stride=stride,
                                                padding=pad,
                                                bias=False)
        else:
            self.conv = getattr(nn, deconv_layer)(in_channels,
                                                  out_channels,
                                                  kernel_size=kernel_size,
                                                  padding=pad,
                                                  output_padding=1,
                                                  stride=stride,
                                                  bias=False)
        if self.norm == 'bn':
            self.bn = getattr(nn, norm_layer)(
                out_channels, track_running_stats=track_running_stats)

    def forward(self, x):
        x = self.conv(x)
        if self.norm == 'bn':
            x = self.bn(x)
        elif self.norm == 'ln':
            x = F.layer_norm(x, x.shape[1:])
        if self.activation == 'relu':
            return F.relu(x)
        else:
            return x


CE = nn.CrossEntropyLoss()
Huber = nn.SmoothL1Loss()


def delta_loss(pred_azi, pred_ele, pred_rol, target_azi, target_ele,
               target_rol):
    # compute the ground truth delta value according to angle value and bin size
    bin = (np.pi * 2) / pred_azi.shape[-1]
    target_delta_azi = ((target_azi % bin) / bin) - 0.5
    target_delta_ele = ((target_ele % bin) / bin) - 0.5
    target_delta_rol = ((target_rol % bin) / bin) - 0.5

    # compute the delta prediction in the ground truth bin
    target_label_azi = ((target_azi - (-np.pi)) // bin).long()
    target_label_ele = ((target_ele - (-np.pi)) // bin).long()
    target_label_rol = ((target_rol - (-np.pi)) // bin).long()
    delta_azi = pred_azi[torch.arange(pred_azi.size(0)),
                         target_label_azi].tanh() / 2
    delta_ele = pred_ele[torch.arange(pred_ele.size(0)),
                         target_label_ele].tanh() / 2
    delta_rol = pred_rol[torch.arange(pred_rol.size(0)),
                         target_label_rol].tanh() / 2
    pred_delta = torch.cat((delta_azi.unsqueeze(1), delta_ele.unsqueeze(1),
                            delta_rol.unsqueeze(1)), 1)
    target_delta = torch.cat(
        (target_delta_azi.unsqueeze(1), target_delta_ele.unsqueeze(1),
         target_delta_rol.unsqueeze(1)), 1)
    return Huber(5. * pred_delta, 5. * target_delta)


def cross_entropy_loss(pred, target, range):
    ## [-pi, pi]
    binSize = range / pred.size(1)
    trueLabel = (target - (-np.pi)) // binSize
    return CE(pred, trueLabel.long())


class CELoss(nn.Module):
    def __init__(self, range):
        super(CELoss, self).__init__()
        self.__range__ = range
        return

    def forward(self, pred, target):
        return cross_entropy_loss(pred, target, self.__range__)


class DeltaLoss(nn.Module):
    def __init__(self):
        super(DeltaLoss, self).__init__()
        return

    def forward(self, pred_azi, pred_ele, pred_rol, target_azi, target_ele,
                target_rol):
        return delta_loss(pred_azi, pred_ele, pred_rol, target_azi, target_ele,
                          target_rol)


class Encoder3D(nn.Module):
    def __init__(self,
                 channel_in=32,
                 channel_out=1,
                 down_sample_times=1,
                 kernel_size=3,
                 nhidden=64,
                 norm='bn'):
        super(Encoder3D, self).__init__()
        self.conv0 = ConvBlock(channel_in, nhidden, norm=norm)
        channels = [
            nhidden, nhidden, nhidden, nhidden * 2, nhidden * 2, nhidden * 4,
            nhidden * 8
        ]
        self.down_sample_times = down_sample_times
        for i in range(down_sample_times):
            if i == down_sample_times - 1:
                k = 1
                pad = 0
                norm_ = None
            else:
                k = 3
                pad = 1
                norm_ = norm
            setattr(
                self, 'conv_%d' % i,
                nn.Sequential(
                    ConvBlock(channels[i],
                              channels[i + 1],
                              stride=2,
                              kernel_size=k,
                              pad=pad,
                              norm=norm_),
                    ConvBlock(channels[i + 1],
                              channels[i + 1],
                              kernel_size=k,
                              pad=pad,
                              norm=norm_)))

        self.last_conv = nn.Conv3d(channels[-1],
                                   channel_out,
                                   1,
                                   stride=1,
                                   padding=0)

    def forward(self, x):
        x = self.conv0(x)
        for i in range(self.down_sample_times):
            x = getattr(self, 'conv_%d' % i)(x)
        return self.last_conv(x).view(x.shape[0], -1)


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''
    def __init__(self, size_in, size_out=None, size_h=None, activation='relu'):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        if activation == 'relu':
            self.actvn = nn.ReLU()
        elif activation == 'softplus':
            self.actvn = nn.Softplus(beta=100)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        #nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        valid = \
                (ix_nw >= 0) & (ix_nw <= IW - 1) & \
                (iy_nw >= 0) & (iy_nw <= IH - 1) & \
                (ix_ne >= 0) & (ix_ne <= IW - 1) & \
                (iy_ne >= 0) & (iy_ne <= IH - 1) & \
                (ix_sw >= 0) & (ix_sw <= IW - 1) & \
                (iy_sw >= 0) & (iy_sw <= IH - 1) & \
                (ix_se >= 0) & (ix_se <= IW - 1) & \
                (iy_se >= 0) & (iy_se <= IH - 1)
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(
        N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(
        N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(
        N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(
        N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    out_val = out_val * valid.unsqueeze(1)
    return out_val


def grid_sample3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():
        ix_bnw = torch.floor(ix)
        iy_bnw = torch.floor(iy)
        iz_bnw = torch.floor(iz)

        ix_bne = ix_bnw + 1
        iy_bne = iy_bnw
        iz_bne = iz_bnw

        ix_bsw = ix_bnw
        iy_bsw = iy_bnw + 1
        iz_bsw = iz_bnw

        ix_bse = ix_bnw + 1
        iy_bse = iy_bnw + 1
        iz_bse = iz_bnw

        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz) + 1

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

    bnw = (ix_bse - ix) * (iy_bse - iy) * (iz_tse - iz)
    bne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_tse - iz)
    bsw = (ix_bne - ix) * (iy - iy_bne) * (iz_tse - iz)
    bse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_tse - iz)
    tnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_bse)
    tne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_bse)
    tsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_bse)
    tse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_bse)

    with torch.no_grad():
        valid = \
                (ix_bnw >= 0) & (ix_bnw <= IW - 1) & \
                (iy_bnw >= 0) & (iy_bnw <= IH - 1) & \
                (iz_bnw >= 0) & (iz_bnw <= ID - 1) & \
                (ix_bne >= 0) & (ix_bne <= IW - 1) & \
                (iy_bne >= 0) & (iy_bne <= IH - 1) & \
                (iz_bne >= 0) & (iz_bne <= ID - 1) & \
                (ix_bsw >= 0) & (ix_bsw <= IW - 1) & \
                (iy_bsw >= 0) & (iy_bsw <= IH - 1) & \
                (iz_bsw >= 0) & (iz_bsw <= ID - 1) & \
                (ix_bse >= 0) & (ix_bse <= IW - 1) & \
                (iy_bse >= 0) & (iy_bse <= IH - 1) & \
                (iz_bse >= 0) & (iz_bse <= ID - 1) & \
                (ix_tnw >= 0) & (ix_tnw <= IW - 1) & \
                (iy_tnw >= 0) & (iy_tnw <= IH - 1) & \
                (iz_tnw >= 0) & (iz_tnw <= ID - 1) & \
                (ix_tne >= 0) & (ix_tne <= IW - 1) & \
                (iy_tne >= 0) & (iy_tne <= IH - 1) & \
                (iz_tne >= 0) & (iz_tne <= ID - 1) & \
                (ix_tsw >= 0) & (ix_tsw <= IW - 1) & \
                (iy_tsw >= 0) & (iy_tsw <= IH - 1) & \
                (iz_tsw >= 0) & (iz_tsw <= ID - 1) & \
                (ix_tse >= 0) & (ix_tse <= IW - 1) & \
                (iy_tse >= 0) & (iy_tse <= IH - 1) & \
                (iz_tse >= 0) & (iz_tse <= ID - 1)
        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

    image = image.view(N, C, ID * IH * IW)

    bnw_val = torch.gather(image, 2,
                           (iz_bnw * IH * IW + iy_bnw * IW +
                            ix_bnw).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2,
                           (iz_bne * IH * IW + iy_bne * IW +
                            ix_bne).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2,
                           (iz_bsw * IH * IW + iy_bsw * IW +
                            ix_bsw).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2,
                           (iz_bse * IH * IW + iy_bse * IW +
                            ix_bse).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))

    tnw_val = torch.gather(image, 2,
                           (iz_tnw * IH * IW + iy_tnw * IW +
                            ix_tnw).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2,
                           (iz_tne * IH * IW + iy_tne * IW +
                            ix_tne).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2,
                           (iz_tsw * IH * IW + iy_tsw * IW +
                            ix_tsw).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2,
                           (iz_tse * IH * IW + iy_tse * IW +
                            ix_tse).long().view(N, 1,
                                                D * H * W).repeat(1, C, 1))

    out_val = (bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W) +
               tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W))
    out_val = out_val * valid.unsqueeze(1)

    return out_val


def extract_mesh(sdf_query_fn,
                 corners,
                 threshold,
                 post_processing=False,
                 sign=1,
                 level=2):
    max_batch_size = 4 * 32 * 32 * 32
    cnt = 0
    with torch.no_grad():
        mesh_extractor = MISE(32, level, threshold)
        dim = 32 * (2**level)
        voxel_size = (corners[1] - corners[0]) / (dim)
        points = mesh_extractor.query()
        while points.shape[0] != 0:
            # Query points
            pointsf = torch.FloatTensor(points).cuda()
            # Normalize to bounding box
            pointsf = pointsf / mesh_extractor.resolution
            pointsf = corners[0][None, :] + pointsf * (corners[1] -
                                                       corners[0])[None, :]
            # Evaluate model and update
            values = sdf_query_fn(pointsf)
            values *= sign
            cnt += pointsf.shape[0]
            values = values.data.cpu().numpy().astype(np.float64)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()

        sdf = mesh_extractor.to_dense()
    try:
        vert, face = convert_sdf_samples_to_trianglemesh(
            sdf,
            corners[0].data.cpu().numpy(),
            voxel_size.data.cpu().numpy(),
            None,
            None,
            threshold=threshold,
            need_normal=False)
        mesh = o3d_mesh(vert, face)
    except Exception as e:
        mesh = None
    return mesh


def build_head(channel_in, channel_out, nhidden=32):
    return nn.Sequential(
        nn.Conv2d(channel_in, nhidden, 3, 1, 1),
        nn.BatchNorm2d(nhidden),
        nn.ReLU(),
        nn.Conv2d(nhidden, nhidden, 3, 1, 1, 1),
        nn.BatchNorm2d(nhidden),
        nn.ReLU(),
        nn.Conv2d(nhidden, nhidden, 3, 1, 1, 1),
        nn.BatchNorm2d(nhidden),
        nn.ReLU(),
        nn.Conv2d(nhidden, channel_out, 1),
    )
