import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from .modules import *


class FeatureNet(nn.Module):
    def __init__(self,
                 model,
                 channel_in=3,
                 channel_out=32,
                 global_feature=False,
                 no_bn_act_last_layer=False,
                 normalize_feature_map=False):
        super(FeatureNet, self).__init__()
        self.normalize_feature_map = normalize_feature_map
        import torchvision.models as models
        if model == 'resnet18':
            self.base_model = models.resnet18(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            feat_enc_channels = [64, 64, 128, 256, 512]
        else:
            print('Not supported encoder: {}'.format(model))

        self.modules = dict()
        feat_dec_channels = np.array(
            [channel_out, 256, 256, 256, 512, feat_enc_channels[-1]])
        kernels = [3, 3, 3, 3, 1]
        if channel_in != 3:
            self.base_model.conv1 = nn.Conv2d(channel_in, 64, 7, 2, 3)
        for i in range(4, 0, -1):
            self.modules[("upconv", i,
                          0)] = self._make_block(feat_dec_channels[i + 1],
                                                 feat_dec_channels[i],
                                                 kernels[i])
            self.modules[("upconv", i, 1)] = self._make_block(
                feat_dec_channels[i] + feat_enc_channels[i - 1],
                feat_dec_channels[i], kernels[i])

        self.modules[("upconv", 0,
                      0)] = self._make_block(feat_dec_channels[1],
                                             feat_dec_channels[0])
        self.modules[("upconv", 0, 1)] = nn.Conv2d(feat_dec_channels[0],
                                                   feat_dec_channels[0], 3, 1,
                                                   1)

        self.temp = nn.ModuleList(list(self.modules.values()))

    def _make_block(self, channel_in, channel_out, kernel_size=3, pad=1):
        pad = {3: 1, 1: 0}
        return ConvBlock(
            channel_in,
            channel_out,
            kernel_size=kernel_size,
            pad=pad[kernel_size],
            conv_layer='Conv2d',
            norm_layer='BatchNorm2d',
        )

    def decoder(self, skip_feat):
        x = skip_feat[-1]
        for i in range(4, -1, -1):
            x = self.modules[("upconv", i, 0)](x)
            x = self.upsample(x)
            if i > 0:
                x = torch.cat((x, skip_feat[i - 1]), 1)
            x = self.modules[("upconv", i, 1)](x)
        return x

    def upsample(self, x):
        """Upsample input tensor by a factor of 2
        """
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x):
        feature = x
        skip_feat = []
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)

        feat = self.decoder(skip_feat)
        if self.normalize_feature_map:
            feat = F.normalize(feat, dim=1)
        return feat
