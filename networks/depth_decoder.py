import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import upsample, Conv3x3, ConvBlock, disp_to_depth, ResBlock


class DepthDecoder(nn.Module):
    def __init__(self, scales=range(4), num_output_channels=1, use_skips=True, use_uncert=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.use_uncert = use_uncert
        self.scales = scales
        self.convs = OrderedDict()
        self.frame_nums = [0, -1, 1]
        self.f_to_ch = {0: 0,
                        -1: 1,
                        1: 2}
        # decoder
        # self.convs[("upconv", 3, 0)] = ConvBlock(512, 256)
        # self.convs[("upconv", 3, 1)] = ConvBlock(256, 128)

        self.convs[("upconv", 3, 0)] = ConvBlock(512, 128)
        self.convs[("upconv", 3, 1)] = ConvBlock(256, 128)

        self.convs[("upconv", 2, 0)] = ConvBlock(128, 64)  # Needs up-sampling after this layer
        self.convs[("upconv", 2, 1)] = ConvBlock(128, 64)  # Needs concatenation before this layer

        self.convs[("upconv", 1, 0)] = ConvBlock(64, 32)  # Needs up-sampling after this layer
        self.convs[("upconv", 1, 1)] = ConvBlock(64, 32)  # Needs concatenation before this layer

        self.convs[("upconv", 0, 0)] = ConvBlock(32, 16)  # Needs up-sampling after this layer
        self.convs[("upconv", 0, 1)] = ConvBlock(16, 16)  # Needs concatenation before this layer

        self.convs[("dispconv", 3)] = Conv3x3(128, self.num_output_channels)
        self.convs[("dispconv", 2)] = Conv3x3(64, self.num_output_channels)
        self.convs[("dispconv", 1)] = Conv3x3(32, self.num_output_channels)
        self.convs[("dispconv", 0)] = Conv3x3(16, self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def predict(self, input_features, frame_id):
        outputs = {}

        skip_layer2, skip_layer3, skip_layer4, _, attention_context = input_features

        # decoder
        # B 512 24 80 --> B 128 24 80
        x = self.convs[("upconv", 3, 0)](attention_context)
        x = torch.cat((x, skip_layer4), dim=1)
        x = self.convs[("upconv", 3, 1)](x)

        disp_3 = self.sigmoid(self.convs[("dispconv", 3)](x))
        outputs[("disp", frame_id, 3)] = disp_3

        # B 128 24 80 --> B 64 24 80
        x = self.convs[("upconv", 2, 0)](x)
        x = torch.cat((upsample(x), skip_layer3), dim=1)
        x = self.convs[("upconv", 2, 1)](x)

        disp_2 = self.sigmoid(self.convs[("dispconv", 2)](x))
        outputs[("disp", frame_id, 2)] = disp_2

        # B 64 48 160 --> B 32 48 160
        x = self.convs[("upconv", 1, 0)](x)
        x = torch.cat((upsample(x), skip_layer2), dim=1)
        x = self.convs[("upconv", 1, 1)](x)

        disp_1 = self.sigmoid(self.convs[("dispconv", 1)](x))
        outputs[("disp", frame_id, 1)] = disp_1

        # B 32 96 320 --> B 16 96 320
        x = self.convs[("upconv", 0, 0)](x)
        x = upsample(x)
        x = self.convs[("upconv", 0, 1)](x)

        disp_0 = self.sigmoid(self.convs[("dispconv", 0)](x))
        outputs[("disp", frame_id, 0)] = disp_0

        return outputs

    def forward(self, triplet_features):
        self.outputs = {}
        for frame_id in self.frame_nums:
            features = [feat[:, self.f_to_ch[frame_id], :, :, :] for feat in triplet_features]
            depth = self.predict(features, frame_id)
            self.outputs.update(depth)

        return self.outputs


class Ref_DepthDecoder(nn.Module):
    def __init__(self, min_disp=0.01, max_disp=10, num_bins=256):
        super(Ref_DepthDecoder, self).__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() and not force_cpu else "cpu")

        self.ref_disp = torch.linspace(min_disp, max_disp, num_bins)
        self.ref_disp = self.ref_disp.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute(0, 3, 1, 2)
        self.ref_disp = nn.Parameter(self.ref_disp, requires_grad=False)

        self.conv = ConvBlock(512, 256)
        self.ddv = Conv3x3(256, 256)

        self.softmax = nn.Softmax(dim=1)

    # @staticmethod
    # def get_uncertainty(ddv):
    #     return torch.std(ddv, dim=1, keepdim=True)
    def softmin(self, feature):
        ratio = self.softmax(feature)
        ref_disp = self.ref_disp.expand(ratio.size()[0], -1, ratio.size()[2], ratio.size()[3])
        return torch.sum(ratio * ref_disp, dim=1, keepdim=True)

    def forward(self, x):
        batch_size, num_views, _, h, w = x.shape
        x = x.view(batch_size*num_views, -1, h, w)
        x = self.conv(x)
        x = self.ddv(x)
        ref_disp = self.softmin(x)
        ref_disp = ref_disp.view(batch_size, num_views, 1, h, w)  # B x num_views x 1 x 24 x 80

        return ref_disp


class DepthDecoder_BaseLine(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder_BaseLine, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

