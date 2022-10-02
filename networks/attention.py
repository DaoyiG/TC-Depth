import torch
import torch.nn as nn
from collections import OrderedDict


class Spatial_Attn(nn.Module):
    """Cross attention Layer using spatial relation"""

    def __init__(self, in_dim, out_dim, radii=0.3):
        super(Spatial_Attn, self).__init__()
        self.chanel_in = in_dim
        self.chanel_out = out_dim
        self.height = 24
        self.width = 80
        self.pix_num = self.height * self.width
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # Learnable parameters
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.7)
        # TODO: decide best radii for Temporal Attn.
        self.sigma_3d = nn.Parameter(torch.ones(1) * radii * 30.0 / 36.0, requires_grad=False)

        self.context_conv = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(512))

        self.ca_conv = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(512))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1024,
                      out_channels=512,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(512))

    def compute_3d_attention(self, distance):
        distance_kernel = torch.exp(-(distance.detach()) / (2 * self.sigma_3d))
        attention_3d = distance_kernel
        return attention_3d

    def forward(self, context_feature, distance):
        """
            inputs :
                mask: binary mask, 0 for invalid depth, 1 for valid depth
                distance : pair-wise euclidean distance of each point (B x num_views x N x N)
                context_feature : input feature maps( B X C1 X W X H) for of context
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        batch_size, num_views, _, height, width = context_feature.size()
        pix_num = height * width

        context_feature = context_feature.view(batch_size * num_views, -1, height, width)
        distance = distance.view(batch_size * num_views, pix_num, pix_num)

        attention = self.compute_3d_attention(distance)

        proj_value = self.value_conv(context_feature).view(batch_size * num_views, -1, pix_num)
        normalizer = torch.sum(attention, dim=1, keepdim=True)
        ca_feature = torch.bmm(proj_value,
                               attention) / normalizer  # attention is a symmetric matrix now, no need for transpose
        ca_feature = ca_feature.view(batch_size*num_views,
                                     self.chanel_out,
                                     height, width).contiguous()

        out = torch.cat((self.ca_conv(ca_feature),
                         self.context_conv(context_feature)), dim=1)

        out = self.conv1(out)
        out = self.gamma * out + context_feature

        out = out.view(batch_size, num_views, -1, height, width)
        attention = attention.view(batch_size, num_views, pix_num, pix_num)[:, 0, :, :]  # B x N x N, only central frame

        return out, attention


class Temp_Attn(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        '''
        The basic implementation for self-attention block/non-local block
        Input:
            B x C x H x W x N
        Parameters:
            in_channels       : the dimension of the input feature map
            key_channels      : the dimension after the key/query transform
            value_channels    : the dimension after the value transform
            scale             : choose the scale to downsample the input feature maps (save memory cost)

        Return:
            B x C x H x W
        '''
        super(Temp_Attn, self).__init__()

        self.scale = scale

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.key_channels = self.in_channels // scale
        self.query_channels = self.in_channels // scale
        self.value_channels = self.in_channels // scale

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.key_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(self.key_channels))

        self.f_query = self.f_key

        self.f_value = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.value_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.f_out = nn.Conv2d(in_channels=self.value_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.fuse_layer = nn.Conv2d(in_channels=out_channels*2,
                                    out_channels=out_channels,
                                    stride=1,
                                    kernel_size=1,
                                    padding=0)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        '''
        Input:
            x: features with shape B x num_view x C x H x W (0, -1, 1)
            pix_coords: list of pix_coords for warping with shape B x H x W x 2 (-1, 1)

        Transform:
            ref_feature: reference frame feature B x C x H x W
            adj_featurs: list of adjacent features with shape B x C x H x W (-1, 1)

        Output:
            B x C x H x W (Reference frame feature)
        '''

        out = []
        attention_maps = []

        batch_size, num_view, _, h, w = x.size()
        assert num_view == 3

        idx_list = [idx for idx in range(num_view)]
        for i in range(num_view):
            list_temp = idx_list.copy()
            ref_id = list_temp.pop(i)
            adj_ids = list_temp

            ref_feature = x[:, ref_id, :, :, :]
            adj_features = [x[:, adj_ids[0], :, :, :],
                            x[:, adj_ids[1], :, :, :]]

            ref_query = self.f_query(ref_feature).view(batch_size, self.query_channels, -1)  # [B, C, N]
            ref_query = ref_query.permute(0, 2, 1)  # [B, N, C]

            matching_context = torch.zeros_like(ref_feature)

            for adj_feature in adj_features:
                adj_value = self.f_value(adj_feature).view(batch_size, self.value_channels, -1)  # [B, C, N]
                adj_value = adj_value.permute(0, 2, 1)  # [B, N, C]

                adj_key = self.f_key(adj_feature).view(batch_size, self.key_channels, -1)  # [B, C, N]

                sim_map = torch.matmul(ref_query, adj_key)  # [B, N, N]
                sim_map = (self.key_channels ** -0.5) * sim_map  # [B, N, N]
                attention = self.softmax(sim_map)  # [B, N, N]

                adj_context = torch.matmul(attention, adj_value)  # [B, N, N] @ [B, N, C] --> [B, N, C]
                adj_context = adj_context.permute(0, 2, 1).contiguous()  # [B, C, N]
                adj_context = adj_context.view(batch_size, self.value_channels, h, w)  # [B, C, H, W]
                adj_context = self.f_out(adj_context)  # [B, C_out, H, W]

                matching_context = matching_context + adj_context

                if i == 0:
                    attention_maps.append(attention.unsqueeze(1))

            fuse = self.fuse_layer(torch.cat((ref_feature, matching_context), dim=1))
            ref_feature = ref_feature + fuse

            out.append(ref_feature.unsqueeze(1))  # [B, 1, C_out, H, W]

        out = torch.cat(out, dim=1)  # [B, num_views, C_out, H, W]
        attention_maps = torch.cat(attention_maps, dim=1)  # [B, 2, N, N] central frame to adjacent frames

        return out, attention_maps


class Spatial_Temp_Attn(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Spatial_Temp_Attn, self).__init__()
        self.spatial_attn = Spatial_Attn(in_channels, out_channels)
        self.temp_attn = Temp_Attn(in_channels, out_channels, scale)

    def forward(self, context_feature, distance):
        x, spatial_attn = self.spatial_attn(context_feature, distance)
        x, temp_attn = self.temp_attn(x)
        return x, spatial_attn, temp_attn


class Spatial_Temp_Module(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, num_view=3):
        super(Spatial_Temp_Module, self).__init__()
        self.num_view = num_view
        self.st_layers = OrderedDict()
        self.st_layers[("spa_temp", 0)] = Spatial_Temp_Attn(in_channels, out_channels, scale)
        self.st_layers[("spa_temp", 1)] = Spatial_Temp_Attn(in_channels, out_channels, scale)
        self.st_layers[("spa_temp", 2)] = Spatial_Temp_Attn(in_channels, out_channels, scale)

        self.module = nn.ModuleList(list(self.st_layers.values()))

    def forward(self, context_feature, distance):
        x, spatial_attn, temp_attn = self.st_layers[("spa_temp", 0)](context_feature, distance)
        x, spatial_attn, _ = self.st_layers[("spa_temp", 1)](x, distance)
        x, spatial_attn, _ = self.st_layers[("spa_temp", 2)](x, distance)
        # x = squeeze_triplet(x)
        # context_feature_skip = squeeze_triplet(context_feature)
        #
        # x = self.fuse_layer(torch.cat((context_feature_skip, x), dim=1))
        #
        # out = unsqueeze_triplet(x, num_view=3) + context_feature
        return x, spatial_attn, temp_attn


def squeeze_triplet(x):
    B, num_view, C, H, W = x.shape
    x = x.view(B*num_view, C, H, W)
    return x


def unsqueeze_triplet(x, num_view):
    B_num_view, C, H, W = x.shape
    x = x.view(B_num_view//num_view, num_view, C, H, W)
    return x