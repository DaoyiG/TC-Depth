import torch
# from torch import Tensor
import torch.nn as nn
from layers import *
from .drn import drn_c_26, drn_d_22, drn_d_54, drn_c_58


class DRNEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, pretrained=True):
        super(DRNEncoder, self).__init__()

        self.encoder = drn_c_26(pretrained=pretrained)
        # self.encoder = drn_d_54(pretrained=pretrained)

        self.conv = nn.Sequential(self.encoder.conv1,
                                  self.encoder.bn1,
                                  self.encoder.relu)

        self.degrid = nn.Sequential(self.encoder.layer5,
                                    self.encoder.layer6,
                                    self.encoder.layer7)

    def forward(self, image_triplets):
        '''
        :param image_triplates: [B x views_num x 3 x 192 x 640], order: 0, -1, 1
        :param pix_coords: list of pix_coords for warping with shape B x H x W x 2 (-1, 1)

        '''

        self.features = []
        batch_size, views_num, _, height_img, width_img = image_triplets.shape

        image_triplets = image_triplets.view(batch_size * views_num, -1, height_img, width_img)
        x = (image_triplets - 0.45) / 0.225

        x = self.conv(x)

        x = self.encoder.layer1(x)

        x = self.encoder.layer2(x)
        x_skip = x.view(batch_size, views_num, -1, height_img // 2, width_img // 2)
        self.features.append(x_skip)  # [B, 32, 320, 96]

        x = self.encoder.layer3(x)
        x_skip = x.view(batch_size, views_num, -1, height_img // 4, width_img // 4)
        self.features.append(x_skip)  # [B, 64, 160, 48]

        x = self.encoder.layer4(x)
        x_skip = x.view(batch_size, views_num, -1, height_img // 8, width_img // 8)
        self.features.append(x_skip)  # [B, 128, 80, 24]

        # Degriding layers in DRN-22
        x = self.degrid(x)

        x = self.encoder.layer8(x)  # [B, 512, 80, 24]
        x = x.view(batch_size, views_num, -1, height_img // 8, width_img // 8)
        self.features.append(x)

        return self.features


