import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
from kitti_utils import *
from layers import *
from utils import readlines
import networks
import datasets
from options import MonodepthOptions
import os

splits_dir = os.path.join("splits")
MIN_DEPTH = 1e-3
MAX_DEPTH = 80


def prepare_spatial_attn(backprojecter, disp, obs_mask=None, require_points=False):
    """Compute distance of anchors in 3d coordinates for cross attention
    anchor_feature: (mask, depth)

    mask: observation mask generated through sparsity invariant CNNs (B x 1 x 24 x 80)
    depth: final output of AnchorAutoencoder (B x 1 x 24 x 80)

     """
    K_latent = np.array([[0.58, 0, 0.5, 0],
                         [0, 1.92, 0.5, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)
    K_latent[0, :] *= 80
    K_latent[1, :] *= 24
    K_latent = torch.from_numpy(K_latent)
    K_latent = K_latent.unsqueeze(0)
    inv_K_latent = np.linalg.pinv(K_latent)
    inv_K_latent = torch.from_numpy(inv_K_latent).cuda()

    batch_size, num_views, _, h, w = disp.shape
    depth = (1. / disp).view(batch_size * num_views, 1, h, w)

    mask = torch.ones_like(depth) if obs_mask is None else obs_mask
    depth = depth * mask
    points = backprojecter(depth, inv_K_latent)[:, 0:3, :]
    points_ba = points.unsqueeze(2).expand(1 * 3, 3, 24 * 80, 24 * 80)
    points_fr = points.unsqueeze(3).expand(1 * 3, 3, 24 * 80, 24 * 80)
    distance = torch.norm(points_fr - points_ba, p=2, dim=1).detach()
    if not require_points:
        return distance
    return distance, points

def predict(opt):
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "tcm_test_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    st_path = os.path.join(opt.load_weights_folder, "spa_temp.pth")
    ref_path = os.path.join(opt.load_weights_folder, "ref_depth.pth")


    encoder_dict = torch.load(encoder_path)
    dataset = datasets.KITTIRAWDataset(opt.data_path_test, filenames,
                                       encoder_dict['height'], encoder_dict['width'],
                                       opt.frame_ids, 4, is_train=False)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    encoder = networks.DRNEncoder(pretrained=False)
    depth_decoder = networks.DepthDecoder()
    spa_temp = networks.Spatial_Temp_Module(in_channels=512, out_channels=512, scale=8)
    ref_depth = networks.Ref_DepthDecoder()

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))
    ref_depth.load_state_dict(torch.load(ref_path))
    spa_temp.load_state_dict(torch.load(st_path))

    if opt.no_cuda is False:
        encoder.cuda()
        encoder.eval()

        depth_decoder.cuda()
        depth_decoder.eval()

        ref_depth.cuda()
        ref_depth.eval()

        spa_temp.cuda()
        spa_temp.eval()

    else:
        encoder.eval()
        depth_decoder.eval()
        ref_depth.eval()
        spa_temp.eval()

    backproject_dense_depth = BackprojectDepth(3, 24, 80).cuda()

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    pred_depth_snippets = []

    with torch.no_grad():
        tbar = tqdm(dataloader)
        for i, data in enumerate(tbar):
            # pred_disp_triplet = {}
            pred_depth_snippet = {}
            input_color_ref = data[("color", 0, 0)].cuda()
            input_color_last = data[("color", -1, 0)].cuda()
            input_color_next = data[("color", 1, 0)].cuda()

            image_triplets = [input_color_ref.unsqueeze(1),
                              input_color_last.unsqueeze(1),
                              input_color_next.unsqueeze(1)]
            image_triplets = torch.cat(image_triplets, dim=1)

            features = encoder(image_triplets)

            # Acquire relative distance for spatial attention
            ref_disp = ref_depth(features[-1].detach())
            distance = prepare_spatial_attn(backproject_dense_depth, ref_disp)

            # Pass through Spatial-Temporal Module
            fused_feature, spatial_attn, temp_attn = spa_temp(context_feature=features[-1],
                                                              distance=distance)
            features.append(fused_feature)

            # Acquire results for Reference Disparity and Main Depth!
            outputs = depth_decoder(features)

            for frame_id in opt.frame_ids:
                pred_disp, pred_depth = disp_to_depth(outputs[("disp", frame_id, 0)], 0.1, 100)
                pred_depth_snippet[frame_id] = pred_depth.cpu().numpy()
            pred_depth_snippets.append(pred_depth_snippet)

    print(len(pred_depth_snippets))
    output_path = os.path.join("pred_depth_snippets_212_debug.npz")
    print("Saving to {} ...".format(output_path))
    np.savez_compressed(output_path, depth=np.array(pred_depth_snippets))


if __name__ == "__main__":
    options = MonodepthOptions()
    predict(options.parse())