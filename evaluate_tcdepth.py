# Implementation is based on script from MonoDepth2: https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from layers import disp_to_depth, BackprojectDepth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
splits_dir = os.path.join("splits")


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def prepare_spatial_attn(disp, obs_mask=None):
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
    inv_K_latent = torch.from_numpy(np.linalg.pinv(K_latent)).cuda()

    backproj_depth_spa = BackprojectDepth(3,
                                          height=24,
                                          width=80).cuda()

    batch_size, num_views, _, h, w = disp.shape
    depth = (1. / disp).view(batch_size * num_views, 1, h, w)

    mask = torch.ones_like(depth) if obs_mask is None else obs_mask
    depth = depth * mask
    points = backproj_depth_spa(depth, inv_K_latent)[:, 0:3, :]
    points_ba = points.unsqueeze(2).expand(batch_size * num_views, 3, 24 * 80, 24 * 80)
    points_fr = points.unsqueeze(3).expand(batch_size * num_views, 3, 24 * 80, 24 * 80)
    distance = torch.norm(points_fr - points_ba, p=2, dim=1).detach()  # (Bxn) x N x N

    distance = distance.view(batch_size, num_views, h * w, h * w)  # B x n x N x N

    return distance


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    print("-> Load GT depth from {}...".format(gt_path))

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

    #########################################################
    # Step 0: Specify the weights path
    #########################################################

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    st_path = os.path.join(opt.load_weights_folder, "spa_temp.pth")
    ref_path = os.path.join(opt.load_weights_folder, "ref_depth.pth")

    #########################################################
    # Step 1: Specify the network components
    #########################################################
    if opt.no_cuda:
        encoder_dict = torch.load(encoder_path, map_location="cpu")
    else:
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

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    #########################################################
    # Step 2: Forward pass of the Network
    #########################################################

    with torch.no_grad():
        tbar = tqdm(dataloader)
        for i, data in enumerate(tbar):
            if opt.no_cuda is False:
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
            distance = prepare_spatial_attn(ref_disp)

            # Pass through Spatial-Temporal Module
            fused_feature, spatial_attn, temp_attn = spa_temp(context_feature=features[-1],
                                                              distance=distance)
            features.append(fused_feature)

            # Acquire results for Reference Disparity and Main Depth!
            outputs = depth_decoder(features)

            pred_disp, _ = disp_to_depth(outputs[("disp", 0, 0)], 0.1, 100)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    #########################################################
    # Step 3: Evaluate depth accuracy
    #########################################################

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    #########################################################
    # Step 4: Output final performance
    #########################################################

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())

