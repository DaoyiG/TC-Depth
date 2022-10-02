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


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
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

def calculate_distance(anchor_feature, mask, no_cuda):
    """Compute distance of anchors in 3d coordinates for cross attention
    anchor_feature: (mask, depth)

    mask: observation mask generated through sparsity invariant CNNs (B x 1 x 24 x 80)
    depth: final output of AnchorAutoencoder (B x 1 x 24 x 80)

     """
    device = torch.device("cpu" if no_cuda else "cuda:0")
    backproject_dense_depth = BackprojectDepth(1, 24, 80).to(device)
    K_latent = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
    K_latent[0, :] *= 80
    K_latent[1, :] *= 24
    K_latent = torch.from_numpy(K_latent)
    K_latent = K_latent.unsqueeze(0)
    inv_K_latent = np.linalg.pinv(K_latent)
    inv_K_latent = torch.from_numpy(inv_K_latent).to(device)
    depth = anchor_feature
    depth = depth * mask
    pix = backproject_dense_depth(depth, inv_K_latent)[:, 0:3, :]
    pix_ba = pix.unsqueeze(2).expand(1, 3, 24 * 80, 24 * 80)
    pix_fr = pix.unsqueeze(3).expand(1, 3, 24 * 80, 24 * 80)
    distance = torch.norm(pix_fr - pix_ba, p=2, dim=1).detach()
    return distance

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    print(gt_path)
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        self_attention_path = os.path.join(opt.load_weights_folder, "self_attention.pth")
        cross_attention_path = os.path.join(opt.load_weights_folder, "cross_attention.pth")
        # ref_encoder_path = os.path.join(opt.reference_weights_folder, "encoder.pth")
        # ref_decoder_path = os.path.join(opt.reference_weights_folder, "depth.pth")

        if opt.no_cuda:
            encoder_dict = torch.load(encoder_path, map_location="cpu")
            # ref_encoder_dict = torch.load(ref_encoder_path, map_location="cpu")
        else:
            encoder_dict = torch.load(encoder_path)
            # ref_encoder_dict = torch.load(ref_encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path_test, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=False, drop_last=False)

        encoder = networks.drn_c_26(False)
        depth_decoder = networks.DepthDecoder()
        # self_attention = networks.Self_Attn(512, 512)
        cross_attention = networks.Cross_Attn(512,512)

        # ref_encoder = networks.drn_c_26(False)
        # ref_depth_decoder = networks.DepthDecoder()

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        #
        # ref_model_dict = ref_encoder.state_dict()
        # ref_encoder.load_state_dict({k: v for k, v in ref_encoder_dict.items() if k in ref_model_dict})

        if opt.no_cuda:
            depth_decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
            # self_attention.load_state_dict(torch.load(self_attention_path, map_location="cpu"))
            cross_attention.load_state_dict(torch.load(cross_attention_path, map_location="cpu"))
            # ref_depth_decoder.load_state_dict(torch.load(ref_decoder_path, map_location="cpu"))


        else:
            depth_decoder.load_state_dict(torch.load(decoder_path))
            # self_attention.load_state_dict(torch.load(self_attention_path))
            cross_attention.load_state_dict(torch.load(cross_attention_path))
            # ref_depth_decoder.load_state_dict(torch.load(ref_decoder_path))

        if opt.no_cuda is False:
            encoder.cuda()
            encoder.eval()
            depth_decoder.cuda()
            depth_decoder.eval()
            # self_attention.cuda()
            # self_attention.eval()
            cross_attention.cuda()
            cross_attention.eval()
            # ref_encoder.cuda()
            # ref_encoder.eval()
            # ref_depth_decoder.cuda()
            # ref_depth_decoder.eval()

        else:
            encoder.eval()
            depth_decoder.eval()
            # self_attention.eval()
            cross_attention.eval()
            # ref_encoder.eval()
            # ref_depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            tbar = tqdm(dataloader)
            for i, data in enumerate(tbar):
            #for data in dataloader:
                if opt.no_cuda is False:
                    input_color = data[("color", 0, 0)].cuda()
                else:
                    input_color = data[("color", 0, 0)]

                # if opt.post_process:
                #     # Post-processed results require each image to have two forward passes
                #     input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                # ref_features = ref_encoder(input_color)
                ref_features = encoder(input_color)
                ref_features.append(ref_features[-1].clone())
                ref_out = depth_decoder(ref_features)
                _, ref_depth = disp_to_depth(ref_out[("disp", 3)], 0.1, 100)

                mask = torch.ones_like(ref_depth)
                distance = calculate_distance(ref_depth, mask, opt.no_cuda)

                features = encoder(input_color)
                # features_spatial, attention_map_spatial = cross_attention(features[-1], mask, distance, ref_depth)
                # # features_context, attention_map_context = self_attention(features_spatial)
                # features.append(features_spatial)
                features.append(features[-1].clone())

                output = depth_decoder(features)

                pred_disp, _ = disp_to_depth(output[("disp", 0)], 0.1, 100)
                #pred_disp = output[("disp", 0)]
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()



    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        #if opt.ext_disp_to_eval:
        #    pred_disp = 1.0/pred_disps[i]
        #else:
        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
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

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
