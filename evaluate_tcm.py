import argparse
import csv
import os

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

splits_dir = os.path.join("splits")
MIN_DEPTH = 1e-3
MAX_DEPTH = 50


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class DepthWarping(nn.Module):
    """
        warp a depth map from frame_idx_0 to frame_idx_1
    """

    def __init__(self, batch_size, height, width):
        super(DepthWarping, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K, T):
        points3D = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        points3D = depth.view(self.batch_size, 1, -1) * points3D
        points3D = torch.cat([points3D, self.ones], 1)
        points3D_transformed = torch.matmul(T, points3D)
        depth_warped = points3D_transformed[:, 2, :].view(self.batch_size, 1, self.height, self.width)

        return depth_warped


# Get camera intrinsics
def get_intrinsics(calib_path):
    cam2cam = read_calib_file(os.path.join(calib_path, 'calib_cam_to_cam.txt'))
    P_rect = cam2cam['P_rect_02'].reshape(3, 4)
    P_rect[:, 3] = 0
    K = np.vstack((P_rect, np.array([[0, 0, 0, 1]]))).astype(np.float32)
    inv_K = np.linalg.pinv(K)
    K = torch.from_numpy(K)
    inv_K = torch.from_numpy(inv_K)
    return K.unsqueeze(0), inv_K.unsqueeze(0)


# Get observation mask to mask out unobservable depth for error computing
def get_mask(depth_target):
    gt_height, gt_width = depth_target.shape[2:]
    mask = np.logical_and(depth_target > MIN_DEPTH, depth_target < MAX_DEPTH)

    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros((gt_height, gt_width))
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    obs_mask = np.logical_and(mask, crop_mask)
    return obs_mask


# Generate pcd for visualization
def generate_pcd(points, filter_mask):
    points_filtered = points.cpu().squeeze().view(3, -1).numpy()
    points_filtered = points_filtered[:, filter_mask]
    points_filtered = points_filtered.T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered)
    return pcd


def compute_track(depth_warp,
                  backprojection,
                  depth_target,
                  depth_source,
                  flow_tar2src,
                  T_tar2src,
                  inv_K,
                  gt_height,
                  gt_width):
    """
    Computation of track (Euclidean distance in 3D) for depth ground_truth or prediction (depending on input)
    """

    depth_tar2src = depth_warp(depth_target,
                               inv_K,
                               T_tar2src)

    depth_src_flow = F.grid_sample(depth_source,
                                   flow_tar2src,
                                   padding_mode="zeros",
                                   align_corners=False)

    # Get 3D points using back-projection, extract 3D coordinates from the results
    points_tar2src = backprojection(depth_tar2src, inv_K).view(1, 4, gt_height, gt_width)[:, :3, ...]
    points_src_flow = backprojection(depth_src_flow, inv_K).view(1, 4, gt_height, gt_width)[:, :3, ...]
    track_gt = torch.norm(points_tar2src - points_src_flow, p=2, dim=1).squeeze()

    return track_gt, points_tar2src, points_src_flow


def compute_errors(gt_track, pred_track, mask, ratio=0.8):
    """
    Computation of TCM error metrics between predicted and ground-truth tracks
    """

    error = torch.abs(gt_track[mask] - pred_track[mask])
    error_threshold = torch.topk(error.view(-1), int(error.view(-1).shape[0] * ratio),
                                 largest=False, sorted=True).values.max()

    error_filter = (gt_track - pred_track).abs() < error_threshold
    final_mask = error_filter * mask
    abs_rel = torch.abs(gt_track[final_mask] - pred_track[final_mask])
    abs_rel = torch.mean(abs_rel)

    rmse = (gt_track[mask] - pred_track[mask]) ** 2
    rmse = torch.topk(rmse, int(rmse.shape[0] * ratio), largest=False, sorted=True).values
    rmse = np.sqrt(rmse.mean())

    sq_rel = (gt_track[mask] - pred_track[mask]) ** 2
    sq_rel = torch.topk(sq_rel, int(sq_rel.shape[0] * ratio), largest=False, sorted=True).values
    sq_rel = torch.mean(sq_rel)

    return (abs_rel, sq_rel, rmse), error_filter


def parse_args():
    parser = argparse.ArgumentParser(description='Testing options for TCM.')

    parser.add_argument('--results_path', type=str,
                        help='path to the pre-computed results from different models',
                        required=True)

    parser.add_argument('--eval_split', type=str,
                        help='which split to use',
                        default='tcm')

    parser.add_argument('--data_path_test', type=str,
                        help='dataset path')

    parser.add_argument('--sample_rate', type=float,
                        help='how many best point pairs to sample for final error',
                        default=0.8)

    parser.add_argument('--model_name', type=str,
                        help='name of the model you test',
                        default='demo')

    parser.add_argument('--visualize', action="store_true",
                        help='visualize point clouds')

    parser.add_argument('--save_error', action="store_true",
                        help='save error for flickering analysis')

    parser.add_argument("--frame_ids", nargs="+", type=int,
                        help="frames to load",
                        default=[0, -1, 1])

    parser.add_argument("--skip_ids", nargs="+", type=int,
                        help="frames to skip",
                        default=[])

    return parser.parse_args()


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    print("Loading predicted results ...")
    pred_path = os.path.join(opt.results_path)
    pred_depth_snippets = np.load(pred_path, fix_imports=True, encoding='latin1', allow_pickle=True)["depth"]

    print("Loading GT results ...")
    snippet_size = len(opt.frame_ids)
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_temp_components_{}frames.npz".format(snippet_size))
    gt_data = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)
    gt_depths = gt_data["depth"]
    gt_poses = gt_data["pose"]
    gt_flows = gt_data["flow"]

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "tcm_test_files.txt"))
    errors = []
    to_write = []
    to_write_snippet = []

    if opt.save_error:
        for fname in ["errors/{}_error_pair_{}.csv", "errors/{}_error_snippet_{}.csv"]:
            f = fname.format(opt.model_name, int(opt.sample_rate * 100))
            if os.path.isfile(f):
                os.remove(f)
        to_write.append(["center_name", "adj_id", "abs_rel", "sq_rel", "rmse"])
        to_write_snippet.append(["center_name", "abs_rel", "sq_rel", "rmse"])

    points_in_total = 0

    for i in tqdm(range(len(gt_depths))):
        if i not in opt.skip_ids:
            filename = filenames[i]
            head = filename.split(' ')[0].split('/')[0]
            K, inv_K = get_intrinsics(os.path.join(opt.data_path_test, head))

            gt_height, gt_width = gt_depths[i][0].shape[2:]

            backprojection = BackprojectDepth(1, gt_height, gt_width)
            depth_warp = DepthWarping(1, gt_height, gt_width)

            # Load gt and pred target depth
            gt_depth_target = torch.from_numpy(gt_depths[i][0])

            # Get observation mask
            obs_mask = get_mask(gt_depth_target).squeeze() == 1

            pred_depth_target = torch.from_numpy(pred_depth_snippets[i][0])
            pred_depth_target = F.interpolate(pred_depth_target,
                                              (gt_height, gt_width),
                                              mode="bilinear",
                                              align_corners=False)

            # Get scaling factor
            scale_tar = torch.median(gt_depth_target.squeeze()[obs_mask]) / \
                        torch.median(pred_depth_target.squeeze()[obs_mask])

            pred_depth_target *= scale_tar
            pred_depth_target = pred_depth_target.clamp(MIN_DEPTH, MAX_DEPTH)

            # Iterate over all source frames to compute 3D track errors
            snippet_error = np.array([0, 0, 0]).astype(np.float64)

            for frame_id in opt.frame_ids[1:]:
                assert frame_id != 0

                # Load interpolated depth, flow, and pose from source frame
                gt_depth_source_interp = torch.from_numpy(gt_depths[i][frame_id])
                flow_tar2src = torch.from_numpy(gt_flows[i][frame_id])
                T_tar2src = torch.from_numpy(gt_poses[i][frame_id])

                # Warp the depth and calculate 3D flow track of gt
                track_gt, points_tar2src_gt, points_src_flow_gt = compute_track(depth_warp,
                                                                                backprojection,
                                                                                gt_depth_target,
                                                                                gt_depth_source_interp,
                                                                                flow_tar2src,
                                                                                T_tar2src,
                                                                                inv_K,
                                                                                gt_height=gt_height,
                                                                                gt_width=gt_width)

                threshold = 0.3
                obs_mask = torch.logical_and((track_gt < threshold), obs_mask)

                # Get the resized pred_disp and pred_depth for triplets
                pred_depth_source = torch.from_numpy(pred_depth_snippets[i][frame_id])
                pred_depth_source = F.interpolate(pred_depth_source,
                                                  (gt_height, gt_width),
                                                  mode="bilinear",
                                                  align_corners=False)

                # scaling the pred depth from target and source using target scaling factor, and clamp the depth
                pred_depth_source *= scale_tar
                pred_depth_source = pred_depth_source.clamp(MIN_DEPTH, MAX_DEPTH)

                # Warp the depth and calculate 3D flow track of prediction
                track_pred, points_tar2src, points_src_flow = compute_track(depth_warp,
                                                                            backprojection,
                                                                            pred_depth_target,
                                                                            pred_depth_source,
                                                                            flow_tar2src,
                                                                            T_tar2src,
                                                                            inv_K,
                                                                            gt_height=gt_height,
                                                                            gt_width=gt_width)

                # Finally, compute errors of 3D track
                error, error_filter = compute_errors(track_gt, track_pred, obs_mask, ratio=opt.sample_rate)
                snippet_error += np.array(error)
                errors.append(error)
                to_write.append([filename, frame_id, error[0].item(), error[1].item(), error[2].item()])

                if opt.visualize:
                    filter_mask_np = (obs_mask * error_filter).squeeze().view(-1).cpu().numpy()
                    pcd_tar2src = generate_pcd(points_tar2src, filter_mask_np)
                    pcd_tar2src.paint_uniform_color([1, 0.706, 0])  # Yellow

                    pcd_src_flow = generate_pcd(points_src_flow, filter_mask_np)
                    pcd_src_flow.paint_uniform_color([0.46, 0.259, 0.553])  # Purple

                    o3d.visualization.draw_geometries([pcd_src_flow, pcd_tar2src])

                points_in_total += (obs_mask * error_filter).sum()

            snippet_error /= len(opt.frame_ids[1:])
            to_write_snippet.append([filename, snippet_error[0], snippet_error[1], snippet_error[2]])

    if opt.save_error:
        with open("errors/{}_error_pair_{}.csv".format(opt.model_name, int(opt.sample_rate * 100)), "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(to_write)

        with open("errors/{}_error_snippet_{}.csv".format(opt.model_name, int(opt.sample_rate * 100)), "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(to_write_snippet)

    print('There are {} points to be evaluated'.format(points_in_total))
    print("{} pairs of frames to be evaluated in total".format(len(errors)))
    final_error = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 3).format("abs_rel", "sq_rel", "rmse"))
    print(("&{: 8.3f}  " * 3).format(*final_error.tolist()) + "\\\\")


if __name__ == "__main__":
    opt = parse_args()
    evaluate(opt)
