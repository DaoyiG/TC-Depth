# Implementation is heavily based on trainer template from MonoDepth2: https://github.com/nianticlabs/monodepth2/blob/master/trainer.py

import time
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks



class Trainer:
    def __init__(self, options):
        self.opt = options

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.ref_net = {}
        self.models = {}
        self.parameters_to_train = []

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # Use Monodepth2 as motion masking provider
        self.ref_net["mono_encoder"] = networks.ResnetEncoder_BaseLine(18, False)
        self.ref_net["mono_encoder"].to(self.device)
        # self.parameters_to_train += list(self.ref_net["mono_encoder"].parameters())

        self.ref_net["mono_depth"] = networks.DepthDecoder_BaseLine(self.ref_net["mono_encoder"].num_ch_enc)
        self.ref_net["mono_depth"].to(self.device)
        # self.parameters_to_train += list(self.ref_net["mono_depth"].parameters())

        self.monodepth_path = self.opt.reference_weights_folder
        assert self.monodepth_path is not None, f"the reference network should directly load weights"
        # self.load_ref_net_weight(load_epoch=0)

        self.models["encoder"] = networks.DRNEncoder(pretrained=True)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(use_uncert=self.opt.use_uncert)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["ref_depth"] = networks.Ref_DepthDecoder()
        self.models["ref_depth"].to(self.device)
        self.parameters_to_train += list(self.models["ref_depth"].parameters())

        self.models["spa_temp"] = networks.Spatial_Temp_Module(in_channels=512,
                                                               out_channels=512,
                                                               scale=8)
        self.models["spa_temp"].to(self.device)
        self.parameters_to_train += list(self.models["spa_temp"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder_BaseLine(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())


        self.model_optimizer = optim.Adam(self.parameters_to_train,
                                          self.opt.learning_rate)

        # self.model_lr_scheduler = optim.lr_scheduler.StepLR(
        #     self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.model_optimizer,
            milestones=self.opt.scheduler_milestones,
            gamma=self.opt.scheduler_gamma)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png'  # if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path_val, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.depth_warp = {}

        K_latent = np.array([[0.58, 0, 0.5, 0],
                             [0, 1.92, 0.5, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.float32)
        K_latent[0, :] *= 80
        K_latent[1, :] *= 24
        K_latent = torch.from_numpy(K_latent)
        K_latent = K_latent.unsqueeze(0)
        self.inv_K_latent = np.linalg.pinv(K_latent)

        self.inv_K_latent = torch.from_numpy(self.inv_K_latent).to(self.device)
        self.K_latent = K_latent.to(self.device)

        # for scale in self.opt.scales:
        source_scale = 0
        h = self.opt.height // (2 ** source_scale)
        w = self.opt.width // (2 ** source_scale)

        self.backproject_depth[source_scale] = BackprojectDepth(self.opt.batch_size, h, w)
        self.backproject_depth[source_scale].to(self.device)

        self.project_3d[source_scale] = Project3D(self.opt.batch_size, h, w)
        self.project_3d[source_scale].to(self.device)

        self.depth_warp[source_scale] = DepthWarping(self.opt.batch_size, h, w)
        self.depth_warp[source_scale].to(self.device)

        self.backproj_depth_spa = BackprojectDepth(self.opt.batch_size * self.num_input_frames,
                                                   height=24,
                                                   width=80).to(self.device)
        self.f_to_ch = {0: 0,
                        -1: 1,
                        1: 2}

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

        # print all json options before training
        filename = os.path.join(self.opt.log_dir, self.opt.model_name, "models/opt.json")
        with open(filename, 'r') as handle:
            parsed = json.load(handle)
            print(json.dumps(parsed, indent=4))

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(0, self.opt.num_epochs):
            if self.opt.motion_masking_begin <= self.epoch <= self.opt.motion_masking_end:
                self.load_ref_net_weight()

            self.run_epoch()
            if self.epoch % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    def prepare_spatial_attn(self, disp, obs_mask=None):
        """Prepare spatial distance for spatial attention
        """
        batch_size, num_views, _, h, w = disp.shape
        depth = (1. / disp).view(batch_size * num_views, 1, h, w)

        mask = torch.ones_like(depth) if obs_mask is None else obs_mask
        depth = depth * mask
        points = self.backproj_depth_spa(depth, self.inv_K_latent)[:, 0:3, :]
        points_ba = points.unsqueeze(2).expand(batch_size * num_views, 3, 24 * 80, 24 * 80)
        points_fr = points.unsqueeze(3).expand(batch_size * num_views, 3, 24 * 80, 24 * 80)
        distance = torch.norm(points_fr - points_ba, p=2, dim=1).detach()  # (Bxn) x N x N

        distance = distance.view(batch_size, num_views, h * w, h * w)  # B x n x N x N

        return distance

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # Otherwise, we only feed the image with frame_id 0 through the depth encoder

        # Generate image triplets for encoder
        image_triplets = torch.cat([inputs["color_aug", 0, 0].unsqueeze(1),
                                    inputs["color_aug", -1, 0].unsqueeze(1),
                                    inputs["color_aug", 1, 0].unsqueeze(1)], dim=1)
        features = self.models["encoder"](image_triplets)

        # Acquire relative distance for spatial attention
        ref_disp = self.models["ref_depth"](features[-1].detach())
        distance = self.prepare_spatial_attn(ref_disp)

        # Pass through Spatial-Temporal Module
        fused_feature, spatial_attn, temp_attn = self.models["spa_temp"](context_feature=features[-1],
                                                                         distance=distance)
        features.append(fused_feature)

        # Acquire results for Reference Disparity and Main Depth!
        outputs = self.models["depth"](features)
        for frame_id in self.opt.frame_ids:
            outputs[("ref_disp", frame_id, 3)] = ref_disp[:, self.f_to_ch[frame_id], :, :, :]

        # Log Attention Maps
        # spatial attention for central frame
        outputs[("spatial_attn", 0)] = spatial_attn.unsqueeze(1)  # B x 1 x N x N
        # temporal attention between central frame and past frame
        outputs[("temp_attn", -1)] = temp_attn[:, 0, :, :].unsqueeze(1)  # B x 1 x N x N
        # temporal attention between central frame and future frame
        outputs[("temp_attn", 1)] = temp_attn[:, 1, :, :].unsqueeze(1)  # B x 1 x N x N

        # Generate Dynamic Teacher Prediction
        if self.epoch >= self.opt.motion_masking_begin:
            with torch.no_grad():
                mono_features = self.ref_net["mono_encoder"](inputs["color_aug", 0, 0])
                mono_out = self.ref_net["mono_depth"](mono_features)
                for s in self.opt.scales:
                    outputs[("mono_disp", 0, s)] = mono_out[("disp", s)]

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                elif self.opt.pose_model_type == "posecnn":
                    pose_inputs = torch.cat(pose_inputs, 1)

                axisangle, translation = self.models["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

                outputs[("cam_T_cam", f_i, 0)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i > 0))
        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        source_scale = 0

        for scale in self.opt.scales:
            disp_ref = outputs[("disp", 0, scale)]

            disp_ref = F.interpolate(
                disp_ref, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            _, depth_ref = disp_to_depth(disp_ref, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth_ref

            if self.opt.motion_masking_begin <= self.epoch <= self.opt.motion_masking_end:
                disp_md = outputs[("mono_disp", 0, scale)]
                disp_md = F.interpolate(
                    disp_md, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                _, depth_ref_md = disp_to_depth(disp_md, self.opt.min_depth, self.opt.max_depth)
                outputs[("mono_depth", 0, scale)] = depth_ref_md

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                disp_adj = outputs[("disp", frame_id, scale)]
                disp_adj = F.interpolate(
                    disp_adj, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                _, depth_adj = disp_to_depth(disp_adj, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", frame_id, scale)] = depth_adj

                if frame_id == "s":
                    T_ref2adj = inputs["stereo_T"]
                    T_adj2ref = -T_ref2adj
                else:
                    T_ref2adj = outputs[("cam_T_cam", 0, frame_id)]
                    T_adj2ref = outputs[("cam_T_cam", frame_id, 0)]

                # Warp adjacent image to reference frame
                cam_points_ref = self.backproject_depth[source_scale](
                    depth_ref, inputs[("inv_K", source_scale)])
                pix_coords_ref = self.project_3d[source_scale](
                    cam_points_ref, inputs[("K", source_scale)], T_ref2adj)

                outputs[("sample", frame_id, scale)] = pix_coords_ref

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords_ref,
                    padding_mode="border", align_corners=True)

                if self.opt.use_geo:
                    # warp depth from central frame to adjacent frame
                    outputs[("depth_warped", frame_id, scale)] = self.depth_warp[source_scale](
                        outputs[("depth", 0, scale)],
                        inputs[("inv_K", source_scale)], T_ref2adj).clamp(self.opt.min_depth,
                                                                          self.opt.max_depth)

                    outputs[("depth_interpolated", frame_id, scale)] = F.grid_sample(
                        outputs[("depth", frame_id, scale)],
                        pix_coords_ref,
                        padding_mode="zeros", align_corners=False).clamp(self.opt.min_depth,
                                                                         self.opt.max_depth)

                    # Warp the warped image from adjacent to reference frame back to adjacent frame
                    # for cycle-consistency loss
                    cam_points_adj = self.backproject_depth[source_scale](
                        depth_adj, inputs[("inv_K", source_scale)])
                    pix_coords_adj = self.project_3d[source_scale](
                        cam_points_adj, inputs[("K", source_scale)], T_adj2ref)

                    # Warp image from reference frame to adjacent frame
                    # border padding
                    color_ref2adj = F.grid_sample(
                        inputs[("color", 0, source_scale)],
                        pix_coords_adj,
                        padding_mode="border", align_corners=True)

                    # Warp warped image back from adjacent frame to reference frame
                    # zeros padding
                    outputs[("color_back", frame_id, scale)] = F.grid_sample(
                        color_ref2adj,
                        pix_coords_ref,
                        padding_mode="zeros", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_geometric_loss(self, depth_warped, depth_interp):

        assert depth_warped.dim() == depth_interp.dim()
        depth_pair = torch.cat((depth_warped, depth_interp), dim=1)
        min_depth, _ = torch.min(depth_pair, dim=1, keepdim=True)
        max_depth, _ = torch.max(depth_pair, dim=1, keepdim=True)
        abs_diff = 1 - min_depth / max_depth

        geometric_loss = abs_diff
        return geometric_loss

    def compute_reference_loss(self, reference_disp, main_disp):
        """Computes difference between the disparity output from reference net and the main branch
        """

        upsampled_reference_disp = F.interpolate(reference_disp,
                                                 [self.opt.height, self.opt.width],
                                                 mode="bilinear",
                                                 align_corners=False)

        target_disp, _ = disp_to_depth(main_disp, self.opt.min_depth, self.opt.max_depth)
        abs_diff = (target_disp - upsampled_reference_disp).abs()
        reference_loss = abs_diff.mean()

        return reference_loss

    @staticmethod
    def compute_motion_masks(teacher_depth, student_depth):
        """
        Generate a mask of where we cannot trust the main pathway of the network, based on the difference
        between the main pathway and the reference monodepth2
        """

        # mask where they differ by a large amount
        mask = ((student_depth - teacher_depth) / teacher_depth) < 0.6
        mask *= ((teacher_depth - student_depth) / student_depth) < 0.6
        return mask

    @staticmethod
    def compute_auto_masks(reprojection_loss, identity_reprojection_loss):
        """
        Compute loss masks for each of standard reprojection and depth hint
        reprojection
        """
        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float().detach()

        return reprojection_loss_mask

    @staticmethod
    def compute_cycle_masks(cycle_weight, ratio=0.3):
        mask = 1 - cycle_weight
        threshold = torch.quantile(mask, q=ratio)
        mask[mask > threshold] = 1.
        mask[mask < threshold] = 0.
        return mask

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            source_scale = 0

            loss = 0
            geometric_loss = 0

            reprojection_losses = []
            identity_reprojection_losses = []
            geometric_losses = []

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                if self.opt.use_geo:
                    pred_cycle = outputs[("color_back", frame_id, scale)]
                    outputs[("cycle_weighting", frame_id, scale)] = self.compute_reprojection_loss(pred_cycle,
                                                                                                   target).detach()
                    outputs[("cycle_mask", frame_id, scale)] = self.compute_cycle_masks(outputs[("cycle_weighting",
                                                                                                frame_id,
                                                                                                scale)])
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                    depth_warped = outputs[("depth_warped", frame_id, scale)]
                    depth_interp = outputs[("depth_interpolated", frame_id, scale)]
                    geometric_loss_adj = self.compute_geometric_loss(depth_warped,
                                                                     depth_interp)
                    geometric_loss_adj = geometric_loss_adj * (outputs[("cycle_mask", frame_id, scale)].detach())
                    geometric_losses.append(geometric_loss_adj)

            # Compute reprojection losses, and its minimum
            reprojection_losses = torch.cat(reprojection_losses, 1)
            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

            # Compute identity reprojection losses, and its minimum
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            # Auto mask: 1 stands for static pixel, 0 stands for dynamic pixel
            auto_mask = self.compute_auto_masks(reprojection_loss,
                                                identity_reprojection_loss)
            outputs["identity_selection/{}".format(scale)] = auto_mask

            # Compute photometric loss, add it to loss of current scale
            final_reprojection_loss = (reprojection_loss * auto_mask).sum() / \
                                      (auto_mask.sum() + 1e-7)
            loss += final_reprojection_loss

            if self.opt.motion_masking_begin <= self.epoch <= self.opt.motion_masking_end:
                # no gradients for mono prediction!
                mono_depth = outputs[("mono_depth", 0, source_scale)].detach()
                main_depth = outputs[("depth", 0, scale)]

                # motion mask: 1 stands for static pixel, 0 stands for dynamic pixel
                outputs[('motion_masking', 0, scale)] = self.compute_motion_masks(teacher_depth=mono_depth,
                                                                                  student_depth=main_depth)
                auto_mask = auto_mask * outputs[('motion_masking', 0, scale)]

                # inconsistency_mask will be 1 where the pred_depth differs a lot from pred_depth_md, indicating this
                # is a dynamic pix, requires consistency loss
                # inconsistency_mask will be 0 where the pred_depth is similar to pred_depth_md
                inconsistency_mask = 1 - outputs[('motion_masking', 0, scale)].float()
                # inconsistency_mask = 1 - auto_mask

                consistency_loss = torch.abs(1 / main_depth - 1 / mono_depth) * inconsistency_mask
                # consistency_loss = consistency_loss.sum() / (inconsistency_mask.sum() + 1e-5)
                consistency_loss = consistency_loss.mean()
                losses['consistency_loss/{}'.format(scale)] = consistency_loss
                loss += consistency_loss

            if self.opt.use_geo:
                # Compute geometric loss, add it to loss of current scale
                geometric_losses = torch.cat(geometric_losses, 1)
                # geometric_loss, _ = torch.min(geometric_losses, dim=1, keepdim=True)
                geometric_loss = torch.mean(geometric_losses, dim=1, keepdim=True)
                final_geometric_loss = self.opt.geo_consistency * (geometric_loss * auto_mask).mean()
                loss += final_geometric_loss
                losses["individual_loss_scale_{}/{}".format(scale, "geometric")] = final_geometric_loss

            # Compute smoothness loss, add it to loss of current scale
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            final_smooth_loss = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            loss += final_smooth_loss

            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            losses["individual_loss_scale_{}/{}".format(scale, "photometric")] = final_reprojection_loss
            losses["individual_loss_scale_{}/{}".format(scale, "edge_smoothness")] = final_smooth_loss

        total_loss /= self.num_scales

        # compute the loss for reference net
        reference_loss = 0
        for i in self.opt.frame_ids:
            reference_loss += self.compute_reference_loss(reference_disp=outputs[("ref_disp", i, 3)],
                                                          main_disp=outputs[("disp", i, 0)].detach())
        reference_loss /= self.num_input_frames
        total_loss += reference_loss
        losses["reference_loss/{}".format("reference_loss")] = reference_loss
        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for frame_id in self.opt.frame_ids:
                if frame_id == 0:
                    # Draw spatial attention of central frame
                    spa_attn_draw_large = draw_attention_stacked(outputs[("spatial_attn", 0)][j].detach())
                    writer.add_image("spatial_attention/{}".format(j),
                                     spa_attn_draw_large, self.step)

                # Draw depth from adjacent views, only scale 0 is enough
                if frame_id != 0:
                    writer.add_image(
                        "disp_frame_{}_s_{}/{}".format(frame_id, 0, j),
                        normalize_image(outputs[("disp", frame_id, 0)][j]), self.step)

                    # Draw temporal attention of central frame
                    temp_attn_draw_large = draw_attention_stacked(outputs[("temp_attn", frame_id)][j].detach())
                    writer.add_image("temporal_attention_frame_0_to_{}/{}".format(frame_id, j),
                                     temp_attn_draw_large, self.step)

                # Draw reference disparity from RefDepth Decoder
                writer.add_image("ref_disp_frame_{}/{}".format(frame_id, j),
                                 normalize_image(outputs[("ref_disp", frame_id, 3)][j]), self.step)

                for s in self.opt.scales:
                    # Draw disparity of center frame of all scales
                    if frame_id == 0 and s == 0 and self.epoch >= self.opt.motion_masking_begin:
                        if self.epoch <= self.opt.motion_masking_end:
                            writer.add_image(
                                "motion_masking_{}_scale_{}/{}".format(frame_id, s, j),
                                outputs[("motion_masking", 0, 0)][j], self.step)

                        writer.add_image(
                            "mono_disp_frame_{}_s_{}/{}".format(frame_id, s, j),
                            normalize_image(outputs[("mono_disp", frame_id, s)][j]), self.step)

                    if frame_id == 0:
                        writer.add_image(
                            "disp_frame_{}_s_{}/{}".format(frame_id, s, j),
                            normalize_image(outputs[("disp", frame_id, s)][j]), self.step)

                    if s == 0:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)

                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                    if s == 0 and frame_id == 0:
                        writer.add_image(
                            "automask_{}/{}".format(s, j),
                            outputs["identity_selection/{}".format(s)][j], self.step)

                    if frame_id != 0 and self.opt.use_geo:
                        writer.add_image(
                            "cycle_from_frame_{}_scale_{}/{}".format(frame_id, s, j),
                            outputs[("color_back", frame_id, s)][j].data, self.step)

                        writer.add_image(
                            "cycle_mask_frame_{}_scale_{}/{}".format(frame_id, s, j),
                            normalize_image(outputs[("cycle_mask", frame_id, s)][j]), self.step)

                        # watch over the geo loss ingradients
                        writer.add_image(
                            "disp_interp_frame_{}_scale_{}/{}".format(frame_id, s, j),
                            normalize_image(1.0 / outputs[("depth_interpolated", frame_id, s)][j]), self.step)

                        writer.add_image(
                            "disp_warped_frame_{}_scale_{}/{}".format(frame_id, s, j),
                            normalize_image(1.0 / outputs[("depth_warped", frame_id, s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            if n is "ref_depth":
                model_dict = self.ref_net[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.ref_net[n].load_state_dict(model_dict)
            else:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def load_ref_net_weight(self):
        if self.epoch <= 19:
            self.monodepth_path = os.path.join(self.monodepth_path, "weights_{}".format(self.epoch))
        else:
            self.monodepth_path = os.path.join(self.monodepth_path, "weights_19")

        self.monodepth_path = os.path.expanduser(self.monodepth_path)

        assert os.path.isdir(self.monodepth_path), \
            "Cannot find folder {}".format(self.opt.reference_weights_folder)

        print("loading weights from folder {} for reference monodepth2 for epoch {}".format(self.monodepth_path, self.epoch))

        for n in ["encoder", "depth"]:
            print("Loading {} weights for reference monodepth2".format(n))
            path = os.path.join(self.monodepth_path, "{}.pth".format(n))
            model_dict = self.ref_net["mono_" + n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.ref_net["mono_" + n].load_state_dict(model_dict)
