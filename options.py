import os
import argparse

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)

# Implementation is heavily based on option template from MonoDepth2: https://github.com/nianticlabs/monodepth2/blob/master/options.py

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="TempConsisDepth options")

        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="model")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data/"))

        # PATHS
        self.parser.add_argument("--data_path_val",
                                 type=str,
                                 help="path to the validation data",
                                 default=os.path.join(file_dir, "kitti_data/"))

        self.parser.add_argument("--data_path_test",
                                 type=str,
                                 help="path to the testing data",
                                 default=os.path.join(file_dir,"kitti_data/"))

        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(file_dir, "log"))

        # baseline * fx
        self.parser.add_argument("--bf",
                                 type=float,
                                 help="baseline times fx for the disp2depth",
                                 default=1.0)

        # TRAINING options
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "eigen_zhou_naive",
                                          "odom", "benchmark"],
                                 default="eigen_zhou")

        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])

        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti")

        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true",
                                 default=True)

        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)

        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)

        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)

        self.parser.add_argument("--geo_consistency",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0.1)

        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])

        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)

        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)

        # self.parser.add_argument("--step_size",
        #                          type=int,
        #                          help="maximum depth",
        #                          default=128)

        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")

        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=6)

        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)

        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=30)

        self.parser.add_argument("--scheduler_milestones",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=[15, 20])

        self.parser.add_argument("--scheduler_gamma",
                                 type=float,
                                 help="ratio the scheduler",
                                 default=0.25)

        # ABLATION options
        self.parser.add_argument("--use_uncert",
                                 type=bool,
                                 help="use uncertainty",
                                 default=False)

        self.parser.add_argument("--motion_masking_begin",
                                 type=int,
                                 help="when to use motion masking",
                                 default=0)

        self.parser.add_argument("--motion_masking_end",
                                 type=int,
                                 help="when to stop motion masking",
                                 default=19)

        self.parser.add_argument("--use_geo",
                                 type=bool,
                                 help="use geometric loss",
                                 default=True)

        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")

        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")

        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")

        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")

        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")

        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])

        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])

        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")

        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load",
                                 default=None
                                 )

        self.parser.add_argument("--reference_weights_folder",
                                 type=str,
                                 help="name of model for reference net to load",
                                 
                                 )

        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose", "cross_attention","ref_depth"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)

        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")

        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")

        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")

        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)

        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")

        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                     "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "tcm"],
                                 help="which split to run eval on")

        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")

        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")

        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")

        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str,
                                 default='..')

        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
