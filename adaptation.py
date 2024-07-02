import os
import copy
import time
import math
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage import transform
import matplotlib.cm
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter
import datasets
import networks
from layers import *
from utils import *
from options import MonodepthOptions
from tqdm import tqdm
import pdb

options = MonodepthOptions()
opts = options.parse()


class Adapt:
    def __init__(self, options):
        self.opt = options
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.h_KITTI = 1.65
        self.h_dgp = 1.63
        self.h_waymo = 2.12

        self.input_height, self.input_width = None, None
        if self.opt.dataset == "dgp":
            self.opt.data_path = "/BS/contact-human-pose2/static00/ddad_train_val/ddad.json"
            # self.gt_height, self.gt_width = 1920, 1152
            self.ratio = self.h_dgp / self.h_KITTI
            self.cam_h = self.h_dgp
        elif self.opt.dataset == "waymo":
            self.opt.data_path = '/BS/databases18/waymo/perception_v_1_2_0/validation/'
            self.opt.depth_path = '/BS/contact-human-pose2/static00/waymo'
            # self.gt_height, self.gt_width = 1920, 1248
            self.ratio = self.h_waymo / self.h_KITTI
            self.cam_h = self.h_waymo
        else:
            # KITTI
            # self.gt_height, self.gt_width = 352, 1216
            self.input_height, self.input_width = 352, 1216
            self.ratio = 1
            self.cam_h = self.h_KITTI

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # create a new log for every save
        self.writer = SummaryWriter(self.log_path)
        self.depth_metric_names = ["sup/abs_rel", "sup/sq_rel", "sup/rms", "sup/log_rms",
                                   "sup/a1", "sup/a2", "sup/a3", "median/global"]
        self.depth_metric_names_local = ["supl/abs_rel", "supl/sq_rel", "supl/rms", "supl/log_rms",
                                         "supl/a1", "supl/a2", "supl/a3", "median/local"]
        self.depth_metric_names_unsup = ["unsupl/abs_rel", "unsupl/sq_rel", "unsupl/rms",
                                               "unsupl/log_rms", "unsupl/a1", "unsupl/a2", "unsupl/a3",
                                               "median/unsup"]

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(self.device)

        self.ssim = SSIM()
        self.ssim.to(self.device)

        # models to train
        self.models = {}
        self.parameters_to_train = []
        self.models["encoder"] = networks.trans_backbone(
            'large07', './models/swin_large_patch4_window7_224_22k.pth')
        self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
        self.models["depth"] = networks.NewCRFDepth(version = 'large07', max_depth = self.MAX_DEPTH)
        self.models["encoder"].to(self.device)
        self.models["depth"] = torch.nn.DataParallel(self.models["depth"])
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["depth"].parameters())

        load_model_list = ['encoder', 'depth']

        if self.opt.load_weights_folder is not None:
            self.load_model(self.opt.load_weights_folder, self.models, load_model_list)

        # ema models for regularisation of global scale
        self.models_ema = copy.deepcopy(self.models)
        for m in self.models_ema.values():
            m.eval()

        # reference models for improvements calculation
        self.models_ref = copy.deepcopy(self.models)
        self.models_ref_state = {}
        for key, value in self.models_ref.items():
            self.models_ref_state[key] = copy.deepcopy(self.models[key].state_dict())
            value.eval()

        # models for regularisation, self-supervised
        print("loading regularisation model from " + self.opt.reg_path)
        # create the reg model
        self.reg_models = {}
        self.reg_parameters_to_train = []
        self.reg_models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.reg_models["encoder"] = torch.nn.DataParallel(self.reg_models["encoder"])
        self.reg_models["encoder"].to(self.device)
        self.reg_models["depth"] = networks.DepthDecoder(self.reg_models["encoder"].module.num_ch_enc, self.opt.scales)
        self.reg_models["depth"] = torch.nn.DataParallel(self.reg_models["depth"])
        self.reg_models["depth"].to(self.device)
        self.reg_models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.reg_models["pose_encoder"] = torch.nn.DataParallel(self.reg_models["pose_encoder"])
        self.reg_models["pose_encoder"].to(self.device)
        self.reg_models["pose"] = networks.PoseDecoder(
            self.reg_models["pose_encoder"].module.num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.reg_models["pose"] = torch.nn.DataParallel(self.reg_models["pose"])
        self.reg_models["pose"].to(self.device)

        self.reg_parameters_to_train += list(self.reg_models["encoder"].parameters())
        self.reg_parameters_to_train += list(self.reg_models["depth"].parameters())
        self.reg_parameters_to_train += list(self.reg_models["pose_encoder"].parameters())
        self.reg_parameters_to_train += list(self.reg_models["pose"].parameters())

        reg_model_folder = os.path.join(self.opt.reg_path)
        self.load_model(reg_model_folder, self.reg_models, ['encoder', 'depth', 'pose_encoder', 'pose'])

        self.reg_models_ref = copy.deepcopy(self.reg_models)
        for m in self.reg_models_ref.values():
            m.eval()

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.reg_model_optimizer = optim.Adam(self.reg_parameters_to_train, self.opt.learning_rate)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # dataloader, test set
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "kitti_depth": datasets.KITTIDepthDataset,
                         "dgp": datasets.DGPDataset,
                         "waymo": datasets.WaymoDataset}
        self.dataset_instance = datasets_dict[self.opt.dataset]
        self.seq_num = 1

        # waymo time of day and weathers
        daytimes = ['Day', 'Dawn/Dusk', 'Night']
        weathers = ['sunny', 'rain']

        if self.opt.dataset == "dgp":
            filenames = "val"
        elif self.opt.dataset == "waymo":
            splits_dir = './splits/waymo_weather'
            filenames = []
            # read weather dics from npy
            split_path = os.path.join(splits_dir, "val.npy")
            weather_dic = np.load(split_path, allow_pickle=True).item()
            for daytime in daytimes:
                if daytime in weather_dic.keys():
                    for weather in weathers:
                        if weather in weather_dic[daytime].keys():
                            filenames.extend(weather_dic[daytime][weather][0:1])
            # filenames = filenames[0:5]
            self.seq_num = len(filenames)
        else:
            splits_dir = './splits/'
            filenames = readlines(os.path.join(splits_dir, self.opt.eval_split, "val_files_bak.txt"))
        self.filenames = filenames

    def run_adapt(self):
        self.start_time = time.time()
        self.step = 0

        errors_tt, errors_local_tt, errors_ref_tt = [], [], []
        errors_t, errors_local_t, errors_unsup_t = [], [], []
        errors_ref_t, errors_local_ref_t, errors_unsup_ref_t, errors_teacher_t, errors_teacher_local_t = [], [], [], [], []

        for i in range(self.seq_num):
            print("seq {}".format(i))
            if self.opt.dataset == "waymo":
                filenames = [self.filenames[i]]
            else:
                filenames = self.filenames
            dataset = self.dataset_instance(self.opt.data_path, filenames,
                                            self.input_height, self.input_width,
                                            [0, -1, 1], self.num_scales,
                                            is_train=True, rotate_aug=False, pseu=True,
                                            depth_path=self.opt.depth_path)
            self.dataloader = DataLoader(dataset, 1, shuffle=False,
                                         num_workers=self.opt.num_workers,
                                         pin_memory=True, drop_last=False)

            pbar = tqdm(total = len(self.dataloader))
            errors, errors_local, errors_unsup = [], [], []
            errors_ref, errors_local_ref, errors_unsup_ref, errors_teacher, errors_teacher_local = [], [], [], [], []

            for batch_idx, inputs in enumerate(self.dataloader):
                self.height = inputs["height_sec"].to(self.device)
                self.width = inputs["width_sec"].to(self.device)
                self.scale_recovery = ScaleRecovery(1, self.height, self.width).to(self.device)

                # checking height and width are multiples of 32
                assert self.height % 32 == 0, "'height' must be a multiple of 32"
                assert self.width % 32 == 0, "'width' mustbe a multiple of 32"

                self.backproject_depth = {}
                self.project_3d = {}
                for scale in self.opt.scales:
                    h = self.height // (2 ** scale)
                    w = self.width // (2 ** scale)

                    self.backproject_depth[scale] = BackprojectDepth(1, h, w)
                    self.backproject_depth[scale].to(self.device)

                    self.project_3d[scale] = Project3D(1, h, w)
                    self.project_3d[scale].to(self.device)

                self.step += 1
                before_op_time = time.time()

                error, error_local, error_unsup, outputs, losses,\
                error_ref, error_local_ref, error_unsup_ref, error_teacher,\
                error_teacher_local = self.process_batch(inputs)

                errors.append(error)
                errors_teacher.append(error_teacher)
                errors_local.append(error_local)
                errors_teacher_local.append(error_teacher_local)
                errors_unsup.append(error_unsup)
                errors_ref.append(error_ref)
                errors_local_ref.append(error_local_ref)
                errors_unsup_ref.append(error_unsup_ref)

                errors_tt.append(error)
                errors_local_tt.append(error_local)
                errors_ref_tt.append(error_ref)

                if batch_idx % 100 == 0:
                    mean_errors_100 = np.array(errors_tt).mean(0)
                    mean_errors_local_100 = np.array(errors_local_tt).mean(0)
                    mean_errors_unsup_100 = np.array(errors_ref_tt).mean(0)

                    mean_errors_dict = {}
                    for i, metric in enumerate(self.depth_metric_names):
                        mean_errors_dict[metric] = mean_errors_100[i]
                    for i, metric in enumerate(self.depth_metric_names_local):
                        mean_errors_dict[metric] = mean_errors_local_100[i]
                    for i, metric in enumerate(self.depth_metric_names_unsup):
                        mean_errors_dict[metric] = mean_errors_unsup_100[i]
                    self.log(inputs, outputs, mean_errors_dict)

                pbar.update(1)
                pbar.set_description("abs_rel diff: {:.4f}, abs_rel ref: {:.4f}".format(error[0]-error_ref[0], error_ref[0]))

            mean_errors = np.array(errors).mean(0)
            mean_errors_teacher = np.array(errors_teacher).mean(0)
            mean_errors_local = np.array(errors_local).mean(0)
            mean_errors_teacher_local = np.array(errors_teacher_local).mean(0)
            mean_errors_unsup = np.array(errors_unsup).mean(0)
            mean_errors_ref = np.array(errors_ref).mean(0)
            mean_errors_local_ref = np.array(errors_local_ref).mean(0)
            mean_errors_unsup_ref = np.array(errors_unsup_ref).mean(0)
            pbar.close()

            print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "median"))
            print("supervised w/o gt median scaling (inital/teacher/student)")
            print(("&{: 8.3f}  " * 8).format(*mean_errors_ref.tolist()) + "\\\\")
            print(("&{: 8.3f}  " * 8).format(*mean_errors_teacher.tolist()) + "\\\\")
            print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
            print("supervised w gt median scaling (inital/teacher/student)")
            print(("&{: 8.3f}  " * 8).format(*mean_errors_local_ref.tolist()) + "\\\\")
            print(("&{: 8.3f}  " * 8).format(*mean_errors_teacher_local.tolist()) + "\\\\")
            print(("&{: 8.3f}  " * 8).format(*mean_errors_local.tolist()) + "\\\\")
            print("self-supervised w gt median scaling (inital/adapted)")
            print(("&{: 8.3f}  " * 8).format(*mean_errors_unsup_ref.tolist()) + "\\\\")
            print(("&{: 8.3f}  " * 8).format(*mean_errors_unsup.tolist()) + "\\\\")

            errors_t.append(mean_errors)
            errors_teacher_t.append(mean_errors_teacher)
            errors_local_t.append(mean_errors_local)
            errors_teacher_local_t.append(mean_errors_teacher_local)
            errors_unsup_t.append(mean_errors_unsup)
            errors_ref_t.append(mean_errors_ref)
            errors_local_ref_t.append(mean_errors_local_ref)
            errors_unsup_ref_t.append(mean_errors_unsup_ref)

        mean_errors_t = np.array(errors_t).mean(0)
        mean_errors_teacher_t = np.array(errors_teacher_t).mean(0)
        mean_errors_local_t = np.array(errors_local_t).mean(0)
        mean_errors_teacher_local_t = np.array(errors_teacher_local_t).mean(0)
        mean_errors_unsup_t = np.array(errors_unsup_t).mean(0)
        mean_errors_ref_t = np.array(errors_ref_t).mean(0)
        mean_errors_local_ref_t = np.array(errors_local_ref_t).mean(0)
        mean_errors_unsup_ref_t = np.array(errors_unsup_ref_t).mean(0)

        print("total")
        print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "median"))
        print("supervised w/o gt median scaling (teacher/student)")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_ref_t.tolist()) + "\\\\")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_teacher_t.tolist()) + "\\\\")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_t.tolist()) + "\\\\")
        print("supervised w gt median scaling (teacher/student)")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_local_ref_t.tolist()) + "\\\\")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_teacher_local_t.tolist()) + "\\\\")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_local_t.tolist()) + "\\\\")
        print("self-supervised w gt median scaling")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_unsup_ref_t.tolist()) + "\\\\")
        print(("&{: 8.3f}  " * 8).format(*mean_errors_unsup_t.tolist()) + "\\\\")
        print("\n-> Done!")


    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # first update the self-supervised models
        for m in self.reg_models.values():
            m.train()

        reg_features = self.reg_models["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
        reg_outputs = self.reg_models["depth"](reg_features)
        reg_outputs.update(self.predict_poses(inputs, reg_features, self.reg_models))
        _, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
        self.generate_images_pred(inputs, reg_outputs)
        unsup_losses = self.compute_losses_unsup(inputs, reg_outputs)
        self.reg_model_optimizer.zero_grad()
        unsup_losses["loss"].backward()
        self.reg_model_optimizer.step()

        for m in self.reg_models.values():
            m.eval()

        for m in self.models.values():
            m.train()

        # run adaptation with pseudo labels
        features = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
        depth = self.models["depth"](features)
        with torch.no_grad():
            reg_features = self.reg_models["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            reg_outputs = self.reg_models["depth"](reg_features)
            reg_outputs.update(self.predict_poses(inputs, reg_features, self.reg_models))
            _, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            # reference models
            features_uncrop_ref = self.models_ref["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            depth_uncrop_ref = self.models_ref["depth"](features_uncrop_ref)

            features_uncrop_ema = self.models_ema["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            depth_uncrop_ema = self.models_ema["depth"](features_uncrop_ema)

            reg_features_ref = self.reg_models_ref["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            reg_outputs_ref = self.reg_models_ref["depth"](reg_features_ref)
            _, depth_unsup_ref = disp_to_depth(reg_outputs_ref[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            pseudo_depth_sup, pseudo_depth_unsup = self.augment_pseudo(inputs,
                                                                       depth_uncrop_ref,
                                                                       reg_depth_unsup,
                                                                       depth_uncrop_ema
                                                                       )

        loss_ada = self.compute_losses(depth, pseudo_depth_sup, pseudo_depth_unsup)
        self.model_optimizer.zero_grad()
        loss_ada.backward()
        self.model_optimizer.step()

        # update ema models
        update_ema_variables(self.models["encoder"],
                             self.models_ema["encoder"],
                             0.99,
                             self.step)
        update_ema_variables(self.models["depth"],
                             self.models_ema["depth"],
                             0.99,
                             self.step)

        for m in self.models.values():
            m.eval()

        with torch.no_grad():
            features_uncrop = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            depth_uncrop = self.models["depth"](features_uncrop)

            features_uncrop_ema = self.models_ema["encoder"]((inputs["color_uncrop", 0, 0]-self.mean)/self.std)
            depth_uncrop_ema = self.models_ema["depth"](features_uncrop_ema)

        error = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop))
        error_teacher = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop_ema))
        error_ref = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop_ref))
        error_local = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop, median_scaling=True))
        error_teacher_local = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop_ema, median_scaling=True))
        error_local_ref = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_uncrop_ref, median_scaling=True))

        error_unsup = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], reg_depth_unsup, median_scaling=True))
        error_unsup_ref = list(self.compute_depth_losses(inputs['depth_gt_uncrop'], depth_unsup_ref, median_scaling=True))

        for idx, term in enumerate(error):
            error[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_teacher):
            error_teacher[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_local):
            error_local[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_teacher_local):
            error_teacher_local[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_unsup):
            error_unsup[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_ref):
            error_ref[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_local_ref):
            error_local_ref[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_unsup_ref):
            error_unsup_ref[idx] = term.detach().cpu().numpy()

        outputs = {}
        outputs['depth'] = depth_uncrop
        outputs['pseudo_depth_sup'] = pseudo_depth_sup
        outputs['pseudo_depth_unsup'] = pseudo_depth_unsup

        losses = {}
        losses['loss'] = loss_ada.detach().cpu().numpy()
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = error[i]
        for i, metric in enumerate(self.depth_metric_names_local):
            losses[metric] = error_local[i]

        return error, error_local, error_unsup, outputs, losses,\
    error_ref, error_local_ref, error_unsup_ref, error_teacher, error_teacher_local


    def predict_poses(self, inputs, features, models):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        # select what features the pose network takes as input
        pose_feats = {f_i: (inputs["color_uncrop", f_i, 0]-self.mean)/self.std for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [models["pose_encoder"](torch.cat(pose_inputs, 1))]

            axisangle, translation = models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.height, self.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            source_scale = 0
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                source_image = inputs[("color_uncrop", frame_id, source_scale)]

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    source_image,
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                outputs[("color_identity", frame_id, scale)] = source_image


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


    def compute_losses_unsup(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color_uncrop", 0, scale)]
            target = inputs[("color_uncrop", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color_uncrop", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_loss = identity_reprojection_losses
            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            to_optimise, idxs = torch.min(combined, dim=1)
            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses


    def compute_losses(self, depth, pseudo_depth_sup, pseudo_depth_unsup):
        mask = pseudo_depth_sup > 1.0
        d = torch.log(depth[mask]) - torch.log(pseudo_depth_sup[mask])
        # scale invariant
        variance_focus = 0.85
        loss_sup = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0

        mask = pseudo_depth_unsup > 1.0
        d = torch.log(depth[mask]) - torch.log(pseudo_depth_unsup[mask])
        # scale invariant
        variance_focus = 0.85
        loss_unsup = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0

        loss = loss_sup + loss_unsup
        return loss


    def augment_pseudo(self, inputs, reg_depth_sup, reg_depth_unsup, depth_uncrop_ref):
        # reg depth sup is merely the current depth; difference is train/eval modes
        # generate mask by thresholding (output)
        # ema model
        mask0 = depth_uncrop_ref > self.MIN_DEPTH
        mask0[:, :, :int(0.3*depth_uncrop_ref.shape[2]), :] = False
        scale_factor = torch.median(depth_uncrop_ref[mask0]) / torch.median(reg_depth_unsup[mask0])
        if self.opt.dataset == "kitti":
            gt_height, gt_width = depth_uncrop_ref.shape[2], depth_uncrop_ref.shape[3]
            # garg/eigen crop
            crop_mask = torch.zeros_like(mask0)
            crop_mask[:, :, int(0.40810811*gt_height):int(0.99189189*gt_height),
                      int(0.03594771*gt_width):int(0.96405229*gt_width)] = 1
            mask0 = crop_mask

        # reg depth unsup, consistency with ema model
        mask = (((depth_uncrop_ref - reg_depth_unsup * scale_factor) ** 2) / depth_uncrop_ref) < self.opt.thres

        reg_depth_unsup = torch.mul(mask.float(), reg_depth_unsup) * scale_factor
        reg_depth_unsup = torch.clamp(reg_depth_unsup, min=self.MIN_DEPTH, max=self.MAX_DEPTH)

        # # reg depth sup
        mask = (((reg_depth_sup - depth_uncrop_ref) ** 2) / reg_depth_sup) < self.opt.thres
        mask = mask * mask0

        reg_depth_sup = torch.mul(mask.float(), depth_uncrop_ref)
        reg_depth_sup = torch.clamp(reg_depth_sup, min=self.MIN_DEPTH, max=self.MAX_DEPTH)

        return reg_depth_sup, reg_depth_unsup


    def tensor_augmentation(self, input, rotate_angle, crop_factor, do_flip):
        # rotate
        rotate_angle = rotate_angle * math.pi / 180
        rot_mat = torch.tensor([[torch.cos(rotate_angle), -torch.sin(rotate_angle), 0],
                                [torch.sin(rotate_angle), torch.cos(rotate_angle), 0]])
        rot_mat = rot_mat[None, ...].repeat(input.shape[0], 1, 1)
        grid = F.affine_grid(rot_mat, input.size()).to(self.device)
        input = F.grid_sample(input, grid).squeeze(0)

        # crop
        x = int(crop_factor * (input.shape[2] - self.width))
        y = int(crop_factor * (input.shape[1] - self.height))
        input = input[:, y:y+self.height, x:x+self.width]

        # flip
        if do_flip:
            input = torch.flip(input, [2])

        return input.unsqueeze(0)


    def compute_depth_losses(self, depth_gt, depth_pred, median_scaling=False):
        # scale by camera height
        gt_height, gt_width = depth_gt.shape[2], depth_gt.shape[3]
        mask = torch.logical_and(depth_gt > self.MIN_DEPTH, depth_gt < self.MAX_DEPTH)
        depth_pred = F.interpolate(depth_pred, [gt_height, gt_width], mode="bilinear", align_corners=False)

        if self.opt.dataset == "kitti":
            # garg/eigen crop
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, int(0.40810811*gt_height):int(0.99189189*gt_height),
                      int(0.03594771*gt_width):int(0.96405229*gt_width)] = 1
            mask = mask * crop_mask

        # mask0 = torch.ones_like(mask)
        # mask0[:, :, :int(0.3*depth_gt.shape[2]), :] = 0
        # mask = mask * mask0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        median_ratio = torch.median(depth_gt)/torch.median(depth_gt)
        if median_scaling:
            median_ratio = torch.median(depth_gt) / torch.median(depth_pred)
            depth_pred *= median_ratio

        depth_gt = torch.clamp(depth_gt, min=self.MIN_DEPTH, max=self.MAX_DEPTH)
        depth_pred = torch.clamp(depth_pred, min=self.MIN_DEPTH, max=self.MAX_DEPTH)

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(depth_gt, depth_pred)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, median_ratio


    def log(self, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writer
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        j = 0
        s = 0
        frame_id = 0

        cmap = matplotlib.cm.viridis
        cmap.set_bad('white',1.)

        # prediction
        pred = copy.deepcopy(outputs["depth"][j]).squeeze().detach().cpu().numpy()
        pred = (255 * cmap(pred/80)).astype('uint8')[:,:,:3]
        pred = np.transpose(pred, (2,0,1))
        writer.add_image("vis/pred_{}".format(s, j),
                         pred,
                         self.step)

        # vis gt displacement map
        depth_gt = copy.deepcopy(inputs["depth_gt"][j]).squeeze().detach().cpu().numpy()
        depth_gt = np.where(depth_gt, depth_gt, np.nan)
        depth_gt = (255 * cmap(depth_gt/80)).astype('uint8')[:,:,:3]
        depth_gt = np.transpose(depth_gt, (2,0,1))
        writer.add_image("depth_gt/{}".format(j),
                         depth_gt,
                         self.step)

        writer.add_image(
            "vis/color_{}_{}_{}".format(frame_id, s, j),
            inputs[("color_uncrop", frame_id, s)][j].data, self.step)

        # pseudo1
        pred = copy.deepcopy(outputs["pseudo_depth_sup"][j]).squeeze().detach().cpu().numpy()
        pred = (255 * cmap(pred/80)).astype('uint8')[:,:,:3]
        pred = np.transpose(pred, (2,0,1))
        writer.add_image("vis/pseudo_sup_{}_{}".format(s, j),
                         pred,
                         self.step)

        # pseudo2
        pred = copy.deepcopy(outputs["pseudo_depth_unsup"][j]).squeeze().detach().cpu().numpy()
        pred = (255 * cmap(pred/80)).astype('uint8')[:,:,:3]
        pred = np.transpose(pred, (2,0,1))
        writer.add_image("vis/pseudo_unsup_{}_{}".format(s, j),
                         pred,
                         self.step)


    def load_model(self, load_path, model_name, models_to_load):
        """Load model(s) from disk
        """
        load_path = os.path.expanduser(load_path)

        assert os.path.isdir(load_path), \
            "Cannot find folder {}".format(load_path)
        print("loading model from folder {}".format(load_path))

        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_path, "{}.pth".format(n))
            model_dict = model_name[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model_name[n].load_state_dict(model_dict)


if __name__ == "__main__":
    adapt = Adapt(opts)
    adapt.run_adapt()
