# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torchvision import transforms

import json
import os

from utils import *
from kitti_utils import *
from layers import *

import matplotlib.cm
import matplotlib.pyplot as plt

import datasets
import networks
from tqdm import tqdm

import copy
import pdb


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80
        if self.opt.dataset == "dgp":
            self.gt_height, self.gt_width = 1920, 1152
        else:
            # kitti
            self.gt_height, self.gt_width = 352, 1216
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' mustbe a multiple of 32"

        # create a new log for every save
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        if not self.opt.monodepth:
            self.opt.scales = [0]
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(self.device)

        if not self.opt.monodepth:
            self.models["encoder"] = networks.trans_backbone(
                'large07', './models/swin_large_patch4_window7_224_22k.pth')
            self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
            self.models["depth"] = networks.NewCRFDepth(version = 'large07', max_depth = 80)
        else:
            # resnet
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            fixing_layers = ['base_model.conv1', '.bn']
            for name, parameters in self.models["encoder"].named_parameters():
                if any(x in name for x in fixing_layers):
                    parameters.requires_grad = False
            self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
            self.models["depth"] = networks.DepthDecoder(self.models["encoder"].module.num_ch_enc, self.opt.scales)

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"] = torch.nn.DataParallel(self.models["depth"])
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.opt.monodepth:
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)
            fixing_layers = ['base_model.conv1', '.bn']

            for name, parameters in self.models["pose_encoder"].named_parameters():
                if any(x in name for x in fixing_layers):
                    parameters.requires_grad = False

            self.models["pose_encoder"] = torch.nn.DataParallel(self.models["pose_encoder"])
            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].module.num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

            self.models["pose"] = torch.nn.DataParallel(self.models["pose"])
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.load_weights_folder is not None:
            self.prev_epoch = int(self.opt.load_weights_folder.split('_')[-1])
        else:
            self.prev_epoch = -1

        eps = 1e-3
        weight_decay = 1e-3
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "kitti_depth": datasets.KITTIDepthDataset,
                         "dgp": datasets.DGPDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        if self.opt.dataset == "dgp":
            train_filenames = "train"
            val_filenames = "val"
        else:
            fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
            train_filenames = readlines(fpath.format("train"))
            val_filenames = readlines(fpath.format("val"))

        img_ext = '.png' if self.opt.png else '.jpg'

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        if self.opt.monodepth:
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, self.num_scales, is_train=True,
                rotate_aug=False, img_ext=img_ext)
            self.unsup_train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            self.num_sup_samples = 0
            self.num_unsup_samples = len(train_dataset)
        else:
            self.sup_train_loader = self.train_loader
            self.num_sup_samples = len(train_dataset)
            self.num_unsup_samples = 0

        if not self.opt.monodepth:
            self.num_total_steps = self.num_sup_samples // self.opt.batch_size * self.opt.num_epochs
        else:
            self.num_total_steps = self.num_unsup_samples // self.opt.batch_size * self.opt.num_epochs

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

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
        if not self.opt.resume:
            start_epoch = 0
            self.step = 0
        else:
            print('resuming from previous training')
            start_epoch = self.prev_epoch + 1
            if not self.opt.monodepth:
                self.step = self.num_sup_samples // self.opt.batch_size * start_epoch
            else:
                self.step = self.num_unsup_samples // self.opt.batch_size * start_epoch

        self.start_time = time.time()
        for self.epoch in range(start_epoch, self.opt.num_epochs):
            if not self.opt.monodepth:
                self.train_loader = self.sup_train_loader
                print("training supervised")
            else:
                print("training self-supervised")
                self.train_loader = self.unsup_train_loader

            print("epoch "+str(self.epoch))
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # print("Training")
        self.set_train()

        pbar = tqdm(total = len(self.train_loader))
        for batch_idx, inputs in enumerate(self.train_loader):
            self.loss_nan = False
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            if self.loss_nan:
                continue
            self.model_optimizer.zero_grad()
            losses["loss"].backward()

            for param_group in self.model_optimizer.param_groups:
                if self.epoch < self.opt.scheduler_step_size:
                    current_lr = self.opt.learning_rate
                else:
                    current_lr = self.opt.learning_rate * 0.1
                # current_lr = (self.opt.learning_rate - 0.1 * self.opt.learning_rate) * \
                # (1 - self.step / self.num_total_steps) ** 0.9 + 0.1 * self.opt.learning_rate
                param_group['lr'] = current_lr
            self.model_optimizer.step()

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                duration = time.time() - before_op_time
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                # self.val()

            self.step += 1

            pbar.update(1)
            pbar.set_description("loss: {:.4f}".format(losses["loss"].item()))

        pbar.close()

        duration = time.time() - before_op_time
        self.log_time(batch_idx, duration, losses["loss"].cpu().data)
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        if "depth_gt" in inputs:
            self.compute_depth_losses(inputs, outputs, losses)
        self.log("train", inputs, outputs, losses)
        # self.val()


    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}

        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        features = self.models["encoder"]((inputs["color_aug", 0, 0]-self.mean)/self.std)

        if not self.opt.monodepth:
            outputs["depth"] = self.models["depth"](features)
            losses = self.compute_losses_supervised(inputs, outputs)
        else:
            outputs = self.models["depth"](features)
            _, outputs["depth"] = disp_to_depth(outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            outputs.update(self.predict_poses(inputs, features, self.models))
            self.generate_images_pred(inputs, outputs, monodepth=self.opt.monodepth)
            losses = self.compute_losses(inputs, outputs)

        return outputs, losses


    def predict_poses(self, inputs, features, models):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            pose_feats = {f_i: (inputs["color_aug", f_i, 0]-self.mean)/self.std for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
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

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            pose_inputs = torch.cat(
                [(inputs[("color_aug", i, 0)]-self.mean)/self.std for i in self.opt.frame_ids if i != "s"], 1)
            pose_inputs = [models["pose_encoder"](pose_inputs)]

            axisangle, translation = models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

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

    def generate_images_pred(self, inputs, outputs, monodepth=True):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            if monodepth:
                disp = outputs[("disp", scale)]
            else:
                disp = outputs["depth"]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            source_scale = 0

            if monodepth:
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            else:
                depth = disp

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                source_image = inputs[("color", frame_id, source_scale)]

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    source_image,
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0

            if self.opt.monodepth:
                disp = outputs[("disp", scale)]
            else:
                disp = outputs["depth"]

            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            depth_gt = inputs["depth_gt"]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses
            reprojection_loss = reprojection_losses
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
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

        if torch.isnan(total_loss):
            print("loss is nan, skipping...")
            self.loss_nan = True

        return losses


    def compute_losses_supervised(self, inputs, outputs):
        """Compute the supervision and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        depth_gt = inputs["depth_gt"]
        gt_height, gt_width = depth_gt.shape[2], depth_gt.shape[3]
        depth_gt[depth_gt > self.MAX_DEPTH] = 0
        for scale in self.opt.scales:
            loss = 0

            depth_pred = outputs["depth"]
            depth_pred = F.interpolate(
                depth_pred, [gt_height, gt_width],
                mode="bilinear", align_corners=False)

            mask = depth_gt > 1.0
            # # garg/eigen crop
            # crop_mask = torch.zeros_like(mask)
            # crop_mask[:, :, int(0.40810811*self.gt_height):int(0.99189189*self.gt_height),
            #           int(0.03594771*self.gt_width):int(0.96405229*self.gt_width)] = 1
            # mask = mask * crop_mask

            d = torch.log(depth_pred[mask]) - torch.log(depth_gt[mask])
            # silog loss from eigen et. al. lambda (variance_focus) = 1. in bts lamda = 0.85
            variance_focus = 0.85
            sil_loss = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0
            loss += sil_loss

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """

        depth_gt = inputs["depth_gt"]
        gt_height, gt_width = depth_gt.shape[2], depth_gt.shape[3]
        # mask = depth_gt > 0
        mask = depth_gt > 1.0

        depth_pred = outputs["depth"]
        depth_pred = F.interpolate(
            depth_pred, [gt_height, gt_width],
            mode="bilinear", align_corners=False)

        # # garg/eigen crop
        # crop_mask = torch.zeros_like(mask)
        # crop_mask[:, :, int(0.40810811*self.gt_height):int(0.99189189*self.gt_height),
        #           int(0.03594771*self.gt_width):int(0.96405229*self.gt_width)] = 1
        # mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        if self.opt.monodepth:
            depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_gt = torch.clamp(depth_gt, min=1e-3, max=80)
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = depth_errors[i].detach().cpu().numpy()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
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
            # vis gt displacement map
            depth_gt = copy.deepcopy(inputs["depth_gt"][j])
            # depth_gt[depth_gt > 0] = depth_to_disp(depth_gt[depth_gt > 0], self.opt.min_depth, self.opt.max_depth)
            writer.add_image(
                "depth_gt/{}".format(j),
                normalize_image(depth_gt),
                self.step)
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)

                if not self.opt.monodepth:
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs["depth"][j]), self.step)
                else:
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.monodepth:
                    for frame_id in self.opt.frame_ids:
                        if s == 0 and frame_id != 0:
                            writer.add_image(
                                "color_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color", frame_id, s)][j].data, self.step)

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
            try:
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Cannot find Adam weights so Adam is randomly initialized")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
