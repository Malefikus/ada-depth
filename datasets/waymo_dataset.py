# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import PIL.Image as pil
from skimage import transform
import random
import copy

import tensorflow.compat.v1 as tf
import math
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import pdb

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def is_list(data):
    """Checks if data is a list."""
    return isinstance(data, list)

def is_numpy(data):
    """Checks if data is a numpy array."""
    return isinstance(data, np.ndarray)

def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor

def make_list(var, n=None):
    """
    Wraps the input into a list, and optionally repeats it to be size n
    Parameters
    ----------
    var : Any
        Variable to be wrapped in a list
    n : int
        How much the wrapped variable will be repeated
    Returns
    -------
    var_list : list
        List generated from var
    """
    var = var if is_list(var) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n, 'Wrong list length for make_list'
        return var * n if len(var) == 1 else var


########################################################################################################################
#### DATASET
########################################################################################################################

class WaymoDataset:
    """
    Waymo dataset class

    Parameters
    ----------
    path : str
        Path to the dataset
    filenames : list of str
        TF record names
    camera : int
        Which camera to get information from
    data_transform : Function
        Transformations applied to the sample
    """
    def __init__(self, path, filenames,
                 height=None,
                 width=None,
                 frame_idxs=[0, -1, 1],
                 num_scales=4,
                 is_train=False,
                 flip_aug=True,
                 rotate_aug=True,
                 img_ext='.jpg',
                 camera=0,
                 pseu=False,
                 tgt_height=1.65,
                 tgt_focal=720,
                 depth_path=None,
                 ):
        self.path = path
        self.depth_path = depth_path
        self.filenames = filenames
        self.camera = camera
        # self.full_res = (1920, 1248)
        # default target KITTI dataset
        self.tgt_height = tgt_height
        self.tgt_focal = tgt_focal
        self.cam_height = 2.12
        self.pseu = pseu

        self.frame_idxs = frame_idxs

        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.is_train = is_train
        self.flip_aug = flip_aug
        self.rotate_aug = rotate_aug
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        self.data_transform = transforms.ColorJitter.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        self.to_tensor = transforms.ToTensor()
        self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # a list of tf record path + frame number
        self.read_tfrecords()


    def read_tfrecords(self):
        tf_dataset = []
        dataset_list = []
        for rec_idx, record in enumerate(self.filenames):
            print('reading ' + record)
            file_path = os.path.join(self.path, record)
            dataset = tf.data.TFRecordDataset(file_path, compression_type='')
            tf_dataset.append(dataset)
            for idx, data in enumerate(dataset):
                # this is for 3-frame, alter this when self.frame_idxs changes!!!
                if idx - 1 < 1:
                    continue
                dataset_list.append(str(rec_idx)+' '+str(idx-1))
        self.tf_dataset = tf_dataset
        self.dataset_list = dataset_list


    def get_frame_sample(self, record_idx, frame_num):
        dataset = self.tf_dataset[record_idx]
        for idx, data in enumerate(dataset):
            if idx < frame_num:
                continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            return frame


    def generate_depth_map(self, frame_sample, gt_size, sample_idx, datum_idx, filename):
        """
        Generates the depth map for a camera by projecting LiDAR information.
        It also caches the depth map following DGP folder structure, so it's not recalculated

        Parameters
        ----------
        gt_size : (width, height)
        sample_idx : int
            sample index
        datum_idx : int
            camera index
        filename :
            Filename used for loading / saving

        Returns
        -------
        depth : np.array [H, W]
            Depth map for that datum in that sample
        """
        # Generate depth filename
        filename = '{}/{}/{}.npz'.format(
            self.depth_path, filename, '%04d'%sample_idx)
        # Load and return if exists
        if os.path.exists(filename):
            return np.load(filename, allow_pickle=True)['depth']
        # Otherwise, create, save and return
        else:
            # reproject point cloud
            (range_images,
             camera_projections,
             range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame_sample)
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame_sample,
                range_images,
                camera_projections,
                range_image_top_pose)

            # 3d points in vehicle frame.
            points_all = np.concatenate(points, axis=0)
            # camera projection corresponding to each point.
            cp_points_all = np.concatenate(cp_points, axis=0)

            images = sorted(frame_sample.images, key=lambda i:i.name)
            cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
            cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

            # The distance between lidar points and vehicle frame origin.
            points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
            cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
            mask = tf.equal(cp_points_all_tensor[..., 0], images[datum_idx].name)
            cp_points_all_tensor = tf.cast(tf.gather_nd(
                cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
            points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

            img_indices = cp_points_all_tensor[..., 1:3].numpy().astype(int).transpose()
            depth_val = points_all_tensor.numpy().reshape(-1)

            depth = np.zeros(gt_size)
            depth[img_indices[0], img_indices[1]] = depth_val
            depth = depth.transpose()

            # Save depth map
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, depth=depth)
            # Return depth map
            return depth


    def get_filename(self, sample_idx, datum_idx):
        """
        Returns the filename for an index, following DGP structure

        Parameters
        ----------
        sample_idx : int
            Sample index
        datum_idx : int
            Datum index

        Returns
        -------
        filename : str
            Filename for the datum in that sample
        """
        scene_idx, sample_idx_in_scene, datum_indices = self.dataset.dataset_item_index[sample_idx]
        scene_dir = self.dataset.dataset_metadata.directory
        filename = self.dataset.get_datum(
            scene_idx, sample_idx_in_scene, datum_indices[datum_idx]).datum.image.filename
        return os.path.splitext(os.path.join(os.path.basename(scene_dir),
                                             filename.replace('rgb', '{}')))[0]

    def get_colour(self, color, do_flip, rotate_angle, crop_factor, width, height):
        # kb crop instead of resizing
        top_margin = int(self.orig_height - self.full_res[1])
        left_margin = int((self.orig_width - self.full_res[0]) / 2)
        color = color.crop((left_margin, top_margin,
                            left_margin + self.full_res[0],
                            top_margin + self.full_res[1]))
        # resize image
        color = color.resize((self.width_sec, self.height_sec), pil.ANTIALIAS)

        # random rotate
        if rotate_angle:
            color = color.rotate(rotate_angle, resample=pil.BILINEAR)

        # random crop
        x = int(crop_factor * (color.size[0] - width))
        y = int(crop_factor * (color.size[1] - height))
        box = (x, y, x + width, y + height)
        color = color.crop(box)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, depth_gt, do_flip, rotate_angle, crop_factor, width, height):

        # kb crop
        top_margin = int(self.orig_height - self.full_res[1])
        left_margin = int((self.orig_width - self.full_res[0]) / 2)
        depth_gt = depth_gt[top_margin:top_margin+self.full_res[1],
                            left_margin:left_margin+self.full_res[0]]

        # random rotate
        depth_gt = pil.fromarray(depth_gt*256).convert('I')
        if rotate_angle:
            depth_gt = depth_gt.rotate(rotate_angle, resample=pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        # random crop
        assert depth_gt.shape[0] >= int(height * self.resize_factor)
        assert depth_gt.shape[1] >= int(width * self.resize_factor)
        x = int(crop_factor * (depth_gt.shape[1] - int(width * self.resize_factor)))
        y = int(crop_factor * (depth_gt.shape[0] - int(height * self.resize_factor)))
        depth_gt = depth_gt[y:y + int(height * self.resize_factor),
                            x:x + int(width * self.resize_factor)]

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if "color_uncrop" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

        for k in list(inputs):
            f = inputs[k]
            if "color_uncrop" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        """Length of dataset"""
        return len(self.dataset_list)

    def __getitem__(self, idx):
        """Get a dataset sample"""
        # Get waymo sample indexing from the dataset list
        indices = self.dataset_list[idx].split(' ')
        record_idx, frame_num = int(indices[0]), int(indices[1])
        frame_samples = {}
        for i in self.frame_idxs:
            frame_samples[i] = self.get_frame_sample(record_idx, frame_num+i)

        inputs = {}

        # augmentations
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and self.flip_aug and random.random() > 0.5
        do_rotate = self.is_train and self.rotate_aug
        rotate_angle = (random.random() - 0.5) * 2 * 1.0 if do_rotate else 0
        crop_factor = random.random()

        inputs["do_flip"] = do_flip
        inputs["rotate_angle"] = torch.tensor(rotate_angle).type(torch.float32)
        inputs["crop_factor"] = torch.tensor(crop_factor).type(torch.float32)

        # camera intrinsics, self.camera (0 for FRONT)
        # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}]
        intrinsics_raw = frame_samples[0].context.camera_calibrations[self.camera].intrinsic
        gt_size = (frame_samples[0].context.camera_calibrations[self.camera].width,\
        frame_samples[0].context.camera_calibrations[self.camera].height)
        inputs["focal_length"] = torch.tensor(intrinsics_raw[0])

        # calculate sizes according to camera params
        self.resize_factor = intrinsics_raw[0] * self.cam_height / (self.tgt_focal * self.tgt_height)
        self.orig_height, self.orig_width = gt_size[1], gt_size[0]
        # calculate secondary sizes
        self.width_sec = int(self.orig_width / self.resize_factor) // 32 * 32
        self.height_sec = int(self.orig_height / self.resize_factor) // 32 * 32
        inputs["width_sec"], inputs["height_sec"] = self.width_sec, self.height_sec
        self.full_res = (int(self.width_sec * self.resize_factor),
                         int(self.height_sec * self.resize_factor))
        if self.height is None:
            self.height, self.width = self.height_sec, self.width_sec
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        cam_K = np.zeros((4, 4), dtype=np.float32)
        cam_K[0, 0], cam_K[1, 1] = intrinsics_raw[0], intrinsics_raw[1]
        cam_K[0, 2], cam_K[1, 2] = intrinsics_raw[2], intrinsics_raw[3]
        cam_K[2, 2] = 1
        cam_K[3, 3] = 1

        cam_K[0, :] = cam_K[0, :] / gt_size[0]
        cam_K[1, :] = cam_K[1, :] / gt_size[1]
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = cam_K.copy()
            K[0, :] *= self.width_sec // (2 ** scale)
            K[1, :] *= self.height_sec // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # rgb images
        for i in self.frame_idxs:
            colour_cur = tf.image.decode_jpeg(frame_samples[i].images[self.camera].image)
            colour_cur = pil.fromarray(colour_cur.numpy())
            inputs[("color", i, -1)] = self.get_colour(colour_cur, do_flip,
                                                       rotate_angle, crop_factor,
                                                       self.width, self.height)
            if self.pseu:
                inputs[("color_uncrop", i, -1)] = self.get_colour(colour_cur, False,
                                                                 0, 1,
                                                                 self.width_sec, self.height_sec)
            else:
                inputs[("color_uncrop", i, -1)] = self.get_colour(colour_cur, do_flip,
                                                                 0, 1,
                                                                 self.width_sec, self.height_sec)

        # colour augmentation for the inputs
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        self.preprocess(inputs, color_aug)
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            del inputs[("color_uncrop", i, -1)]
            del inputs[("color_uncrop_aug", i, -1)]

        # depth
        filename = frame_samples[0].context.name
        depth_raw = self.generate_depth_map(frame_samples[0], gt_size,
                                            frame_num, self.camera, filename)
        depth_gt = self.get_depth(depth_raw, do_flip,
                                  rotate_angle, crop_factor,
                                  self.width, self.height)
        inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        depth_gt_uncrop = self.get_depth(depth_raw, False, 0, 1,
                                         self.width_sec, self.height_sec)
        inputs["depth_gt_uncrop"] = np.expand_dims(depth_gt_uncrop, 0)
        inputs["depth_gt_uncrop"] = torch.from_numpy(inputs["depth_gt_uncrop"].astype(np.float32))

        return inputs

########################################################################################################################
