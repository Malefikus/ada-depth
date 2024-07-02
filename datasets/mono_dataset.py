# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os, sys
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import PIL.Image as pil

import torch
import torch.utils.data as data
from torchvision import transforms
sys.path.append('../')
from layers import *
import pdb


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 flip_aug=True,
                 rotate_aug=True,
                 img_ext='.jpg',
                 min_depth=0.1,
                 max_depth=100,
                 pseu=False,
                 depth_path=None,
                 ):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.flip_aug = flip_aug
        self.rotate_aug = rotate_aug
        self.pseu = pseu
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            # self.brightness = (0.9, 1.1)
            # self.contrast = (0.9, 1.1)
            # self.saturation = (0.9, 1.1)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            # self.brightness = 0.1
            # self.contrast = 0.1
            # self.saturation = 0.1
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        self.min_depth = min_depth
        self.max_depth = max_depth

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
                # inputs[(n, im, i)] = self.normalise(self.to_tensor(f))
                # inputs[(n + "_aug", im, i)] = self.normalise(self.to_tensor(color_aug(f)))
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

        for k in list(inputs):
            f = inputs[k]
            if "color_uncrop" in k:
                n, im, i = k
                # inputs[(n, im, i)] = self.normalise(self.to_tensor(f))
                # inputs[(n + "_aug", im, i)] = self.normalise(self.to_tensor(color_aug(f)))
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and self.flip_aug and random.random() > 0.5
        do_rotate = self.is_train and self.rotate_aug
        if do_rotate:
            # rotate_angle set to 1.0
            rotate_angle = (random.random() - 0.5) * 2 * 1.0
        else:
            rotate_angle = 0
        crop_factor = random.random()

        inputs["do_flip"] = do_flip
        inputs["rotate_angle"] = torch.tensor(rotate_angle).type(torch.float32)
        inputs["crop_factor"] = torch.tensor(crop_factor).type(torch.float32)

        inputs["width_sec"], inputs["height_sec"] = self.width, self.height

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        self.get_cam_params(side, do_flip, crop_factor, self.width, self.height)
        inputs["focal_length"] = torch.tensor(self.focal_length)

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip,
                                                          0, 1, self.width, self.height)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip,
                                                          rotate_angle, crop_factor, self.width, self.height)
                if self.pseu:
                    # as pseudo labels flip = False
                    inputs[("color_uncrop", i, -1)] = self.get_color(folder, frame_index + i, side, False,
                                                                    0, 1, self.full_res_shape[0], self.full_res_shape[1])
                else:
                    inputs[("color_uncrop", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip,
                                                                    0, 1, self.full_res_shape[0], self.full_res_shape[1])


        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

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

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip,
                                      rotate_angle, crop_factor, self.width, self.height)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

            depth_gt_uncrop = self.get_depth(folder, frame_index, side, False,
                                             0, 1, self.full_res_shape[0], self.full_res_shape[1])
            inputs["depth_gt_uncrop"] = np.expand_dims(depth_gt_uncrop, 0)
            inputs["depth_gt_uncrop"] = torch.from_numpy(inputs["depth_gt_uncrop"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip,
                  rotate_angle, crop_factor, width, height):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip,
                  rotate_angle, crop_factor, width, height):
        raise NotImplementedError
