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

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.utils.camera import Camera, generate_depth_map
from dgp.utils.pose import Pose

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


def stack_sample(sample):
    """Stack a sample from multiple sensors"""
    # If there is only one sensor don't do anything
    if len(sample) == 1:
        return sample[0]

    # Otherwise, stack sample
    stacked_sample = {}
    for key in sample[0]:
        # Global keys (do not stack)
        if key in ['idx', 'dataset_idx', 'sensor_name', 'filename']:
            stacked_sample[key] = sample[0][key]
        else:
            # Stack torch tensors
            if is_tensor(sample[0][key]):
                stacked_sample[key] = torch.stack([s[key] for s in sample], 0)
            # Stack numpy arrays
            elif is_numpy(sample[0][key]):
                stacked_sample[key] = np.stack([s[key] for s in sample], 0)
            # Stack list
            elif is_list(sample[0][key]):
                stacked_sample[key] = []
                # Stack list of torch tensors
                if is_tensor(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            torch.stack([s[key][i] for s in sample], 0))
                # Stack list of numpy arrays
                if is_numpy(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            np.stack([s[key][i] for s in sample], 0))

    # Return stacked sample
    return stacked_sample

########################################################################################################################
#### DATASET
########################################################################################################################

class DGPDataset:
    """
    DGP dataset class

    Parameters
    ----------
    path : str
        Path to the dataset
    split : str {'train', 'val', 'test'}
        Which dataset split to use
    cameras : list of str
        Which cameras to get information from
    depth_type : str
        Which lidar will be used to generate ground-truth information
    back_context : int
        Size of the backward context
    forward_context : int
        Size of the forward context
    data_transform : Function
        Transformations applied to the sample
    """
    def __init__(self, path, split,
                 height=None,
                 width=None,
                 frame_idxs=[0, -1, 1],
                 num_scales=4,
                 is_train=False,
                 flip_aug=True,
                 rotate_aug=True,
                 img_ext='.jpg',
                 camera=['camera_01'],
                 pseu=False,
                 tgt_height=1.65,
                 tgt_focal=720,
                 depth_path=None,
                 ):
        self.path = path
        self.split = split
        self.dataset_idx = 0
        # default target KITTI dataset
        self.tgt_height = tgt_height
        self.tgt_focal = tgt_focal
        self.cam_height = 1.63
        self.pseu = pseu

        self.frame_idxs = frame_idxs
        back_context = abs(min(frame_idxs)) if min(frame_idxs) < 0 else 0
        forward_context = max(frame_idxs) if max(frame_idxs) > 0 else 0
        self.bwd = back_context
        self.fwd = forward_context

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

        self.depth_type = 'lidar'

        self.dataset = SynchronizedSceneDataset(path,
            split=split,
            datum_names=camera,
            backward_context=back_context,
            forward_context=forward_context,
            requested_annotations=None,
            only_annotated_datums=False,
        )

    def generate_depth_map(self, sample_idx, datum_idx, filename):
        """
        Generates the depth map for a camera by projecting LiDAR information.
        It also caches the depth map following DGP folder structure, so it's not recalculated

        Parameters
        ----------
        sample_idx : int
            sample index
        datum_idx : int
            Datum index
        filename :
            Filename used for loading / saving

        Returns
        -------
        depth : np.array [H, W]
            Depth map for that datum in that sample
        """
        # Generate depth filename
        filename = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('depth/{}'.format(self.depth_type)))
        # Load and return if exists
        if os.path.exists(filename):
            return np.load(filename, allow_pickle=True)['depth']
        # Otherwise, create, save and return
        else:
            # Get pointcloud
            scene_idx, sample_idx_in_scene, _ = self.dataset.dataset_item_index[sample_idx]
            pc_datum_idx_in_sample = self.dataset.get_datum_index_for_datum_name(
                scene_idx, sample_idx_in_scene, self.depth_type)
            pc_datum_data = self.dataset.get_point_cloud_from_datum(
                scene_idx, sample_idx_in_scene, pc_datum_idx_in_sample)
            # Create camera
            camera_rgb = self.get_current('rgb', datum_idx)
            camera_pose = self.get_current('pose', datum_idx)
            camera_intrinsics = self.get_current('intrinsics', datum_idx)
            camera = Camera(K=camera_intrinsics, p_cw=camera_pose.inverse())
            # Generate depth map
            world_points = pc_datum_data['pose'] * pc_datum_data['point_cloud']
            depth = generate_depth_map(camera, world_points, camera_rgb.size[::-1])
            # Save depth map
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, depth=depth)
            # Return depth map
            return depth

    def get_current(self, key, sensor_idx):
        """Return current timestep of a key from a sensor"""
        return self.sample_dgp[self.bwd][sensor_idx][key]

    def get_backward(self, key, sensor_idx):
        """Return backward timesteps of a key from a sensor"""
        return [] if self.bwd == 0 else \
            [self.sample_dgp[i][sensor_idx][key] \
             for i in range(0, self.bwd)]

    def get_forward(self, key, sensor_idx):
        """Return forward timestep of a key from a sensor"""
        return [] if self.fwd == 0 else \
            [self.sample_dgp[i][sensor_idx][key] \
             for i in range(self.bwd + 1, self.bwd + self.fwd + 1)]

    def get_context(self, key, sensor_idx):
        """Get both backward and forward contexts"""
        return self.get_backward(key, sensor_idx) + self.get_forward(key, sensor_idx)

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
        # orig_height, orig_width = color.size[1], color.size[0]

        # color = color.resize(self.full_res, pil.ANTIALIAS)
        # kb crop instead of resizing
        top_margin = int(self.orig_height - self.full_res[1])
        left_margin = int((self.orig_width - self.full_res[0]) / 2)
        color = color.crop((left_margin, top_margin,
                            left_margin + self.full_res[0],
                            top_margin + self.full_res[1]))
        # resize image
        color = color.resize((self.width, self.height), pil.ANTIALIAS)

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
        assert depth_gt.shape[0] >= height
        assert depth_gt.shape[1] >= width
        x = int(crop_factor * (depth_gt.shape[1] - width))
        y = int(crop_factor * (depth_gt.shape[0] - height))
        depth_gt = depth_gt[y:y + height,
                            x:x + width]

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
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a dataset sample"""
        # Get DGP sample (if single sensor, make it a list)
        self.sample_dgp = self.dataset[idx]
        self.sample_dgp = [make_list(sample) for sample in self.sample_dgp]

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

        # i: sensor index
        sensor_id = 0

        # camera intrinsics
        intrinsics_raw = self.get_current('intrinsics', sensor_id)
        inputs["focal_length"] = torch.tensor(intrinsics_raw[0, 0])

        # calculate sizes according to camera params
        self.resize_factor = 2180 * self.cam_height / (self.tgt_focal * self.tgt_height)

        gt_size = self.sample_dgp[self.bwd][sensor_id]['rgb'].size
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
        cam_K[:3, :3] = intrinsics_raw
        cam_K[2, 2] = 1
        cam_K[3, 3] = 1
        gt_size = self.get_current('rgb', sensor_id).size
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
            idx_cur = self.bwd + i
            colour_cur = self.sample_dgp[idx_cur][sensor_id]['rgb']
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
        filename = self.get_filename(idx, sensor_id)
        depth_raw = self.generate_depth_map(idx, sensor_id, filename)
        if self.height is None:
            depth_gt = self.get_depth(depth_raw, do_flip,
                                      rotate_angle, crop_factor,
                                      int(self.width * self.resize_factor),
                                      int(self.height * self.resize_factor))
        else:
            depth_gt = self.get_depth(depth_raw, do_flip,
                                      rotate_angle, crop_factor,
                                      self.width, self.height)

        inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        depth_gt_uncrop = self.get_depth(depth_raw, False, 0, 1,
                                         int(self.width_sec * self.resize_factor),
                                         int(self.height_sec * self.resize_factor))
        inputs["depth_gt_uncrop"] = np.expand_dims(depth_gt_uncrop, 0)
        inputs["depth_gt_uncrop"] = torch.from_numpy(inputs["depth_gt_uncrop"].astype(np.float32))

        return inputs

########################################################################################################################
