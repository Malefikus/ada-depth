# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
from skimage import transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
import pdb


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # monodepth
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)
        # self.full_res_shape = (1242, 375)
        # BTS
        # self.K = np.array([[0.59, 0, 0.5, 0],
        #                    [0, 2.05, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (1216, 352)

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip,
                  rotate_angle, crop_factor, width, height):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        orig_height, orig_width = color.size[1], color.size[0]

        # color = color.resize(self.full_res_shape, pil.ANTIALIAS)
        # kb crop instead of resizing
        top_margin = int(orig_height - self.full_res_shape[1])
        left_margin = int((orig_width - self.full_res_shape[0]) / 2)
        color = color.crop((left_margin, top_margin,
                            left_margin + self.full_res_shape[0],
                            top_margin + self.full_res_shape[1]))
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

    def get_cam_params(self, side, do_flip, crop_factor, width, height):
        # load the camera calibration file
        date = self.filenames[0].split()[0].split('/')[0]
        cam_params_file = os.path.join(
            self.data_path,
            date,
            "calib_cam_to_cam.txt"
        )
        file_dic = {}
        with open(cam_params_file, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                key = line.split(': ')[0]
                val = line.split(': ')[1]
                file_dic[key] = val.split()
        gt_imgsize = [float(i) for i in file_dic["S_rect_0{}".format(self.side_map[side])]]
        self.gt_imgsize = np.array(gt_imgsize)
        cam_K = [float(i) for i in file_dic["P_rect_0{}".format(self.side_map[side])]]
        cam_K = np.array(cam_K).reshape((3,4))
        self.focal_length = cam_K[0, 0]
        # move the principal points according to crop factors
        # kb crop
        top_margin = int(self.gt_imgsize[1] - self.full_res_shape[1])
        left_margin = int((self.gt_imgsize[0] - self.full_res_shape[0]) / 2)
        cam_K[0, 2] = cam_K[0, 2] - left_margin
        cam_K[1, 2] = cam_K[1, 2] - top_margin

        # # random crop
        # x = int(crop_factor * (self.full_res_shape[0] - width))
        # y = int(crop_factor * (self.full_res_shape[1] - height))
        # cam_K[0, 2] = cam_K[0, 2] - x
        # cam_K[1, 2] = cam_K[1, 2] - y
        # if do_flip:
        #     cam_K[0, 2] = width - cam_K[0, 2]

        cam_K[0, :] = cam_K[0, :] / self.gt_imgsize[0]
        cam_K[1, :] = cam_K[1, :] / self.gt_imgsize[1]
        cam_K[:, 3] = 0
        self.K = np.zeros((4, 4), dtype=np.float32)
        self.K[:3, :] = cam_K
        self.K[3, 3] = 1


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip,
                  rotate_angle, crop_factor, width, height):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        # depth_gt = skimage.transform.resize(
        #     depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True,
        #     mode='constant', anti_aliasing=False)
        # kb crop instead of resizing
        top_margin = int(depth_gt.shape[0] - self.full_res_shape[1])
        left_margin = int((depth_gt.shape[1] - self.full_res_shape[0]) / 2)
        depth_gt = depth_gt[top_margin:top_margin + self.full_res_shape[1],
                            left_margin:left_margin + self.full_res_shape[0]]

        depth_gt = pil.fromarray(depth_gt*256).convert('I')
        # random rotate
        if rotate_angle:
            depth_gt = depth_gt.rotate(rotate_angle, resample=pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        # random crop
        assert depth_gt.shape[0] >= height
        assert depth_gt.shape[1] >= width
        x = int(crop_factor * (depth_gt.shape[1] - width))
        y = int(crop_factor * (depth_gt.shape[0] - height))
        depth_gt = depth_gt[y:y + height, x:x + width]

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip,
                  rotate_angle, crop_factor, width, height):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        orig_height, orig_width = depth_gt.size[1], depth_gt.size[0]
        # depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        top_margin = int(orig_height - self.full_res_shape[1])
        left_margin = int((orig_width - self.full_res_shape[0]) / 2)
        depth_gt = depth_gt.crop((left_margin, top_margin,
                            left_margin + self.full_res_shape[0],
                            top_margin + self.full_res_shape[1]))

        # random rotate
        if rotate_angle:
            depth_gt = depth_gt.rotate(rotate_angle, resample=pil.NEAREST)

        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        # random crop
        assert depth_gt.shape[0] >= height
        assert depth_gt.shape[1] >= width
        x = int(crop_factor * (depth_gt.shape[1] - width))
        y = int(crop_factor * (depth_gt.shape[0] - height))
        depth_gt = depth_gt[y:y + height, x:x + width]

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
