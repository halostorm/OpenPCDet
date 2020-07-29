# -*- coding: UTF-8 -*-
import numpy as np
from pypcd import pypcd
from .geometry import *
import random
import numba

import json
from math import *
from scipy.spatial.transform import Rotation as R


def get_objects_from_label(label_file):
    values = json.load(open(label_file, 'r'))
    objects = [Object3d(value) for value in values]
    return objects


LIDAR_TYPE = ['Car', 'Truck', 'Bus', 'Cyclist', 'Pedestrian',
              'Tricar', 'TrafficCone', 'Unknow', 'DontCare', '/',
              'Car', 'Truck', 'Bus', 'Tricar', 'Other']


class Object3d(object):
    def __init__(self, value):
        self.type = LIDAR_TYPE[value['type']]
        self.loc = np.array(value['position'])
        self.rota = np.array(value['rpy'])
        self.heading = float(value['rpy'][2])
        self.size = np.array(value['size'])
        self.score = value['score'] if 'score' in value else -1
        self.rotation_z = atan2(self.loc[1], self.loc[0])
        # self.points_num = value['points_num']
        self.level = 0
        self.get_corners()

    def get_corners(self):
        pts_size = []
        pts_size.append(
            [+ self.size[0] / 2, + self.size[1] / 2, + self.size[2] / 2])
        pts_size.append(
            [+ self.size[0] / 2, - self.size[1] / 2, + self.size[2] / 2])
        pts_size.append(
            [- self.size[0] / 2, - self.size[1] / 2, + self.size[2] / 2])
        pts_size.append(
            [- self.size[0] / 2, + self.size[1] / 2, + self.size[2] / 2])

        pts_size.append(
            [+ self.size[0] / 2, + self.size[1] / 2, - self.size[2] / 2])
        pts_size.append(
            [+ self.size[0] / 2, - self.size[1] / 2, - self.size[2] / 2])
        pts_size.append(
            [- self.size[0] / 2, - self.size[1] / 2, - self.size[2] / 2])
        pts_size.append(
            [- self.size[0] / 2, + self.size[1] / 2, - self.size[2] / 2])

        r = R.from_euler('xyz', self.rota)
        rota_matrix = r.as_dcm()
        pts_rota = np.dot(rota_matrix, np.transpose(np.array(pts_size)))
        self.corners = np.transpose(pts_rota) + self.loc


@numba.njit
def get_corners_offset(dimension):
    """
    The center point is the physical center point of the 3D box, not the
    bottom or top center point.
    Inputs:
      dimension: [N, 3(w, l, h)] tensor
    Returns:
      corners_offset: [N, 8, 3(x, y, z)] tensor. Corners is in the following order:
              7 -------- 6
             /|         /|             z
            4 -------- 5 |             |  y
            | |    •   | |             | /
            | 3 -------- 2             .---- x
            |/         |/
            0 -------- 1
      Corners offset represents the offset betweet centers and corners.
      Formular:
                corners = corners_offset + center_point
    """
    corners_norm = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype=np.float64)
    corners_norm = corners_norm.reshape(1, 8, 3)
    dimension = dimension.reshape(-1, 1, 3)
    corners_offset = dimension * corners_norm

    return corners_offset


def center_to_corner_boxes_3d_layer(box_3d, rotate_axis):
    """
    Input:
      box_3d: [N, 7(x, y, z, d_x, d_y, d_z, r)] tensor, d_x, d_y, d_z represents
        dimensions in x, y, z axis respectively.
      rotate_axis: 'x' or 'y' or 'z', represent the rotation axis of r
        variable in box_3d
    Returns:
      box_corners_3d: [N, 8, 3], the second axis represents the eight corners of
        target cube; the third axis represents x, y, z coord of corners.
    """
    num_box = box_3d.shape[0]
    get_rot_matrix = {'x': lambda r: get_rotx_matrix(r),
                      'y': lambda r: get_roty_matrix(r),
                      'z': lambda r: get_rotz_matrix(r)}

    corners_offset = get_corners_offset(box_3d[:, 3:6])

    box_corners_3d = np.zeros((num_box, 8, 3), dtype=np.float64)
    for i in range(num_box):
        rot_mat = get_rot_matrix[rotate_axis](box_3d[i, 6])
        corners_offset_trans = points_transform(corners_offset[i], rot_mat.T)

        box_corners_3d[i] = box_3d[i, :3].reshape(1, 3) + corners_offset_trans

    return box_corners_3d


def parse_points(pcloud, box_3d):
    box_3d_corners = center_to_corner_boxes_3d_layer(
        np.expand_dims(np.array(box_3d[:7], dtype=np.float64),
                       axis=0), rotate_axis='z')
    box_3d_surfaces = corner_to_surfaces_3d_jit(
        box_3d_corners)
    pt_masks = points_in_convex_polygon_3d_jit(
        pcloud, box_3d_surfaces)
    gt_points = pcloud[pt_masks[0, :]]

    return gt_points
