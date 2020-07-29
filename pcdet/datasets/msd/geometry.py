# -*- coding: UTF-8 -*-
""" Define the operations for camera transformation
"""
import numba
import numpy as np
import os, math

__all__ = ['projection_decompose', 'get_frustum',
           'points_transform', 'points_transform_v2', 'corner_to_surfaces_3d_jit',
           'points_in_convex_polygon_3d_jit', 'remove_outside_image_points',
           'get_ranges', 'get_rotx_matrix', 'get_roty_matrix', 'get_rotz_matrix',
           'limit_period', 'euler_angles_from_rotation_matrix']


def projection_decompose(proj):
    """ Decompose the camera projection matrix (proj) into C @ [R | T], where
    C is the camera intrinsic parameters, R is the extrinsic rotation matrix,
    T = R @ t, where t is the extrinsic translation vector. The implementation
    is stable for all kitti projection matrix. If applied to other dataset, the
    users should verify it before running.

    :param proj: 3x4 3D camera projection matrix
    """
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def get_frustum(image_box, C, near_clip=0.001, far_clip=100):
    """ Generate the frustum of Image box w.r.t camera intrinsic matrix C.
    Notice, the returned frustum is under camera coordinate system, which means
    x-axis is horizontal towards right, z-axis is forwarding, y-aix is vertial
    towards floor.

    :param image_box: the box on image plane [x1, y1, x2, y2]. (0-indexed)
    :param C: the camera intrinsic matrix, 3x3. User can get the matrix from
    projection matrix (3x4) by function projection_decompose.
    :param near_clip: near coordinate for z-axis
    :param far_clip: far coordinate for z-axis
    """
    fu = C[0, 0]
    fv = C[1, 1]
    # original point offset
    cuv = C[0:2, 2]

    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, None]
    b = image_box
    # 4 x 2 matrix
    box_corners = np.array(
        [[b[0], b[1]],  # left-top
         [b[2], b[1]],  # right-top
         [b[2], b[3]],  # right-bottom
         [b[0], b[3]],  # left-bottom
         ], dtype=C.dtype)

    # inverse projection transformation, that is fX/Z + Px.
    near_box_corners = (box_corners - cuv) / np.array(
        [fu / near_clip, fv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - cuv) / np.array(
        [fu / far_clip, fv / far_clip], dtype=C.dtype)

    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=0)
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


def points_transform(points, trans_matrix_T):
    """ Transform points from one coordinate system to another
    Inputs:
      points: [N, 3(x, y, z)+...] the 2-D points tensor,
        where the first 3 elements in columns is x, y, z.
      trans_matrix_T: 4x4 transformation matrix from one coordinate to another
    """
    points_xyz = points[:, :3]
    points_xyz1 = np.concatenate(
        [points_xyz, np.ones((points.shape[0], 1))], axis=1)
    trans_points_xyz1 = points_xyz1 @ trans_matrix_T
    trans_points = trans_points_xyz1[:, :3] / trans_points_xyz1[:, 3:4]

    if points.shape[1] > 3:
        trans_points = np.concatenate([trans_points, points[:, 3:]])
    return trans_points


@numba.njit
def points_transform_v2(points, centers, point_masks, loc_trans, rot_trans):
    """ useful function to perform the points transform in random_perturb mode.
    :param points: point cloud points
    :param centers: the centers used to align the points to origin
    :param point_masks: whether the point will be transformed in this center
    :param loc_trans: location transformation
    :param rot_trans: rotation transformation
    """
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=np.float32)
    for i in range(num_box):
        rot = rot_trans[i]
        c, s = np.cos(rot), np.sin(rot)
        rot_mat_T[i, 0, 0] = c
        rot_mat_T[i, 0, 1] = s
        rot_mat_T[i, 1, 0] = -s
        rot_mat_T[i, 1, 1] = c
        rot_mat_T[i, 2, 2] = 1

    for i in range(num_points):
        for j in range(num_box):
            if point_masks[i, j] == 1:
                points[i, :3] -= centers[j, :3]
                points[i:i + 1, :3] = np.ascontiguousarray(
                    points[i:i + 1, :3]) @ rot_mat_T[j]
                points[i, :3] += centers[j, :3]
                points[i, :3] += loc_trans[j]
                # only apply 1st box transform
                break


@numba.njit
def corner_to_surfaces_3d_jit(corners):
    """ Convert 3d box (frustrum) from corners to surfaces that normal vectors
    all direct to external. Given a box like below, the corners is packed in
    ascend order.

    3d box:
      floor points: 0, 1, 2, 3
      ceil points: 4, 5, 6, 7
      vertical edge: (0, 4), (1, 5), (2, 6), (3, 7)

    :param: coerners (float array, [N, 8, 3]): 3d box (frustrum) corners.
    """
    num_boxes = corners.shape[0]
    surfaces = np.zeros(
        (num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([
        [0, 3, 2, 1],  # front
        [4, 5, 6, 7],  # back
        [0, 4, 7, 3],  # left
        [1, 2, 6, 5],  # right
        [0, 1, 5, 4],  # ceil
        [2, 3, 7, 6],  # floor
    ])
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]

    return surfaces


@numba.jit
def points_in_convex_polygon_3d_jit(points, surfaces):
    """ check points is in 3d convex polygons.
    :param points: [num, 3] array
    :param surfaces: [num_polygon, num_surfaces, num_points, 3]
    """
    dtype = surfaces.dtype
    num_points = points.shape[0]
    num_polygon, num_surfaces, num_surface_points, _ = surfaces.shape
    # store the normal vector for each surface pointing to the outside of
    # polygon.
    normal_vectors = np.zeros((num_polygon, num_surfaces, 3), dtype=dtype)
    # store the D values for each surface
    d = np.zeros((num_polygon, num_surfaces), dtype=dtype)
    bool_indices = np.ones((num_polygon, num_points), dtype=np.bool_)

    for i in range(num_polygon):
        for j in range(num_surfaces):
            sv1 = surfaces[i, j, 0] - surfaces[i, j, 2]
            sv2 = surfaces[i, j, 1] - surfaces[i, j, 2]

            # (x1 - x2) x (x1 - x3), outer product
            normal_vectors[i, j, 0] = sv1[1] * sv2[2] - sv1[2] * sv2[1]
            normal_vectors[i, j, 1] = sv1[2] * sv2[0] - sv1[0] * sv2[2]
            normal_vectors[i, j, 2] = sv1[0] * sv2[1] - sv1[1] * sv2[0]

            # using Ax + By + Cz + D = 0 to compute D
            d[i, j] = 0 - (surfaces[i, j, 0, 0] * normal_vectors[i, j, 0] +
                           surfaces[i, j, 0, 1] * normal_vectors[i, j, 1] +
                           surfaces[i, j, 0, 2] * normal_vectors[i, j, 2])

    sign = 0
    for i in range(num_points):
        for j in range(num_polygon):
            for k in range(num_surfaces):
                sign = (points[i, 0] * normal_vectors[j, k, 0] +
                        points[i, 1] * normal_vectors[j, k, 1] +
                        points[i, 2] * normal_vectors[j, k, 2] +
                        d[j, k])
                if sign >= 0:
                    bool_indices[j, i] = False
                    break

    return bool_indices


def remove_outside_image_points(points, proj_matrix, trans_matrix_T,
                                image_shape):
    """ Remove the points outside the frustum of image plane
    :param points: the 2-D points tensor, where the first 3 elements in columns
    is x, y, z.
    :param proj_matrix: 3x4 projection matrix
    :param image_shape: [height, width] of image
    :param trans_matrix_T: 4x4 transformation matrix from camera coordinate system
    to lidar coordinate system.
    """

    C, R, T = projection_decompose(proj_matrix)
    image_box = [0, 0, image_shape[1] - 1, image_shape[0] - 1]
    frustum = get_frustum(image_box, C)
    frustum -= T
    frustum = (np.linalg.inv(R) @ frustum.T).T
    # transform the points into lidar coordiante
    frustum = points_transform(frustum, trans_matrix_T)
    # extract the surface of frustum
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[None])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    inside_points = points[indices[0]]
    return inside_points


def get_ranges(points, boundaries):
    """ Get the minimum and maximum range of points
    in the boundaries.
    :param points: [N, 4], x, y, z, r
    :param boundaries: numpy, [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    """
    ranges_list = []
    for i in range(3):
        ranges = [np.max([np.min(points[:, i]), boundaries[i, 0]]),
                  np.min([np.max(points[:, i]), boundaries[i, 1]])]
        ranges_list.append(np.array(ranges))

    return ranges_list


def get_rotx_matrix(rx):
    """
    :param rz: angle in radian around x-axis
    """
    c = np.cos(rx)
    s = np.sin(rx)

    mat = np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])
    return mat


def get_roty_matrix(ry):
    """
    :param rz: angle in radian around y-axis
    """
    c = np.cos(ry)
    s = np.sin(ry)

    mat = np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])
    return mat


def get_rotz_matrix(rz):
    """
    :param rz: angle in radian around z-axis (up)
    """
    c = np.cos(rz)
    s = np.sin(rz)

    mat = np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return mat


def limit_period(val, offset=0., period=np.pi):
    """ limit the value period in interval [-period, period]
    """
    return val - np.floor(val / period + offset) * period


def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x - y) <= atol + rtol * abs(y)


def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    yaw = 0.0
    if isclose(R[2, 0], -1.0):
        pitch = np.pi / 2.0
        roll = np.arctan2(R[0, 1], R[0, 2])
    elif isclose(R[2, 0], 1.0):
        pitch = -np.pi / 2.0
        roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = -np.arcsin(R[2, 0])
        cos_pitch = np.cos(pitch)
        roll = np.arctan2(R[2, 1] / cos_pitch, R[2, 2] / cos_pitch)
        yaw = np.arctan2(R[1, 0] / cos_pitch, R[0, 0] / cos_pitch)
    return roll, pitch, yaw


def get_fc_from_intrinsic(K):
    """ return fx, fy, cx, cy """
    return K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def distort_point(homo_x, homo_y, distortion_coefficients):
    k = [0] * 12
    for i, distortion in enumerate(distortion_coefficients):
        k[i] = distortion

    r2 = homo_x * homo_x + homo_y * homo_y
    r4 = r2 * r2
    r6 = r4 * r2

    a1 = 2 * homo_x * homo_y
    a2 = r2 + 2 * homo_x * homo_x
    a3 = r2 + 2 * homo_y * homo_y

    cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6
    icdist2 = 1.0 / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6)

    dist_x = homo_x * cdist * icdist2 + k[2] * a1 + k[3] * a2 + k[8] * r2 + k[9] * r4
    dist_y = homo_y * cdist * icdist2 + k[2] * a3 + k[3] * a1 + k[10] * r2 + k[11] * r4

    return dist_x, dist_y


def undistort_point(x, y, K, distortion_coefficients):
    k = [0] * 12
    for i, distortion in enumerate(distortion_coefficients):
        k[i] = distortion

    fx, fy, cx, cy = get_fc_from_intrinsic(K)
    x = (x - cx) / fx
    y = (y - cy) / fy
    x0 = x
    y0 = y

    for i in range(20):
        r2 = x * x + y * y
        icdist = (1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2) / (1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2)
        deltaX = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x) + k[8] * r2 + k[9] * r2 * r2
        deltaY = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y + k[10] * r2 + k[11] * r2 * r2
        x = (x0 - deltaX) * icdist
        y = (y0 - deltaY) * icdist

        dist_x, dist_y = distort_point(x, y, distortion_coefficients)
        error = (np.abs(dist_x - x0) + np.abs(dist_y - y0)) * fx
        if error < 0.1:
            break

    if error > 1:
        return None, None

    point = np.array([x, y, 1]).reshape(3, 1)
    point = K.dot(point)
    x = point[0, 0] / point[2, 0]
    y = point[1, 0] / point[2, 0]

    return x, y
