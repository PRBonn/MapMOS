# MIT License
#
# Copyright (c) 2023 Benedikt Mersch, Tiziano Guadagnino, Ignacio Vizzo, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import open3d as o3d
import numpy as np
import os
from pathlib import Path


def save_to_ply(
    scan_points: np.ndarray, pred_labels: np.ndarray, gt_labels: np.ndarray, filename: str
):
    os.makedirs(Path(filename).parent, exist_ok=True)
    pcd_current_scan = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(scan_points)
    ).paint_uniform_color([0, 0, 0])

    scan_colors = np.array(pcd_current_scan.colors)

    tp = (pred_labels == 1) & (gt_labels == 1)
    fp = (pred_labels == 1) & (gt_labels != 1)
    fn = (pred_labels != 1) & (gt_labels == 1)

    scan_colors[tp] = [0, 1, 0]
    scan_colors[fp] = [1, 0, 0]
    scan_colors[fn] = [0, 0, 1]

    pcd_current_scan.colors = o3d.utility.Vector3dVector(scan_colors)
    o3d.io.write_point_cloud(filename, pcd_current_scan)


def save_to_kitti(pred_labels: np.ndarray, filename: str):
    os.makedirs(Path(filename).parent, exist_ok=True)
    kitti_labels = np.copy(pred_labels)
    kitti_labels[pred_labels == 0] = 9
    kitti_labels[pred_labels == 1] = 251
    kitti_labels = kitti_labels.reshape(-1).astype(np.int32)
    kitti_labels.tofile(filename)
