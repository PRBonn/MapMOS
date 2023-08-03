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

import numpy as np
from mapmos.pybind import mapmos_pybind


class VoxelHashMap:
    def __init__(
        self,
        voxel_size: float,
        max_distance: float,
        max_points_per_voxel: int = 1,
    ):
        self._internal_map = mapmos_pybind._VoxelHashMap(
            voxel_size=voxel_size,
            max_distance=max_distance,
            max_points_per_voxel=max_points_per_voxel,
        )

    def clear(self):
        return self._internal_map._clear()

    def empty(self):
        return self._internal_map._empty()

    def update(self, points: np.ndarray, pose: np.ndarray, timestamp: int):
        self._internal_map._update(mapmos_pybind._Vector3dVector(points), pose, timestamp)

    def point_cloud_with_timestamps(self):
        map_points, map_timestamps = self._internal_map._point_cloud_with_timestamps()
        return np.asarray(map_points), np.asarray(map_timestamps)

    def remove_voxels_far_from_location(self, location: np.ndarray):
        self._internal_map._remove_far_away_points(location)

    def update_belief(self, points: np.ndarray, logits: np.ndarray):
        self._internal_map._update_belief(mapmos_pybind._Vector3dVector(points), logits)

    def get_belief(self, points: np.ndarray):
        belief = self._internal_map._get_belief(mapmos_pybind._Vector3dVector(points))
        return np.asarray(belief)
