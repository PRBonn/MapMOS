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

import os
import torch
import numpy as np
from typing import Dict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from mapmos.utils.cache import get_cache, memoize
from mapmos.config import MapMOSConfig, DataConfig, OdometryConfig
from mapmos.odometry import Odometry
from mapmos.mapping import VoxelHashMap
from mapmos.datasets import dataset_factory, sequence_dataloaders


def collate_fn(batch):
    # Returns tensor of [batch, x, y, z, t, scan_index, label]
    tensor_batch = None
    for i, (
        scan_points,
        map_points,
        scan_timestamps,
        map_timestamps,
        scan_labels,
        map_labels,
    ) in enumerate(batch):
        ones = torch.ones(len(scan_points), 1).type_as(scan_points)
        scan_points = torch.hstack(
            [
                i * ones,
                scan_points,
                0.0 * ones,
                scan_timestamps,
                scan_labels,
            ]
        )

        ones = torch.ones(len(map_points), 1).type_as(map_points)
        map_points = torch.hstack(
            [
                i * ones,
                map_points,
                -1.0 * ones,
                map_timestamps,
                map_labels,
            ]
        )

        tensor = torch.vstack([scan_points, map_points])
        tensor_batch = tensor if tensor_batch is None else torch.vstack([tensor_batch, tensor])
    return tensor_batch


class MapMOSDataModule(LightningDataModule):
    """Training and validation set for Pytorch Lightning"""

    def __init__(self, dataloader: str, data_dir: Path, config: MapMOSConfig, cache_dir: Path):
        super(MapMOSDataModule, self).__init__()
        self.dataloader = dataloader
        self.data_dir = data_dir
        self.config = config
        self.cache_dir = cache_dir
        if self.cache_dir == None:
            print("No cache specified, therefore disabling shuffle during training!")
        self.shuffle = True if self.cache_dir is not None else False

        assert dataloader in sequence_dataloaders()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_set = MapMOSDataset(
            self.dataloader, self.data_dir, self.config, self.config.training.train, self.cache_dir
        )
        val_set = MapMOSDataset(
            self.dataloader, self.data_dir, self.config, self.config.training.val, self.cache_dir
        )
        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.config.training.batch_size,
            collate_fn=collate_fn,
            shuffle=self.shuffle,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )

        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )

        self.valid_iter = iter(self.valid_loader)

        print(
            "Loaded {:d} training and {:d} validation samples.".format(len(train_set), len(val_set))
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


class MapMOSDataset(Dataset):
    """Caches and returns scan and local maps for multiple sequences"""

    def __init__(
        self,
        dataloader: str,
        data_dir: Path,
        config: MapMOSConfig,
        sequences: list,
        cache_dir: Path,
    ):
        self.config = config
        self.sequences = sequences

        # Cache
        if cache_dir is not None:
            directory = os.path.join(cache_dir, dataloader)
            self.use_cache = True
            self.cache = get_cache(directory=directory)
            print("Using cache at ", directory)
        else:
            self.use_cache = False
            self.cache = get_cache(directory=os.path.join(data_dir, "cache"))

        # Create datasets and map a sample index to the sequence and scan index
        self.datasets = {}
        self.idx_mapper = {}
        idx = 0
        for sequence in self.sequences:
            self.datasets[sequence] = dataset_factory(
                dataloader=dataloader,
                data_dir=data_dir,
                sequence=sequence,
            )
            for sample_idx in range(len(self.datasets[sequence])):
                self.idx_mapper[idx] = (sequence, sample_idx)
                idx += 1

        self.sequence = None
        self.odometry = Odometry(self.config.data, self.config.odometry)

    def __len__(self):
        return len(self.idx_mapper.keys())

    def __getitem__(self, idx):
        sequence, scan_index = self.idx_mapper[idx]
        (
            scan_points,
            map_points,
            scan_timestamps,
            map_timestamps,
            scan_labels,
            map_labels,
        ) = self.get_scan_and_map(
            sequence,
            scan_index,
            dict(self.config.data),
            dict(self.config.odometry),
        )
        return scan_points, map_points, scan_timestamps, map_timestamps, scan_labels, map_labels

    @memoize()
    def get_scan_and_map(
        self,
        sequence: int,
        scan_index: int,
        data_config_dict: Dict,
        odometry_config_dict: Dict,
    ):
        """Returns scan points, map points in local frame and labels. Scan and map need to be in
        local frame to allow for efficient cropping (sample point does not change). We use the
        VoxelHashMap to keep track of the GT labels for map points.
        """
        scan_points, timestamps, scan_labels = self.datasets[sequence][scan_index]

        # Only consider valid points
        valid_mask = scan_labels != -1
        scan_points = scan_points[valid_mask]
        scan_labels = scan_labels[valid_mask]

        if self.sequence != sequence:
            data_config = DataConfig().parse_obj(data_config_dict)
            odometry_config = OdometryConfig().parse_obj(odometry_config_dict)

            self.odometry = Odometry(data_config, odometry_config)
            self.gt_map = VoxelHashMap(odometry_config.voxel_size, data_config.max_range)
            self.sequence = sequence

        registered_map_points, map_timestamps = self.odometry.get_map_points()
        map_belief = self.gt_map.get_belief(registered_map_points)
        map_labels = np.zeros_like(map_belief)
        map_labels[map_belief > 0] = 1.0

        scan_points_registered = self.odometry.register_points(scan_points, timestamps, scan_index)
        update = -1.0 * np.ones_like(scan_labels)
        update[scan_labels == 1] = 1.0
        self.gt_map.update_belief(scan_points_registered, update)

        self.gt_map.remove_voxels_far_from_location(self.odometry.current_location())

        map_points = self.odometry.transform(
            registered_map_points, np.linalg.inv(self.odometry.current_pose())
        )

        scan_timestamps = scan_index * np.ones(len(scan_points))
        return (
            torch.tensor(scan_points).to(torch.float32).reshape(-1, 3),
            torch.tensor(map_points).to(torch.float32).reshape(-1, 3),
            torch.tensor(scan_timestamps).to(torch.float32).reshape(-1, 1),
            torch.tensor(map_timestamps).to(torch.float32).reshape(-1, 1),
            torch.tensor(scan_labels).to(torch.float32).reshape(-1, 1),
            torch.tensor(map_labels).to(torch.float32).reshape(-1, 1),
        )
