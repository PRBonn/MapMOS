#!/usr/bin/env python3
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
import typer
import importlib

import numpy as np
import torch
from typing import List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from mapmos.datasets.mapmos_dataset import MapMOSDataset, collate_fn
from mapmos.config import load_config


def cache_to_ply(
    data: Path = typer.Argument(
        ...,
        help="The data directory used by the specified dataloader",
        show_default=False,
    ),
    dataloader: str = typer.Argument(
        ...,
        help="The dataloader to be used",
        show_default=False,
    ),
    cache_dir: Path = typer.Argument(
        ...,
        help="The directory where the cache should be created",
        show_default=False,
    ),
    sequence: List[str] = typer.Option(
        None,
        "--sequence",
        "-s",
        show_default=False,
        help="[Optional] Cache specific sequences",
        rich_help_panel="Additional Options",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        show_default=False,
        help="[Optional] Path to the configuration file",
    ),
):
    try:
        o3d = importlib.import_module("open3d")
    except ModuleNotFoundError as err:
        print(f'open3d is not installed on your system, run "pip install open3d"')
        exit(1)

    for seq in sequence:
        # Run
        cfg = load_config(config)

        data_iterable = DataLoader(
            MapMOSDataset(
                dataloader=dataloader,
                data_dir=data,
                config=cfg,
                sequences=seq,
                cache_dir=cache_dir,
            ),
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=0,
            batch_sampler=None,
        )

        dataset_sequence = (
            data_iterable.dataset.datasets[seq].sequence_id
            if hasattr(data_iterable.dataset.datasets[seq], "sequence_id")
            else os.path.basename(data_iterable.dataset.datasets[seq].data_dir)
        )
        path = os.path.join("ply", dataset_sequence)
        os.makedirs(path, exist_ok=True)

        for idx, batch in enumerate(
            tqdm(data_iterable, desc="Writing data to ply", unit=" items", dynamic_ncols=True)
        ):
            mask_scan = batch[:, 4] == idx
            scan_points = batch[mask_scan, 1:4]
            scan_labels = batch[mask_scan, 6]

            map_points = batch[~mask_scan, 1:4]
            map_timestamps = batch[~mask_scan, 5]
            map_labels = batch[~mask_scan, 6]

            min_time = torch.min(batch[:, 5])
            max_time = torch.max(batch[:, 5])

            pcd_scan = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(scan_points.numpy())
            ).paint_uniform_color([0, 0, 1])
            scan_colors = np.array(pcd_scan.colors)
            scan_colors[scan_labels == 1] = [1, 0, 0]
            pcd_scan.colors = o3d.utility.Vector3dVector(scan_colors)

            pcd_map = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(map_points.numpy())
            ).paint_uniform_color([0, 0, 0])
            map_colors = np.array(pcd_map.colors)
            map_timestamps_norm = (map_timestamps - min_time) / (max_time - min_time)
            for i in range(len(map_colors)):
                t = map_timestamps_norm[i]
                map_colors[i, :] = [t, t, t]
            map_colors[map_labels == 1] = [1, 0, 0]
            pcd_map.colors = o3d.utility.Vector3dVector(map_colors)

            o3d.io.write_point_cloud(os.path.join(path, f"{idx:06}.ply"), pcd_scan + pcd_map)


if __name__ == "__main__":
    typer.run(cache_to_ply)
