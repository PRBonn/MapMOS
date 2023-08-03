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
import time
from pathlib import Path
from typing import Optional
from collections import deque
import torch
import numpy as np
from tqdm.auto import trange

from kiss_icp.pipeline import OdometryPipeline

from mapmos.mapmos_net import MapMOSNet
from mapmos.odometry import Odometry
from mapmos.mapping import VoxelHashMap
from mapmos.metrics import get_confusion_matrix
from mapmos.utils.visualizer import MapMOSVisualizer, StubVisualizer
from mapmos.utils.pipeline_results import MOSPipelineResults
from mapmos.utils.save import PlyWriter, KITTIWriter, StubWriter
from mapmos.config import load_config


class MapMOSPipeline(OdometryPipeline):
    def __init__(
        self,
        dataset,
        weights: Path,
        config: Optional[Path] = None,
        visualize: bool = False,
        save_ply: bool = False,
        save_kitti: bool = False,
        n_scans: int = -1,
        jump: int = 0,
    ):
        self._dataset = dataset
        self._n_scans = (
            len(self._dataset) - jump if n_scans == -1 else min(len(self._dataset) - jump, n_scans)
        )
        self._first = jump
        self._last = self._first + self._n_scans

        # Config and output dir
        self.config = load_config(config)
        self.results_dir = None

        # Pipeline
        state_dict = {
            k.replace("mos.", ""): v for k, v in torch.load(weights)["state_dict"].items()
        }
        self.model = MapMOSNet(self.config.mos.voxel_size_mos)

        self.model.load_state_dict(state_dict)
        self.model.cuda().eval().freeze()

        self.odometry = Odometry(self.config.data, self.config.odometry)
        self.belief = VoxelHashMap(
            voxel_size=self.config.mos.voxel_size_belief,
            max_distance=self.config.mos.max_range_belief,
        )
        self.buffer = deque(maxlen=self.config.mos.delay_belief)

        # Results
        self.results = MOSPipelineResults()
        self.poses = self.odometry.poses
        self.has_gt = hasattr(self._dataset, "gt_poses")
        self.gt_poses = self._dataset.gt_poses[self._first : self._last] if self.has_gt else None
        self.dataset_name = self._dataset.__class__.__name__
        self.dataset_sequence = (
            self._dataset.sequence_id
            if hasattr(self._dataset, "sequence_id")
            else os.path.basename(self._dataset.data_dir)
        )
        self.times_mos = []
        self.times_belief = []
        self.confusion_matrix_belief = torch.zeros(2, 2)

        # Visualizer
        self.visualize = visualize
        self.visualizer = MapMOSVisualizer() if visualize else StubVisualizer()

        self.ply_writer = PlyWriter() if save_ply else StubWriter()
        self.kitti_writer = KITTIWriter() if save_kitti else StubWriter()

    # Public interface  ------
    def run(self):
        self._create_output_dir()
        with torch.no_grad():
            self._run_pipeline()
        self._run_evaluation()
        self._write_result_poses()
        self._write_gt_poses()
        self._write_cfg()
        self._write_log()
        return self.results

    def _preprocess(self, points):
        ranges = np.linalg.norm(points - self.odometry.current_location(), axis=1)
        max_range = self.config.mos.max_range_mos
        mask = ranges <= max_range if max_range > 0 else np.ones_like(ranges, dtype=bool)
        mask = np.logical_and(mask, ranges >= self.config.mos.min_range_mos)
        return mask

    # Private interface  ------
    def _run_pipeline(self):
        pbar = trange(self._first, self._last, unit=" frames", dynamic_ncols=True)
        for scan_index in pbar:
            local_scan, timestamps, gt_labels = self._next(scan_index)
            map_points, map_indices = self.odometry.get_map_points()
            scan_points = self.odometry.register_points(local_scan, timestamps, scan_index)

            scan_mask = self._preprocess(scan_points)
            scan_points = torch.tensor(scan_points[scan_mask], dtype=torch.float32, device="cuda")
            gt_labels = gt_labels[scan_mask]

            map_mask = self._preprocess(map_points)
            map_points = torch.tensor(map_points[map_mask], dtype=torch.float32, device="cuda")
            map_indices = torch.tensor(map_indices[map_mask], dtype=torch.float32, device="cuda")

            start_time = time.perf_counter_ns()
            pred_logits_scan, pred_logits_map = self.model.predict(
                scan_points,
                map_points,
                scan_index * torch.ones(len(scan_points)).type_as(scan_points),
                map_indices,
            )
            self.times_mos.append(time.perf_counter_ns() - start_time)

            # Detach, move to CPU
            pred_logits_scan = pred_logits_scan.detach().cpu().numpy().astype(np.float64)
            pred_logits_map = pred_logits_map.detach().cpu().numpy().astype(np.float64)
            scan_points = scan_points.cpu().numpy().astype(np.float64)
            map_points = map_points.cpu().numpy().astype(np.float64)
            torch.cuda.empty_cache()

            pred_labels_scan = self.model.to_label(pred_logits_scan)
            pred_labels_map = self.model.to_label(pred_logits_map)

            # Probabilistic Volumetric Fusion
            map_mask = pred_logits_map > 0
            points_stacked = np.vstack([scan_points, map_points[map_mask]])
            logits_stacked = np.vstack(
                [pred_logits_scan.reshape(-1, 1), pred_logits_map[map_mask].reshape(-1, 1)]
            ).reshape(-1)

            start_time = time.perf_counter_ns()
            self.belief.update_belief(points_stacked, logits_stacked)
            belief_scan = self.belief.get_belief(scan_points)
            self.times_belief.append(time.perf_counter_ns() - start_time)
            belief_labels_scan = self.model.to_label(belief_scan)

            if self.visualize:
                belief_map = self.belief.get_belief(map_points)
                belief_labels_map = self.model.to_label(belief_map)
                self.visualizer.update(
                    scan_points,
                    map_points,
                    pred_labels_scan,
                    pred_labels_map,
                    belief_labels_scan,
                    belief_labels_map,
                    self.odometry.current_pose(),
                )

            # Evaluate and save with delay
            self.buffer.append([scan_index, scan_points, gt_labels])
            if len(self.buffer) == self.buffer.maxlen:
                query_index, query_points, query_labels = self.buffer.popleft()
                self.process_final_prediction(query_index, query_points, query_labels)

            # Clean up
            self.belief.remove_voxels_far_from_location(self.odometry.current_location())

        # Clear buffer at the end
        while len(self.buffer) != 0:
            query_index, query_points, query_labels = self.buffer.popleft()
            self.process_final_prediction(query_index, query_points, query_labels)

    def process_final_prediction(self, query_index, query_points, query_labels):
        belief_query = self.belief.get_belief(query_points)
        belief_labels_query = self.model.to_label(belief_query)
        self.confusion_matrix_belief += get_confusion_matrix(
            torch.tensor(belief_labels_query, dtype=torch.int32),
            torch.tensor(query_labels, dtype=torch.int32),
        )
        self.ply_writer.write(
            query_points,
            belief_labels_query,
            query_labels,
            filename=f"{self.results_dir}/ply/{query_index:06}.ply",
        )
        self.kitti_writer.write(
            belief_labels_query,
            filename=f"{self.results_dir}/bin/sequences/{self.dataset_sequence}/predictions/{query_index:06}.label",
        )

    def _next(self, idx):
        dataframe = self._dataset[idx]
        try:
            local_scan, timestamps, gt_labels = dataframe
        except ValueError:
            try:
                local_scan, timestamps = dataframe
                gt_labels = -1 * np.ones(local_scan.shape[0])
            except ValueError:
                local_scan = dataframe
                gt_labels = -1 * np.ones(local_scan.shape[0])
                timestamps = np.zeros(local_scan.shape[0])
        return local_scan.reshape(-1, 3), timestamps.reshape(-1), gt_labels.reshape(-1)

    def _run_evaluation(self):
        if self.has_gt:
            self.results.eval_odometry(self.odometry.get_poses(), self.gt_poses)
        self.results.eval_mos(self.confusion_matrix_belief, desc="\nBelief")
        self.results.eval_fps(self.times_mos, desc="\nAverage Frequency MOS")
        self.results.eval_fps(self.times_belief, desc="Average Frequency Belief")
