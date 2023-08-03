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

import time
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from tqdm.auto import trange

from mapmos.mapping import VoxelHashMap
from mapmos.metrics import get_confusion_matrix
from mapmos.pipeline import MapMOSPipeline
from mapmos.utils.save import save_to_ply, save_to_kitti


class PaperPipeline(MapMOSPipeline):
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
        super().__init__(
            dataset=dataset,
            weights=weights,
            config=config,
            visualize=visualize,
            save_ply=save_ply,
            save_kitti=save_kitti,
            n_scans=n_scans,
            jump=jump,
        )
        self.belief_scan_only = VoxelHashMap(
            voxel_size=self.config.mos.voxel_size_belief,
            max_distance=self.config.mos.max_range_belief,
        )

        self.times_belief_scan_only = []

        self.confusion_matrix_mos = torch.zeros(2, 2)
        self.confusion_matrix_belief_scan_only = torch.zeros(2, 2)
        self.confusion_matrix_belief_no_delay = torch.zeros(2, 2)

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

            pred_labels_scan = (
                np.zeros_like(pred_logits_scan)
                if len(map_points) == 0
                else self.model.to_label(pred_logits_scan)
            )
            self.confusion_matrix_mos += get_confusion_matrix(
                torch.tensor(pred_labels_scan, dtype=torch.int32),
                torch.tensor(gt_labels, dtype=torch.int32),
            )

            # Probabilistic volumetric fusion with scan prediction only
            start_time = time.perf_counter_ns()
            self.belief_scan_only.update_belief(scan_points, pred_logits_scan)
            belief = self.belief_scan_only.get_belief(scan_points)
            self.times_belief_scan_only.append(time.perf_counter_ns() - start_time)
            belief_labels = (
                np.zeros_like(belief) if len(map_points) == 0 else self.model.to_label(belief)
            )
            self.confusion_matrix_belief_scan_only += get_confusion_matrix(
                torch.tensor(belief_labels, dtype=torch.int32),
                torch.tensor(gt_labels, dtype=torch.int32),
            )

            # Probabilistic volumetric fusion with scan and moving map predictions
            map_mask = pred_logits_map > 0
            points_stacked = np.vstack([scan_points, map_points[map_mask]])
            logits_stacked = np.vstack(
                [pred_logits_scan.reshape(-1, 1), pred_logits_map[map_mask].reshape(-1, 1)]
            ).reshape(-1)
            start_time = time.perf_counter_ns()
            self.belief.update_belief(points_stacked, logits_stacked)
            belief_with_map = self.belief.get_belief(scan_points)
            self.times_belief.append(time.perf_counter_ns() - start_time)
            belief_labels_with_map = (
                np.zeros_like(belief_with_map)
                if len(map_points) == 0
                else self.model.to_label(belief_with_map)
            )

            self.confusion_matrix_belief_no_delay += get_confusion_matrix(
                torch.tensor(belief_labels_with_map, dtype=torch.int32),
                torch.tensor(gt_labels, dtype=torch.int32),
            )

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

            # Probabilistic volumetric fusion with scan and moving map predictions and delay
            self.buffer.append([scan_index, scan_points, gt_labels])
            if len(self.buffer) == self.buffer.maxlen:
                query_index, query_points, query_labels = self.buffer.popleft()
                self.process_final_prediction(query_index, query_points, query_labels)

            # Clean up
            self.belief_scan_only.remove_voxels_far_from_location(self.odometry.current_location())
            self.belief.remove_voxels_far_from_location(self.odometry.current_location())

        # Clear buffer at the end
        while len(self.buffer) != 0:
            query_index, query_points, query_labels = self.buffer.popleft()
            self.process_final_prediction(query_index, query_points, query_labels)

    def _run_evaluation(self):
        if self.has_gt:
            self.results.eval_odometry(self.odometry.get_poses(), self.gt_poses)
        self.results.eval_mos(self.confusion_matrix_mos, desc="\nScan Prediction")
        self.results.eval_fps(self.times_mos, desc="Average Frequency MOS")
        self.results.eval_mos(self.confusion_matrix_belief_scan_only, desc="\nBelief, Scan Only")
        self.results.eval_fps(
            self.times_belief_scan_only, desc="Average Frequency Belief, Scan Only"
        )
        self.results.eval_mos(self.confusion_matrix_belief_no_delay, desc="\nBelief, No Delay ")
        self.results.eval_mos(
            self.confusion_matrix_belief,
            desc="\nBelief",
        )
        self.results.eval_fps(self.times_belief, desc="Average Frequency Belief")
