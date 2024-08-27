// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "VoxelHashMap.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>
namespace mapmos {

std::vector<Eigen::Vector3d> VoxelHashMap::GetPoints(const std::vector<Voxel> &query_voxels) const {
    std::vector<Eigen::Vector3d> points;
    points.reserve(query_voxels.size() * static_cast<size_t>(max_points_per_voxel_));
    std::for_each(query_voxels.cbegin(), query_voxels.cend(), [&](const auto &query) {
        auto search = map_.find(query);
        if (search != map_.end()) {
            for (const auto &point : search->second.points) {
                points.emplace_back(point);
            }
        }
    });
    return points;
}

std::vector<Eigen::Vector3d> VoxelHashMap::Pointcloud() const {
    std::vector<Eigen::Vector3d> points;
    points.reserve(max_points_per_voxel_ * map_.size());
    for (const auto &[voxel, voxel_block] : map_) {
        (void)voxel;
        for (const auto &point : voxel_block.points) {
            points.push_back(point);
        }
    }
    return points;
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<int>> VoxelHashMap::PointcloudWithTimestamps()
    const {
    std::vector<Eigen::Vector3d> points;
    std::vector<int> timestamps;
    points.reserve(max_points_per_voxel_ * map_.size());
    for (const auto &[voxel, voxel_block] : map_) {
        (void)voxel;
        for (const auto &point : voxel_block.points) {
            points.push_back(point);
        }
        for (const auto &timestamp : voxel_block.timestamps) {
            timestamps.push_back(timestamp);
        }
    }
    return std::make_tuple(points, timestamps);
}

std::tuple<std::vector<VoxelHashMap::Voxel>, std::vector<double>> VoxelHashMap::VoxelsWithBelief()
    const {
    std::vector<Voxel> voxels;
    std::vector<double> belief;
    for (auto map_element : map_) {
        voxels.push_back(map_element.first);
        belief.push_back(map_element.second.belief.value_);
    }
    return make_tuple(voxels, belief);
}

void VoxelHashMap::Update(const std::vector<Eigen::Vector3d> &points,
                          const Eigen::Vector3d &origin,
                          const int timestamp) {
    AddPoints(points, timestamp);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const std::vector<Eigen::Vector3d> &points,
                          const Sophus::SE3d &pose,
                          const int timestamp) {
    std::vector<Eigen::Vector3d> points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return pose * point; });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin, timestamp);
}

void VoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d> &points, const int timestamp) {
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        const auto voxel = PointToVoxel(point);
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point, timestamp);
        } else {
            map_.insert(
                {voxel, VoxelBlock{{point}, {timestamp}, {Belief{}}, max_points_per_voxel_}});
        }
    });
}

void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    const auto max_distance2 = max_distance_ * max_distance_;
    for (auto it = map_.begin(); it != map_.end();) {
        const auto &[voxel, voxel_block] = *it;
        Eigen::Vector3d pt(voxel[0] * voxel_size_, voxel[1] * voxel_size_, voxel[2] * voxel_size_);
        if ((pt - origin).squaredNorm() >= (max_distance2)) {
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}

void VoxelHashMap::UpdateBelief(const std::vector<Eigen::Vector3d> &points,
                                const std::vector<double> &updates) {
    std::vector<Voxel> voxels_to_update;
    voxels_to_update.reserve(points.size());
    for (size_t i = 0; i < points.size(); i++) {
        auto voxel = PointToVoxel(points[i]);
        voxels_to_update.emplace_back(voxel);
        map_[voxel].belief.accumulatePartialUpdate(updates[i]);
    }

    tbb::parallel_for_each(voxels_to_update.cbegin(), voxels_to_update.cend(),
                           [this](const auto &voxel) { map_[voxel].belief.update(); });
}

std::vector<double> VoxelHashMap::GetBelief(const std::vector<Eigen::Vector3d> &points) const {
    std::vector<double> beliefs(points.size(), 0.0);
    std::transform(points.cbegin(), points.cend(), beliefs.begin(), [this](const auto &p) {
        auto voxel = PointToVoxel(p);
        if (map_.contains(voxel)) {
            return map_.at(voxel).belief.value_;
        }
        return 0.0;
    });
    return beliefs;
}

}  // namespace mapmos
