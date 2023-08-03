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

namespace {
struct ResultTuple {
    ResultTuple(std::size_t n) {
        source.reserve(n);
        target.reserve(n);
    }
    std::vector<Eigen::Vector3d> source;
    std::vector<Eigen::Vector3d> target;
};
}  // namespace

namespace mapmos {

VoxelHashMap::Vector3dVectorTuple VoxelHashMap::GetCorrespondences(
    const Vector3dVector &points, double max_correspondance_distance) const {
    // Lambda Function to obtain the KNN of one point, maybe refactor
    auto GetClosestNeighboor = [&](const Eigen::Vector3d &point) {
        auto kx = static_cast<int>(point[0] / voxel_size_);
        auto ky = static_cast<int>(point[1] / voxel_size_);
        auto kz = static_cast<int>(point[2] / voxel_size_);
        std::vector<Voxel> voxels;
        voxels.reserve(27);
        for (int i = kx - 1; i < kx + 1 + 1; ++i) {
            for (int j = ky - 1; j < ky + 1 + 1; ++j) {
                for (int k = kz - 1; k < kz + 1 + 1; ++k) {
                    voxels.emplace_back(i, j, k);
                }
            }
        }

        using Vector3dVector = std::vector<Eigen::Vector3d>;
        Vector3dVector neighboors;
        neighboors.reserve(27 * max_points_per_voxel_);
        std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel) {
            auto search = map_.find(voxel);
            if (search != map_.end()) {
                const auto &points = search->second.points;
                if (!points.empty()) {
                    for (const auto &point : points) {
                        neighboors.emplace_back(point);
                    }
                }
            }
        });

        Eigen::Vector3d closest_neighbor;
        double closest_distance2 = std::numeric_limits<double>::max();
        std::for_each(neighboors.cbegin(), neighboors.cend(), [&](const auto &neighbor) {
            double distance = (neighbor - point).squaredNorm();
            if (distance < closest_distance2) {
                closest_neighbor = neighbor;
                closest_distance2 = distance;
            }
        });

        return closest_neighbor;
    };
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    const auto [source, target] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        // Identity
        ResultTuple(points.size()),
        // 1st lambda: Parallel computation
        [max_correspondance_distance, &GetClosestNeighboor](
            const tbb::blocked_range<points_iterator> &r, ResultTuple res) -> ResultTuple {
            auto &[src, tgt] = res;
            src.reserve(r.size());
            tgt.reserve(r.size());
            for (const auto &point : r) {
                Eigen::Vector3d closest_neighboors = GetClosestNeighboor(point);
                if ((closest_neighboors - point).norm() < max_correspondance_distance) {
                    src.emplace_back(point);
                    tgt.emplace_back(closest_neighboors);
                }
            }
            return res;
        },
        // 2nd lambda: Parallel reduction
        [](ResultTuple a, const ResultTuple &b) -> ResultTuple {
            auto &[src, tgt] = a;
            const auto &[srcp, tgtp] = b;
            src.insert(src.end(),  //
                       std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
            tgt.insert(tgt.end(),  //
                       std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
            return a;
        });

    return std::make_tuple(source, target);
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

void VoxelHashMap::Update(const Vector3dVector &points,
                          const Eigen::Vector3d &origin,
                          const int timestamp) {
    AddPoints(points, timestamp);
    RemoveFarAwayPoints(origin);
}

void VoxelHashMap::Update(const Vector3dVector &points,
                          const Sophus::SE3d &pose,
                          const int timestamp) {
    Vector3dVector points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return pose * point; });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin, timestamp);
}

void VoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d> &points, const int timestamp) {
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        auto voxel = Voxel((point / voxel_size_).template cast<int>());
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point, timestamp);
        } else {
            map_.insert(
                {voxel, VoxelBlock{{point}, {timestamp}, {Belief()}, max_points_per_voxel_}});
        }
    });
}

void VoxelHashMap::UpdateBelief(const Vector3dVector &points, const std::vector<double> &updates) {
    std::vector<Voxel> voxels_to_update;
    voxels_to_update.reserve(points.size());

    for (size_t i = 0; i < points.size(); i++) {
        auto voxel = Voxel((points[i] / voxel_size_).template cast<int>());
        voxels_to_update.emplace_back(voxel);
        map_[voxel].belief.accumulatePartialUpdate(updates[i]);
    }
    tbb::parallel_for_each(voxels_to_update.cbegin(), voxels_to_update.cend(),
                           [this](const auto &voxel) { map_[voxel].belief.update(); });
}

std::vector<double> VoxelHashMap::GetBelief(const Vector3dVector &points) const {
    std::vector<double> beliefs(points.size(), 0.0);
    std::transform(points.cbegin(), points.cend(), beliefs.begin(), [this](const auto &p) {
        auto voxel = Voxel((p / voxel_size_).template cast<int>());
        if (map_.contains(voxel)) {
            return map_.at(voxel).belief.value_;
        }
        return 0.0;
    });
    return beliefs;
}
void VoxelHashMap::RemoveFarAwayPoints(const Eigen::Vector3d &origin) {
    for (const auto &[voxel, voxel_block] : map_) {
        Eigen::Vector3d pt(voxel[0] * voxel_size_, voxel[1] * voxel_size_, voxel[2] * voxel_size_);
        const auto max_distance2 = max_distance_ * max_distance_;
        if ((pt - origin).squaredNorm() > (max_distance2)) {
            map_.erase(voxel);
        }
    }
}
}  // namespace mapmos
