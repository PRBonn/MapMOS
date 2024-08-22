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
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modifed and drastically simplified, but if you are using this module you
// should at least acknoowledge the work from CT-ICP by giving a star on GitHub
#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

namespace mapmos {
struct Belief {
    void update() {
        value_ += this->num > 0 ? this->sum / this->num : 0.0;
        this->sum = 0.0;
        this->num = 0;
    }
    void accumulatePartialUpdate(const double &update) {
        this->sum += update;
        this->num++;
    }
    double value_ = 0.0;

protected:
    double sum = 0.0;
    int num = 0;
};

struct VoxelHashMap {
    using Voxel = Eigen::Vector3i;
    struct VoxelBlock {
        // buffer of points with a max limit of n_points
        std::vector<Eigen::Vector3d> points;
        std::vector<int> timestamps;
        Belief belief;
        int num_points_;
        inline void AddPoint(const Eigen::Vector3d &point, int timestamp) {
            if (points.size() < static_cast<size_t>(num_points_)) {
                points.push_back(point);
                timestamps.push_back(timestamp);
            }
        }
    };
    struct VoxelHash {
        size_t operator()(const Voxel &voxel) const {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
        }
    };

    explicit VoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          max_points_per_voxel_(max_points_per_voxel) {}

    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }
    inline Voxel PointToVoxel(const Eigen::Vector3d &point) const {
        return Voxel(static_cast<int>(std::floor(point.x() / voxel_size_)),
                     static_cast<int>(std::floor(point.y() / voxel_size_)),
                     static_cast<int>(std::floor(point.z() / voxel_size_)));
    }
    void Update(const std::vector<Eigen::Vector3d> &points,
                const Eigen::Vector3d &origin,
                const int timestamp);
    void Update(const std::vector<Eigen::Vector3d> &points,
                const Sophus::SE3d &pose,
                const int timestamp);
    void AddPoints(const std::vector<Eigen::Vector3d> &points, const int timestamp);
    std::vector<double> GetBelief(const std::vector<Eigen::Vector3d> &points) const;
    void UpdateBelief(const std::vector<Eigen::Vector3d> &points,
                      const std::vector<double> &updates);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<Eigen::Vector3d> Pointcloud() const;
    std::tuple<std::vector<Eigen::Vector3d>, std::vector<int>> PointcloudWithTimestamps() const;
    std::tuple<std::vector<Voxel>, std::vector<double>> VoxelsWithBelief() const;
    std::vector<Eigen::Vector3d> GetPoints(const std::vector<Voxel> &query_voxels) const;

    double voxel_size_;
    double max_distance_;
    int max_points_per_voxel_;
    tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_;
};
}  // namespace mapmos
