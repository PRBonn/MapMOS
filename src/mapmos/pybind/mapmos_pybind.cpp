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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "Registration.hpp"
#include "VoxelHashMap.hpp"
#include "stl_vector_eigen.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

namespace mapmos {
PYBIND11_MODULE(mapmos_pybind, m) {
    auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_Vector3dVector", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    // Map representation
    py::class_<VoxelHashMap> internal_map(m, "_VoxelHashMap", "Don't use this");
    internal_map
        .def(py::init<double, double, int>(), "voxel_size"_a, "max_distance"_a,
             "max_points_per_voxel"_a)
        .def("_clear", &VoxelHashMap::Clear)
        .def("_empty", &VoxelHashMap::Empty)
        .def("_update",
             py::overload_cast<const VoxelHashMap::Vector3dVector &, const Eigen::Vector3d &,
                               const int>(&VoxelHashMap::Update),
             "points"_a, "origin"_a, "timestamp"_a)
        .def(
            "_update",
            [](VoxelHashMap &self, const VoxelHashMap::Vector3dVector &points,
               const Eigen::Matrix4d &T, const int timestamp) {
                Sophus::SE3d pose(T);
                self.Update(points, pose, timestamp);
            },
            "points"_a, "pose"_a, "timestamp"_a)
        .def("_add_points", &VoxelHashMap::AddPoints, "points"_a, "timestamp"_a)
        .def("_remove_far_away_points", &VoxelHashMap::RemoveFarAwayPoints, "origin"_a)
        .def("_point_cloud", &VoxelHashMap::Pointcloud)
        .def("_point_cloud_with_timestamps", &VoxelHashMap::PointcloudWithTimestamps)
        .def("_update_belief", &VoxelHashMap::UpdateBelief, "points"_a, "updates"_a)
        .def("_get_belief", &VoxelHashMap::GetBelief, "points"_a)
        .def("_get_correspondences", &VoxelHashMap::GetCorrespondences, "points"_a,
             "max_correspondance_distance"_a);

    // Point Cloud registration
    m.def(
        "_register_point_cloud",
        [](const std::vector<Eigen::Vector3d> &points, const VoxelHashMap &voxel_map,
           const Eigen::Matrix4d &T_guess, double max_correspondence_distance, double kernel) {
            Sophus::SE3d initial_guess(T_guess);
            return RegisterFrame(points, voxel_map, initial_guess, max_correspondence_distance,
                                 kernel)
                .matrix();
        },
        "points"_a, "voxel_map"_a, "initial_guess"_a, "max_correspondance_distance"_a, "kernel"_a);
}
}  // namespace mapmos
