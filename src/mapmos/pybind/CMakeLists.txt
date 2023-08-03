# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
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
cmake_minimum_required(VERSION 3.16...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX VERSION "${SKBUILD_PROJECT_VERSION}")

# Setup build options
option(USE_SYSTEM_EIGEN3 "Use system pre-installed Eigen" ON)
option(USE_SYSTEM_SOPHUS "Use system pre-installed Sophus" ON)
option(USE_SYSTEM_TSLMAP "Use system pre-installed tsl_robin" ON)
option(USE_SYSTEM_TBB "Use system pre-installed oneAPI/tbb" ON)

# Set build type (repeat here for C++ only consumers)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(3rdparty/find_dependencies.cmake)

set(PYBIND11_NEWPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Python bindings
pybind11_add_module(mapmos_pybind MODULE mapmos_pybind.cpp VoxelHashMap.cpp Registration.cpp)
target_compile_features(mapmos_pybind PRIVATE cxx_std_17)
target_link_libraries(mapmos_pybind PUBLIC Eigen3::Eigen tsl::robin_map TBB::tbb Sophus::Sophus)
install(TARGETS mapmos_pybind LIBRARY DESTINATION .)
