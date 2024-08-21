# MIT License
#
# Copyright (c) 2024 Benedikt Mersch, Luca Lobefaro, Ignazio Vizzo, Tiziano Guadagnino
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
import datetime
import importlib
import os
from abc import ABC

import numpy as np

# Button names
START_BUTTON = " START\n[SPACE]"
PAUSE_BUTTON = " PAUSE\n[SPACE]"
NEXT_FRAME_BUTTON = "NEXT FRAME\n\t\t [N]"
SCREENSHOT_BUTTON = "SCREENSHOT\n\t\t  [S]"
LOCAL_VIEW_BUTTON = "LOCAL VIEW\n\t\t [G]"
GLOBAL_VIEW_BUTTON = "GLOBAL VIEW\n\t\t  [G]"
CENTER_VIEWPOINT_BUTTON = "CENTER VIEWPOINT\n\t\t\t\t[C]"
QUIT_BUTTON = "QUIT\n  [Q]"

# Colors
BACKGROUND_COLOR = [0.0, 0.0, 0.0]
FRAME_COLOR = [0.8470, 0.1058, 0.3764]
MAP_COLOR = [0.0, 0.3019, 0.2509]

# Size constants
FRAME_PTS_SIZE = 0.06
MAP_PTS_SIZE = 0.08


class StubVisualizer(ABC):
    def __init__(self):
        pass

    def update(
        self,
        scan_points,
        map_points,
        pred_labels_scan,
        pred_labels_map,
        belief_labels_scan,
        belief_labels_map,
        pose,
    ):
        pass


class MapMOSVisualizer(StubVisualizer):
    # Public Interface ----------------------------------------------------------------------------
    def __init__(self):
        try:
            self._ps = importlib.import_module("polyscope")
            self._gui = self._ps.imgui
        except ModuleNotFoundError as err:
            print(f'polyscope is not installed on your system, run "pip install polyscope"')
            exit(1)

        # Initialize GUI controls
        self._background_color = BACKGROUND_COLOR
        self._frame_size = FRAME_PTS_SIZE
        self._map_size = MAP_PTS_SIZE
        self._block_execution = True
        self._play_mode = False
        self._toggle_frame = True
        self._toggle_map = True
        self._global_view = False
        self._last_pose = np.eye(4)

        # Initialize Visualizer
        self._initialize_visualizer()

    def update(
        self,
        scan_points,
        map_points,
        pred_labels_scan,
        pred_labels_map,
        belief_labels_scan,
        belief_labels_map,
        pose,
    ):
        self._update_geometries(
            scan_points,
            map_points,
            pred_labels_scan,
            pred_labels_map,
            belief_labels_scan,
            belief_labels_map,
            pose,
        )
        while self._block_execution:
            self._ps.frame_tick()
            if self._play_mode:
                break
        self._block_execution = not self._block_execution

    # Private Interface ---------------------------------------------------------------------------
    def _initialize_visualizer(self):
        self._ps.set_program_name("MapMOS Visualizer")
        self._ps.init()
        self._ps.set_ground_plane_mode("none")
        self._ps.set_background_color(BACKGROUND_COLOR)
        self._ps.set_verbosity(0)
        self._ps.set_user_callback(self._main_gui_callback)
        self._ps.set_build_default_gui_panels(False)

    def _update_geometries(
        self,
        scan_points,
        map_points,
        pred_labels_scan,
        pred_labels_map,
        belief_labels_scan,
        belief_labels_map,
        pose,
    ):
        scan_cloud = self._ps.register_point_cloud(
            "scan",
            scan_points,
            point_render_mode="quad",
        )

        scan_colors = np.zeros((len(pred_labels_scan), 3))
        scan_colors[pred_labels_scan == 1, :] = [1, 0, 0]
        scan_cloud.set_radius(self._frame_size, relative=False)
        scan_cloud.add_color_quantity("colors", scan_colors, enabled=True)
        if self._global_view:
            scan_cloud.set_transform(pose)
        else:
            scan_cloud.set_transform(np.eye(4))
        scan_cloud.set_enabled(self._toggle_frame)

        map_cloud = self._ps.register_point_cloud(
            "map",
            map_points,
            point_render_mode="quad",
        )
        map_colors = np.zeros((len(pred_labels_map), 3))
        map_colors[pred_labels_map == 1, :] = [1, 0, 0]
        map_cloud.set_radius(self._map_size, relative=False)
        map_cloud.add_color_quantity("colors", map_colors, enabled=True)
        if self._global_view:
            map_cloud.set_transform(np.eye(4))
        else:
            map_cloud.set_transform(np.linalg.inv(pose))
        map_cloud.set_enabled(self._toggle_map)

        self._last_pose = pose

    # GUI Callbacks ---------------------------------------------------------------------------
    def _start_pause_callback(self):
        button_name = PAUSE_BUTTON if self._play_mode else START_BUTTON
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Space):
            self._play_mode = not self._play_mode

    def _next_frame_callback(self):
        if self._gui.Button(NEXT_FRAME_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
            self._block_execution = not self._block_execution

    def _screenshot_callback(self):
        if self._gui.Button(SCREENSHOT_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_S):
            image_filename = "mapmos_" + (
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
            )
            self._ps.screenshot(image_filename)

    def _center_viewpoint_callback(self):
        if self._gui.Button(CENTER_VIEWPOINT_BUTTON) or self._gui.IsKeyPressed(
            self._gui.ImGuiKey_C
        ):
            self._ps.reset_camera_to_home_view()

    def _toggle_buttons_andslides_callback(self):
        # FRAME
        changed, self._frame_size = self._gui.SliderFloat(
            "##frame_size", self._frame_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("scan").set_radius(self._frame_size, relative=False)
        self._gui.SameLine()
        changed, self._toggle_frame = self._gui.Checkbox("Frame Cloud", self._toggle_frame)
        if changed:
            self._ps.get_point_cloud("scan").set_enabled(self._toggle_frame)

        # MAP
        changed, self._map_size = self._gui.SliderFloat(
            "##map_size", self._map_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("map").set_radius(self._map_size, relative=False)
        self._gui.SameLine()
        changed, self._toggle_map = self._gui.Checkbox("Local Map", self._toggle_map)
        if changed:
            self._ps.get_point_cloud("map").set_enabled(self._toggle_map)

    def _background_color_callback(self):
        changed, self._background_color = self._gui.ColorEdit3(
            "Background Color",
            self._background_color,
        )
        if changed:
            self._ps.set_background_color(self._background_color)

    def _global_view_callback(self):
        button_name = LOCAL_VIEW_BUTTON if self._global_view else GLOBAL_VIEW_BUTTON
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_G):
            self._global_view = not self._global_view
            if self._global_view:
                self._ps.get_point_cloud("scan").set_transform(self._last_pose)
                self._ps.get_point_cloud("map").set_transform(np.eye(4))
            else:
                self._ps.get_point_cloud("scan").set_transform(np.eye(4))
                self._ps.get_point_cloud("map").set_transform(np.linalg.inv(self._last_pose))
            self._ps.reset_camera_to_home_view()

    def _quit_callback(self):
        self._gui.SetCursorPosX(
            self._gui.GetCursorPosX() + self._gui.GetContentRegionAvail()[0] - 50
        )
        if (
            self._gui.Button(QUIT_BUTTON)
            or self._gui.IsKeyPressed(self._gui.ImGuiKey_Escape)
            or self._gui.IsKeyPressed(self._gui.ImGuiKey_Q)
        ):
            print("Destroying Visualizer")
            self._ps.unshow()
            os._exit(0)

    def _main_gui_callback(self):
        # GUI callbacks
        self._start_pause_callback()
        if not self._play_mode:
            self._gui.SameLine()
            self._next_frame_callback()
        self._gui.SameLine()
        self._screenshot_callback()
        self._gui.Separator()
        self._toggle_buttons_andslides_callback()
        self._background_color_callback()
        self._global_view_callback()
        self._gui.SameLine()
        self._center_viewpoint_callback()
        self._gui.Separator()
        self._quit_callback()
