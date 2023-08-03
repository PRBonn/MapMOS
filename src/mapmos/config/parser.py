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
# NOTE: This module was contributed by Markus Pielmeier on PR #63
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseSettings, PrivateAttr

from mapmos.config.config import (
    DataConfig,
    OdometryConfig,
    MOSConfig,
    TrainingConfig,
)


class MapMOSConfig(BaseSettings):
    out_dir: str = "results"
    data: DataConfig = DataConfig()
    odometry: OdometryConfig = OdometryConfig()
    mos: MOSConfig = MOSConfig()
    training: TrainingConfig = TrainingConfig()
    _config_file: Optional[Path] = PrivateAttr()

    def __init__(self, config_file: Optional[Path] = None, *args, **kwargs):
        self._config_file = config_file
        super().__init__(*args, **kwargs)

    def _yaml_source(self) -> Dict[str, Any]:
        data = None
        if self._config_file is not None:
            with open(self._config_file) as cfg_file:
                data = yaml.safe_load(cfg_file)
        return data or {}

    class Config:
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return init_settings, MapMOSConfig._yaml_source


def load_config(config_file: Optional[Path]) -> MapMOSConfig:
    """Load configuration from an Optional yaml file."""
    config = MapMOSConfig(config_file=config_file)
    return config


def write_config(config: MapMOSConfig, filename: str):
    with open(filename, "w") as outfile:
        yaml.dump(config.dict(), outfile, default_flow_style=False)
