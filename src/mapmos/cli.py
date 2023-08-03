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

# Check that the python bindings are properly built and can be loaded at runtime
try:
    from mapmos.pybind import mapmos_pybind
except Exception as e:
    print(80 * "*")
    print("[ERRROR] MapMOS Python bindings not available. Please build!")
    print(80 * "*")
    raise SystemExit()

import glob
import os
from pathlib import Path
from typing import Optional

import typer
from kiss_icp.tools.cmd import guess_dataloader
from mapmos.datasets import (
    available_dataloaders,
    jumpable_dataloaders,
    sequence_dataloaders,
    supported_file_extensions,
)


def name_callback(value: str):
    if not value:
        return value
    dl = available_dataloaders()
    if value not in dl:
        raise typer.BadParameter(f"Supported dataloaders are:\n{', '.join(dl)}")
    return value


app = typer.Typer(add_completion=False, rich_markup_mode="rich")

# Remove from the help those dataloaders we explicitly say how to use
_available_dl_help = available_dataloaders()
_available_dl_help.remove("generic")
_available_dl_help.remove("mcap")
_available_dl_help.remove("ouster")
_available_dl_help.remove("rosbag")
_available_dl_help.remove("mapmos_dataset")

docstring = f"""
MapMOS: Building Volumetric Beliefs for Dynamic Environments Exploiting Map-Based Moving Object Segmentation\n
\b
[bold green]Examples: [/bold green]
# Process all pointclouds in the given <data-dir> \[{", ".join(supported_file_extensions())}]
$ mapmos_pipeline --visualize <weights>:page_facing_up: <data-dir>:open_file_folder:

# Process a given [bold]ROS1/ROS2 [/bold]rosbag file (directory:open_file_folder:, ".bag":page_facing_up:, or "metadata.yaml":page_facing_up:)
$ mapmos_pipeline --visualize <weights>:page_facing_up: <path-to-my-rosbag>[:open_file_folder:/:page_facing_up:]

# Process [bold]mcap [/bold] recording
$ mapmos_pipeline --visualize <weights>:page_facing_up: <path-to-file.mcap>:page_facing_up:

# Process [bold]Ouster pcap[/bold] recording (requires ouster-sdk Python package installed)
$ mapmos_pipeline --visualize <weights>:page_facing_up: <path-to-ouster.pcap>:page_facing_up: \[--meta <path-to-metadata.json>:page_facing_up:]

# Use a more specific dataloader to also load GT moving labels and compute metrics: {", ".join(_available_dl_help)}
$ mapmos_pipeline --dataloader kitti --sequence 07 --visualize <weights>:page_facing_up: <path-to-kitti-root>:open_file_folder:
"""


@app.command(help=docstring)
def mapmos_pipeline(
    weights: Path = typer.Argument(
        ...,
        help="Path to the model's weights (.ckpt)",
        show_default=False,
    ),
    data: Path = typer.Argument(
        ...,
        help="The data directory used by the specified dataloader",
        show_default=False,
    ),
    dataloader: str = typer.Option(
        None,
        show_default=False,
        case_sensitive=False,
        autocompletion=available_dataloaders,
        callback=name_callback,
        help="[Optional] Use a specific dataloader from those supported by MapMOS",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        show_default=False,
        help="[Optional] Path to the configuration file",
    ),
    # Aditional Options ---------------------------------------------------------------------------
    visualize: bool = typer.Option(
        False,
        "--visualize",
        "-v",
        help="[Optional] Open an online visualization of the KISS-ICP pipeline",
        rich_help_panel="Additional Options",
    ),
    sequence: Optional[str] = typer.Option(
        None,
        "--sequence",
        "-s",
        show_default=False,
        help="[Optional] For some dataloaders, you need to specify a given sequence",
        rich_help_panel="Additional Options",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-t",
        show_default=False,
        help="[Optional] Only valid when processing rosbag files",
        rich_help_panel="Additional Options",
    ),
    save_ply: bool = typer.Option(
        False,
        "--save_ply",
        "-ply",
        help="[Optional] Save predictions to ply file",
        rich_help_panel="Additional Options",
    ),
    save_kitti: bool = typer.Option(
        False,
        "--save_kitti",
        "-kitti",
        help="[Optional] Save predictions to SemanticKITTI .lable file",
        rich_help_panel="Additional Options",
    ),
    n_scans: int = typer.Option(
        -1,
        "--n-scans",
        "-n",
        show_default=False,
        help="[Optional] Specify the number of scans to process, default is the entire dataset",
        rich_help_panel="Additional Options",
    ),
    jump: int = typer.Option(
        0,
        "--jump",
        "-j",
        show_default=False,
        help="[Optional] Specify if you want to start to process scans from a given starting point",
        rich_help_panel="Additional Options",
    ),
    meta: Optional[Path] = typer.Option(
        None,
        "--meta",
        "-m",
        exists=True,
        show_default=False,
        help="[Optional] For Ouster pcap dataloader, specify metadata json file path explicitly",
        rich_help_panel="Additional Options",
    ),
    paper: bool = typer.Option(
        False,
        "--paper",
        "-paper",
        help="[Optional] Use this to compare different belief fusion strategies as done in the original paper",
        rich_help_panel="Additional Options",
    ),
):
    # Attempt to guess some common file extensions to avoid using the --dataloader flag
    if not dataloader:
        dataloader, data = guess_dataloader(data, default_dataloader="generic")

    # Validate some options
    if dataloader in sequence_dataloaders() and sequence is None:
        print('You must specify a sequence "--sequence"')
        raise typer.Exit(code=1)

    if jump != 0 and dataloader not in jumpable_dataloaders():
        print(f"[WARNING] '{dataloader}' does not support '--jump', starting from first frame")
        jump = 0
    # Lazy-loading for faster CLI
    from mapmos.datasets import dataset_factory

    if paper:
        from mapmos.paper_pipeline import PaperPipeline as Pipeline
    else:
        from mapmos.pipeline import MapMOSPipeline as Pipeline

    Pipeline(
        dataset=dataset_factory(
            dataloader=dataloader,
            data_dir=data,
            # Additional options
            sequence=sequence,
            topic=topic,
            meta=meta,
        ),
        weights=weights,
        config=config,
        visualize=visualize,
        save_ply=save_ply,
        save_kitti=save_kitti,
        n_scans=n_scans,
        jump=jump,
    ).run().print()


def run():
    app()
