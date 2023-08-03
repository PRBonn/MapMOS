<div align="center">
  <h1>Building Volumetric Beliefs for Dynamic Environments Exploiting Map-Based Moving Object Segmentation</h1>
  <a href="https://github.com/PRBonn/MapMOS#how-to-use-it"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/PRBonn/MapMOS#installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/mersch2023ral.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://lbesson.mit-license.org/"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>

<p>
  <img src="https://github.com/PRBonn/MapMOS/assets/38326482/cd594591-8c2c-41f0-8412-cd5d1d2fd7d4" width="500"/>
</p>

<p>
  <i>Our approach identifies moving objects in the current scan (blue points) and the local map (black points) of the environment and maintains a volumetric belief map representing the dynamic environment.</i>
</p>

<details>
<summary><b>Click here for qualitative results!</b></summary>
  
[![MapMOS](https://github.com/PRBonn/MapMOS/assets/38326482/a4238431-bd2d-4b2c-991b-7ff5e9378a8e)](https://github.com/PRBonn/MapMOS/assets/38326482/04c7e5a2-dd44-431a-95b0-c42d5605078a)

 <i>Our predictions for the KITTI Tracking sequence 19 with true positives (green), false positives (red), and false negatives (blue).</i>

</details>


</div>


## Installation
First, make sure the [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) is installed on your system, see [here](https://github.com/NVIDIA/MinkowskiEngine#installation) for more details.

Next, clone our repository
```bash
git clone git@github.com:PRBonn/MapMOS && cd MapMOS
```

and install with
```bash
make install
```

**or**
```bash
make install-all
```
if you want to install the project with all optional dependencies (needed for the visualizer). In case you want to edit the Python code, install in editable mode:
```bash
make editable
```

## How to Use It
Just type

```bash
mapmos_pipeline --help
```
to see how to run MapMOS. 
<details>
<summary>This is what you should see</summary>

![Screenshot from 2023-08-03 13-07-14](https://github.com/PRBonn/MapMOS/assets/38326482/c769afa6-709d-4648-b42d-11092d5b92ac)

</details>

Check the [Download](#downloads) section for a pre-trained model. Like [KISS-ICP](https://github.com/PRBonn/kiss-icp), our pipeline runs on a variety of point cloud data formats like `bin`, `pcd`, `ply`, `xyz`, `rosbags`, and more. To visualize these, just type 

```bash
mapmos_pipeline --visualize /path/to/weights.ckpt /path/to/data
```

<details>
<summary>Want to evaluate with ground truth labels?</summary>

Because these lables come in all shapes, you need to specify a dataloader. This is currently available for SemanticKITTI and NuScenes as well as our post-processed KITTI Tracking sequence 19 and Apollo sequences (see [Downloads](#downloads)).

</details>

<details>
<summary>Want to reproduce the results from the paper?</summary>
For reproducing the results of the paper, you need to pass the corresponding config file. They will make sure that the de-skewing option and the maximum range are set properly. To compare different map fusion strategies from our paper, just pass the `--paper` flag to the `mapmos_pipeline`.

</details>


## Training
To train our approach, you need to first cache your data. To see how to do that, just `cd` into the `MapMOS` repository and type

```bash
python3 scripts/precache.py --help
```

After this, you can run the training script. Again, `--help` shows you how:
```bash
python3 scripts/train.py --help
```

<details>
<summary>Want to verify the cached data?</summary>

You can inspect the cached training samples by using the script `python3 scripts/cache_to_ply.py --help`.

</details>

<details>
<summary>Want to change the logging directory?</summary>

The training log and checkpoints will be saved by default to the current working directory. To change that, export the `export LOGS=/your/path/to/logs` environment variable before running the training script.

</details>


## Downloads
You can download the post-processed and labeled [Apollo dataset](https://www.ipb.uni-bonn.de/html/projects/apollo_dataset/LiDAR-MOS.zip) and [KITTI Tracking sequence 19](https://www.ipb.uni-bonn.de/html/projects/kitti-tracking/post-processed/kitti-tracking.zip) from our website.

The [weights](https://www.ipb.uni-bonn.de/html/projects/MapMOS/mapmos.ckpt) of our pre-trained model can be downloaded as well.

## Publication
If you use our code in your academic work, please cite the corresponding [paper](http://www.ipb.uni-bonn.de/pdfs/mersch2023ral.pdf):

```bibtex
@article{mersch2023ral,
  author = {B. Mersch and T. Guadagnino and X. Chen and I. Vizzo and J. Behley and C. Stachniss},
  title = {{Building Volumetric Beliefs for Dynamic Environments Exploiting Map-Based Moving Object Segmentation}},
  journal = {IEEE Robotics and Automation Letters (RA-L)},
  volume = {8},
  number = {8},
  pages = {5180--5187},
  year = {2023},
  issn = {2377-3766},
  doi = {10.1109/LRA.2023.3292583},
  codeurl = {https://github.com/PRBonn/MapMOS},
}
```

## Acknowledgments
This implementation is heavily inspired by [KISS-ICP](https://github.com/PRBonn/kiss-icp).

## License
This project is free software made available under the MIT License. For details see the LICENSE file.
