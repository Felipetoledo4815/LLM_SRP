# LLM_SRP

## Download datasets
Please follow the instructions in this [README](./dataset/README.md).

## Getting started
1. Create a local `.env` file on the root of the repository containing the following variables:
```bash
NUSCENES_MINI=/absolute_path*/to/nuscenes_mini/
NUSCENES=/absolute_path*/to/nuscenes/
```
**Note that you need to specify the absolute path to the dataset.*

2. Create a conda environment with the following command:
```bash
conda create -n llm_srp python=3.10
```
3. Activate the conda environment:
```bash
conda activate llm_srp
```
4. Install all packages in the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Exploring the dataset
- If you want to have a quick overview of the dataset, you can use the jupyter-notebook called `explore_llm_srp.ipynb`.
- To execute minimal versions of different scripts in this repository, execute the following command from the root of the repository:
```bash
python -m dataset.utils.tests.run_scene_plot
```

## Installing the package
If you want to install the package to use it in other projects, you can run the following command from the root of the repository:
```bash
pip install -e .
```

## Exploring Kitti dataset with LLM SRP
[Kitti](https://www.cvlibs.net/datasets/kitti/) takes advantage of their autonomous driving platform Annieway to develop novel challenging real-world computer vision benchmarks. 
 For our interest we are utilizing the [kitti 3d object detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset.

We are using the left color camera images for our operations. The following files shall be required from the kitti 3d object detection dataset
1. Left color images of object data set (12 GB)
2. Camera calibration matrices of object data set (16 MB)
3. Training labels of object data set (5 MB)

We are only using the train portion of the above mentioned database. Once data is downloaded follow the given steps

Add the following line to your `.env` file

```commandline
KITTI_3D_DATASET=/absolute_path*/path/to/kitti_3d_object/
```

Now, extract the previously mentioned file in the `KITTI_3D_DATASET` directory from the `.env` file.

To inspect the results from the kitti dataset, run the following command

```commandline
python -m dataset.utils.tests.kitti_scene_plot
```

To just test the kitti dataset run the following command

```commandline
python -m dataset.tests.run_kitti
```
To get the 3d plot of the vehicles along with the scene plot run
```commandline
python -m dataset.utils.tests.kitti_with_3d_plot
```