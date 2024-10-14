# Download datasets

## NuScenes
Download Nuscenes from https://www.nuscenes.org/nuscenes. You will need to create an account. To get started just download the mini partition and set it up by following: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_tutorial.ipynb.

## KITTI
[Kitti](https://www.cvlibs.net/datasets/kitti/) takes advantage of their autonomous driving platform "Annieway" to develop novel challenging real-world computer vision benchmarks. 
For our interest we are utilizing the [kitti 3d object detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset.

We are using the left color camera images for our operations. The following files are required from the kitti 3d object detection dataset
- Left color images of object data set (12 GB)
- Camera calibration matrices of object data set (16 MB)
- Training labels of object data set (5 MB)

We are only using the training set of the above mentioned database since it has labels. Once data is downloaded, extract all .zip files in a folder, e.g., "kitti_3d".

```bash
mkdir -p {ABSOLUTE_PATH}/kitti_3d/
unzip data_object_label_2.zip -d ./data_object_label_2
unzip data_object_calib.zip -d ./data_object_calib
unzip data_object_image_2.zip -d ./data_object_image_2
```

Then add `KITTI_3D_DATASET` environment variable your `.env` file, pointing to the "kitti_3d" folder where extracted all the files.

```bash
KITTI_3D_DATASET={ABSOLUTE_PATH}/kitti_3d/
```

<!-- TODO: Add 'how-to-test' for all datasets.

To inspect the results from the kitti dataset, run the following command

```bash
python -m dataset.utils.tests.kitti_scene_plot
```

To just test the kitti dataset run the following command

```bash
python -m dataset.tests.run_kitti
```

To get the 3d plot of the vehicles along with the scene plot run
```bash
python -m dataset.utils.tests.kitti_with_3d_plot
``` -->