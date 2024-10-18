# Download datasets

## NuScenes
Download Nuscenes from https://www.nuscenes.org/nuscenes. You will need to create an account. To get started just download the mini partition and set it up by following: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_tutorial.ipynb.

## Waymo
Download Waymo v1.4.3 from https://waymo.com/open/download/. You will need to create an account. Waymo version 1.4.3 is the latest version with tfrecords files. The reason why we are using this version and not version 2.0.1 with .parket files is twofold: 1) there is more documnetation on how to use this version of the API, and 2) files are easier to work with.

To get started, you can download just one of the tfrecords files from [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_3/individual_files/training?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&authuser=1). After downloading it, you will need to execute the following command to split the tfrecords file into separated images and labels.

```bash
python ./dataset/parse_waymo_dataset.py --dataset_path {ABSOLUTE_PATH_TO_FOLDER_CONTAINING_TFRECORDS} --output_path {ABSOLUTE_PATH_TO_OUTPUT_FOLDER}
```
Additionally, you can specify the `--remove_tfrecords` flag to remove the tfrecords files after processing them in case you would like to save some HDD space.

This script will fill up the output_folder with jpg images and pkl labels, that then will be read by the WaymoDataset class.

If you wish to download the whole dataset, you can do so by downloading the full training and test splits into 2 separate folders, and then running the above command for each of the folders containing the tfrecords. 

**Note:** After downloading one file or the entire training or testing split, you will need to add `WAYMO_{SPLIT}` environment variable to your `.env` file, pointing to the "output" folder where the images and labels are stored.

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