import os
from dotenv import load_dotenv

load_dotenv()

# Nuscenes mini
nuscenes_mini = {
    "root_folder": os.getenv("NUSCENES_MINI"),
    "version": "v1.0-mini",
    "verbose": False,
    "field_of_view": 66.4
}

# Nuscenes full
nuscenes = {
    "root_folder": os.getenv("NUSCENES"),
    "version": "v1.0-trainval",
    "verbose": False,
    "field_of_view": 66.4
}

# Kitti
kitti = {
    "root_folder": os.getenv("KITTI_3D_DATASET"),
    "version": "data_object_image_2",
    "verbose": False,
    "field_of_view": 90
}

# Waymo validation
waymo_validation = {
    "root_folder": os.getenv("WAYMO_VAL"),
    "field_of_view": 50.4   # https://arxiv.org/pdf/1912.04838
}
