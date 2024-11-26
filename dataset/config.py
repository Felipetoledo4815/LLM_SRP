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

open_lane_v1 = {
    "root_folder": os.getenv("OPEN_LANE_V1"),
    "field_of_view": 50.4
}

# Kitti training
kitti_training = {
    "root_folder": os.getenv("KITTI"),
    "partition": "training",
    "field_of_view": 90
}

# Waymo training
waymo_training = {
    "root_folder": os.getenv("WAYMO_TRAIN"),
    "field_of_view": 50.4   # https://arxiv.org/pdf/1912.04838
}

# Waymo validation
waymo_validation = {
    "root_folder": os.getenv("WAYMO_VAL"),
    "field_of_view": 50.4   # https://arxiv.org/pdf/1912.04838
}
