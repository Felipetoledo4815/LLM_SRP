import os
from dotenv import load_dotenv

load_dotenv()

# Nuscenes mini
nuscenes_mini = {
    "root_folder": os.getenv("NUSCENES_MINI"),
    "version": "v1.0-mini",
    "verbose": False,
    "field_of_view": 180
}

# Nuscenes full
nuscenes = {
    "root_folder": os.getenv("NUSCENES"),
    "version": "v1.0-trainval",
    "verbose": False,
    "field_of_view": 66.4
}

#kitti
kitti = {
    "root_folder": os.getenv("KITTI"),
    "version": "image_03",
    "verbose": False,
    "field_of_view": 90
}
kitti_tracking = {
    "root_folder": os.getenv("KITTI_TRACKING"),
    "version": "data_tracking_image_2",
    "verbose": False,
    "field_of_view": 90
}

