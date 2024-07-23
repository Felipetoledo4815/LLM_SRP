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
