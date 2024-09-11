from typing import List, Dict
from dataset import config
from dataset.nuscenes_dataset import NuscenesDataset


DATASET_CLASSES = {
    "nuscenes": NuscenesDataset
}


class DatasetFactory:
    def __init__(self, datasets: List[str], configs: Dict[str, str] | None = None) -> None:
        self.datasets = {}
        for d in datasets:
            assert d in DATASET_CLASSES, f"Error: Dataset {d} not found!"
            d_class = DATASET_CLASSES[d]
            d_config = d
            if configs is not None and d in configs.keys():
                d_config = configs[d]
                print(d_config)
            self.datasets[d] = d_class(getattr(config, d_config)) # getattr is basically passing the config of d_config to NuscenesDataset class

    def get_datasets(self) -> dict:
        return self.datasets
