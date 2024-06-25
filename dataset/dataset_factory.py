from typing import List
from dataset import config
from dataset.nuscenes_dataset import NuscenesDataset


DATASET_CLASSES = {
    "nuscenes": NuscenesDataset
}

class DatasetFactory:
    def __init__(self, datasets: List[str]) -> None:
        self.datasets = {}
        for d in datasets:
            assert d in DATASET_CLASSES, f"Error: Dataset {d} not found!"
            d_class = DATASET_CLASSES[d]
            self.datasets[d] = d_class(getattr(config, d))

    def get_datasets(self) -> dict:
        return self.datasets
