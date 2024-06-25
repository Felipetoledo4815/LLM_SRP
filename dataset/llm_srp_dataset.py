# Template from https://www.squash.io/creating-custom-datasets-and-dataloaders-in-pytorch/
from typing import List, Tuple
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from dataset.dataset_factory import DatasetFactory

class LLMSRPDataset(Dataset):
    def __init__(self, datasets: List[str]) -> None:
        dataset_factory = DatasetFactory(datasets)
        self.datasets = dataset_factory.get_datasets()
        self.dataset_limits = {}
        self.length = 0
        for dataset_name, dataset in self.datasets.items():
            self.length += len(dataset)
            self.dataset_limits[self.length] = dataset_name

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List[Tuple[str, str, str]]]:
        dataset_name, index_base = self.__get_dataset_name_and_index_base__(index)
        dataset_index = index - index_base
        img_path = self.datasets[dataset_name].get_image(dataset_index)
        img = np.array(Image.open(img_path))
        sg_triplets = self.datasets[dataset_name].get_sg_triplets(dataset_index)
        return img, sg_triplets

    def __get_dataset_name_and_index_base__(self, index: int) -> Tuple[str, int]:
        previous_limit = 0
        for dataset_length, dataset_name in self.dataset_limits.items():
            if index < dataset_length:
                return dataset_name, previous_limit
            previous_limit = dataset_length
        raise ValueError(f"Error: Index {index} out of bounds!")

    def collate_fn(self, batch: List[Tuple[np.ndarray, List[Tuple[str, str, str]]]]):
        images = torch.stack([torch.tensor(item[0]) for item in batch])
        sg_triplets_lengths = [len(item[1]) for item in batch]
        max_length = max(sg_triplets_lengths)
        padded_sg_triplets = []
        for sg_triplets in [item[1] for item in batch]:
            pads_to_add = max_length - len(sg_triplets)
            if pads_to_add > 0:
                for _ in range(pads_to_add):
                    sg_triplets.append(("pad","pad","pad"))
            padded_sg_triplets.append(sg_triplets)
        return images, padded_sg_triplets
