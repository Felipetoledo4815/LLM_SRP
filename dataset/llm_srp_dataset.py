# Template from https://www.squash.io/creating-custom-datasets-and-dataloaders-in-pytorch/
from typing import List, Tuple, Dict
from enum import Enum, auto
from torch.utils.data import Dataset
from PIL import Image
import warnings
import numpy as np
import torch
from dataset.dataset_factory import DatasetFactory
from dataset.utils.data_clases import EntityType, RelationshipType


class TripletsFormat(Enum):
    """
    Triplets format.
    Options:
        DEFAULT: Array of tuples with three strings.
    """
    DEFAULT = auto()


class ImageFormat(Enum):
    """
    Image format.
    Options:
        DEFAULT: String with the image path.
        NP_ARRAY: Numpy array with the image.
    """
    DEFAULT = auto()
    NP_ARRAY = auto()


class BoundingBoxFormat(Enum):
    """
    Bounding box format.
    Options:
        DEFAULT: List of four integers describing 2 corners of the bounding box (x1, y1, x2, y2).
        These are the bottom left and top right corners.
    """
    DEFAULT = auto()


class LLMSRPDataset(Dataset):
    def __init__(self, datasets: List[str],
                 split: str | None = None,
                 output_format: Tuple[ImageFormat, TripletsFormat, BoundingBoxFormat] = (
                     ImageFormat.DEFAULT, TripletsFormat.DEFAULT, BoundingBoxFormat.DEFAULT),
                 configs: Dict[str, str] | None = None) -> None:
        dataset_factory = DatasetFactory(datasets, configs)
        self.split = split
        self.output_format = output_format
        self.datasets = dataset_factory.get_datasets()
        self.dataset_limits_inv = {}
        self.dataset_limits = {}
        self.length = 0
        for dataset_name, dataset in self.datasets.items():
            self.dataset_limits[dataset_name] = (self.length, self.length + len(dataset))
            self.length += len(dataset)
            self.dataset_limits_inv[self.length] = dataset_name
        self.index = 0
        self.indices = list(range(self.length))
        if split is not None:
            self.set_split(split)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[np.ndarray | str,
                                               List[Tuple[str, str, str]],
                                               List[Tuple[str, List[Tuple[str, str, str]]]]]:
        index = self.indices[index]
        dataset_name, index_base = self.__get_dataset_name_and_index_base__(index)
        dataset_index = index - index_base
        img = self.datasets[dataset_name].get_image(dataset_index)
        if self.output_format[0] == ImageFormat.NP_ARRAY:
            img = np.array(Image.open(img))
        sg_triplets = self.datasets[dataset_name].get_sg_triplets(dataset_index)
        bb_triplets = self.datasets[dataset_name].get_bb_triplets(dataset_index)
        return img, sg_triplets, bb_triplets

    def __get_dataset_name_and_index_base__(self, index: int) -> Tuple[str, int]:
        previous_limit = 0
        for dataset_length, dataset_name in self.dataset_limits_inv.items():
            if index < dataset_length:
                return dataset_name, previous_limit
            previous_limit = dataset_length
        raise ValueError(f"Error: Index {index} out of bounds!")

    def collate_fn(self, batch: List[Tuple[torch.Tensor, List[Tuple[str, str, str]]]]):
        images = torch.stack([torch.tensor(item[0]) for item in batch])
        sg_triplets_lengths = [len(item[1]) for item in batch]
        max_length = max(sg_triplets_lengths)
        padded_sg_triplets = []
        for sg_triplets in [item[1] for item in batch]:
            pads_to_add = max_length - len(sg_triplets)
            if pads_to_add > 0:
                for _ in range(pads_to_add):
                    sg_triplets.append(("pad", "pad", "pad"))
            padded_sg_triplets.append(sg_triplets)
        return images, padded_sg_triplets

    def plot_data_point(self, index: int, out_path: None | str = None) -> None:
        index = self.indices[index]
        dataset_name, index_base = self.__get_dataset_name_and_index_base__(index)
        dataset_index = index - index_base
        self.datasets[dataset_name].plot_data_point(dataset_index, out_path)

    def plot_bounding_box(self, index: int, bbs: List[str], entity_types: List[str],
                          out_path: None | str = None) -> None:
        index = self.indices[index]
        dataset_name, index_base = self.__get_dataset_name_and_index_base__(index)
        dataset_index = index - index_base
        self.datasets[dataset_name].plot_bounding_box(dataset_index, bbs, entity_types, out_path)

    def get_dataset_names(self) -> List[str]:
        return list(self.datasets.keys())

    def get_entity_names(self) -> List[str]:
        return EntityType.get_types()

    def get_relationship_names(self) -> List[str]:
        return RelationshipType.get_types()

    def set_split(self, split) -> None:
        if split not in ["train", "val", "test"]:
            warnings.warn(f"Split {split} not recognized. Using all data. Split must be 'train', 'val' or 'test'")
        else:
            # Splits (75% train, 5% val, 20% test)
            # First 75% train, next 5% val, last 20% test
            all_indices = list(range(self.length))
            if split == "train":
                new_indices = []
                for dataset_name, _ in self.datasets.items():
                    indices_start = self.dataset_limits[dataset_name][0]
                    indices_end = int(self.dataset_limits[dataset_name][1] * 0.75)
                    new_indices.extend(all_indices[indices_start:indices_end])
                self.indices = new_indices
            elif split == "val":
                new_indices = []
                for dataset_name, _ in self.datasets.items():
                    indices_start = int(self.dataset_limits[dataset_name][1] * 0.75)
                    indices_end = int(self.dataset_limits[dataset_name][1] * 0.80)
                    new_indices.extend(all_indices[indices_start:indices_end])
                self.indices = new_indices
            elif split == "test":
                new_indices = []
                for dataset_name, _ in self.datasets.items():
                    indices_start = int(self.dataset_limits[dataset_name][1] * 0.80)
                    new_indices.extend(all_indices[indices_start:])
                self.indices = new_indices
