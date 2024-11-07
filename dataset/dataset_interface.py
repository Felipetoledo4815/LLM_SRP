# Tips at
# - https://stackabuse.com/guide-to-interfaces-in-python/
# - https://www.delftstack.com/howto/python/python-abstract-property/

from typing import List, Tuple
from abc import ABC, abstractmethod
from dataset.utils.data_clases import EgoVehicle, Entity

class DatasetInterface(ABC):
    @abstractmethod
    def get_image(self, index: int) -> str:
        """Returns image path given an index"""

    @abstractmethod
    def get_ego_vehicle(self, index: int) -> EgoVehicle:
        """Returns ego vehicle given an index"""

    @abstractmethod
    def get_entities(self, index: int) -> List[Entity]:
        """Returns list of entities given an index"""

    @abstractmethod
    def get_sg_triplets(self, index: int) -> List[Tuple[str, str, str]]:
        """Returns list of scene graph triplets given an index"""

    @abstractmethod
    def get_bb_triplets(self, index: int) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
        """Returns bounding box and scene graph triplets given an index"""

    @abstractmethod
    def plot_data_point(self, index: int, out_path: None | str = None) -> None:
        """Plot data point given an index"""

    @abstractmethod
    def plot_bounding_box(self, index: int, bbs: List[str], entities: List[str] | None = None,
                          out_path: None | str = None) -> None:
        """Plot bounding box given an index, and bounding boxes"""

    @abstractmethod
    def entity_type_mapper(self, ann) -> str:
        """Maps entity type to 'vehicle', 'person', or 'bicycle'"""

    @abstractmethod
    def __len__(self) -> int:
        """Returns length of the dataset"""
