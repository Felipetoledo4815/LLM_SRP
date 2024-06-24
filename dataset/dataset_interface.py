# Tips at
# - https://stackabuse.com/guide-to-interfaces-in-python/
# - https://www.delftstack.com/howto/python/python-abstract-property/

from typing import List
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
    def __len__(self) -> int:
        """Returns length of the dataset"""
