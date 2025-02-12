from typing import List, Tuple
from abc import ABC, abstractmethod

class VLMInterface(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """Initializes the VLM"""

    @abstractmethod
    def inference(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, float]:
        """Returns the output produced by the VLM"""

    @abstractmethod
    def parse_prompt(self, prompt: str, images: List[str], **kwargs):
        """Parses the prompt and returns prompt with VLM format"""
