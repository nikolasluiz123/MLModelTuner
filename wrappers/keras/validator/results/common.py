from abc import abstractmethod, ABC
from typing import Any


class KerasValidationResult(ABC):

    def __init__(self, model, history):
        self.model = model
        self.history = history

    @abstractmethod
    def append_data(self, pipeline_infos: dict[str, Any]) -> dict[str, Any]:
        ...