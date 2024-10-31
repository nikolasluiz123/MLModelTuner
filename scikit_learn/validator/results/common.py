from abc import ABC, abstractmethod
from typing import Any


class ValidationResult(ABC):

    @abstractmethod
    def append_data(self, pipeline_infos: dict[str, Any]) -> dict[str, Any]:
        ...