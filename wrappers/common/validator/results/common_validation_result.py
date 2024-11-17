from abc import ABC, abstractmethod
from typing import Any


class CommonValidationResult(ABC):

    @abstractmethod
    def append_data(self, pipeline_infos: dict[str, Any]) -> dict[str, Any]:
        ...