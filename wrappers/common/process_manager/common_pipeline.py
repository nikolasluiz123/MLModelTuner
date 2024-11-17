from abc import ABC, abstractmethod
from typing import Any


class CommonPipeline(ABC):

    @abstractmethod
    def get_dict_pipeline_data(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def get_execution_times(self):
        ...