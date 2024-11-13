import time
from abc import abstractmethod

from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.validator.results.common import KerasValidationResult


class KerasCrossValidator:

    def __init__(self):
        self.start_validation = 0
        self.end_validation = 0

    def process(self, train_data, validation_data, pipeline: KerasPipeline, history_index: int = None) -> KerasValidationResult:
        self.start_validation = time.time()
        result = self._on_execute(train_data, validation_data, pipeline, history_index)
        self.end_validation = time.time()

        return result

    @abstractmethod
    def _on_execute(self, train_data, validation_data, pipeline: KerasPipeline, history_index: int = None) -> KerasValidationResult:
        ...