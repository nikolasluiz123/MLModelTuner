import time
from abc import ABC, abstractmethod


class CommonDataPreProcessor(ABC):

    def __init__(self):
        self.start_pre_processing = 0
        self.end_pre_processing = 0

    def get_train_data(self) -> tuple:
        self.start_pre_processing = time.time()
        tuple_data = self._on_execute_train_process()
        self.end_pre_processing = time.time()

        return tuple_data

    @abstractmethod
    def _on_execute_train_process(self) -> tuple:
        ...

    def get_data_additional_validation(self):
        ...