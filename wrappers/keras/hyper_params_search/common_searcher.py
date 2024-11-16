import time
from abc import ABC, abstractmethod

from keras.src.callbacks import Callback
from keras_tuner import HyperModel


class KerasCommonHyperParamsSearcher(ABC):

    def __init__(self,
                 objective: str | list[str],
                 directory: str,
                 project_name: str,
                 epochs: int,
                 batch_size: int,
                 log_level: int,
                 callbacks: list[Callback]):
        self.objective = objective
        self.directory = directory
        self.project_name = project_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_level = log_level
        self.callbacks = callbacks

        self.start_search_parameter_time = 0
        self.end_search_parameter_time = 0

    def process(self,
                train_data,
                validation_data,
                model: HyperModel):
        self.start_search_parameter_time = time.time()

        model = self._on_execute(train_data=train_data,
                                 validation_data=validation_data,
                                 model=model)

        self.end_search_parameter_time = time.time()

        return model

    @abstractmethod
    def _on_execute(self,
                    train_data,
                    validation_data,
                    model: HyperModel):
        ...
