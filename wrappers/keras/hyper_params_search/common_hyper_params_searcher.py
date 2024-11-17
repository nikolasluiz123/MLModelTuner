import time
from abc import ABC, abstractmethod

from keras.src.callbacks import Callback
from keras_tuner import HyperModel

from wrappers.common.hyper_params_searcher.common_hyper_params_search import CommonHyperParamsSearch


class KerasCommonHyperParamsSearcher(CommonHyperParamsSearch):

    def __init__(self,
                 objective: str | list[str],
                 directory: str,
                 project_name: str,
                 epochs: int,
                 batch_size: int,
                 callbacks: list[Callback],
                 log_level: int = 0):
        super().__init__(log_level)

        self.objective = objective
        self.directory = directory
        self.project_name = project_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks

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

    @abstractmethod
    def get_fields_oracle_json_file(self) -> list[str]:
        ...
