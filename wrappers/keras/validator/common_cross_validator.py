import time
from abc import abstractmethod

from wrappers.keras.config.configurators import HyperBandConfig, SearchConfig, FinalFitConfig
from wrappers.keras.validator.results.common import KerasValidationResult


class KerasCrossValidator:

    def __init__(self):
        self.start_validation = 0
        self.end_validation = 0

    def process(self,
                train_data,
                validation_data,
                model,
                project_name: str,
                hyper_band_config: HyperBandConfig,
                search_config: SearchConfig,
                final_fit_config: FinalFitConfig) -> KerasValidationResult:
        self.start_validation = time.time()

        result = self._on_execute(train_data=train_data,
                                  validation_data=validation_data,
                                  model=model,
                                  project_name=project_name,
                                  hyper_band_config=hyper_band_config,
                                  search_config=search_config,
                                  final_fit_config=final_fit_config)

        self.end_validation = time.time()

        return result

    @abstractmethod
    def _on_execute(self,
                    train_data,
                    validation_data,
                    model,
                    project_name: str,
                    hyper_band_config: HyperBandConfig,
                    search_config: SearchConfig,
                    final_fit_config: FinalFitConfig) -> KerasValidationResult:
        ...