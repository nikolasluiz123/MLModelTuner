from keras_tuner import HyperModel

from wrappers.keras.config.configurators import HyperBandConfig, SearchConfig, FinalFitConfig
from wrappers.keras.history_manager.common import KerasHistoryManager
from wrappers.keras.pre_processing.pre_processor import KerasDataPreProcessor
from wrappers.keras.validator.common_cross_validator import KerasCrossValidator


class KerasPipeline:

    def __init__(self,
                 model: HyperModel,
                 data_pre_processor: KerasDataPreProcessor,
                 validator: KerasCrossValidator,
                 hyper_band_config: HyperBandConfig,
                 search_config: SearchConfig,
                 final_fit_config: FinalFitConfig,
                 history_manager: KerasHistoryManager):
        self.model = model
        self.data_pre_processor = data_pre_processor
        self.validator = validator
        self.hyper_band_config = hyper_band_config
        self.search_config = search_config
        self.final_fit_config = final_fit_config
        self.history_manager = history_manager

    def get_dict_pipeline_data(self) -> dict[str, str]:
        return {
            'model': type(self.model).__name__,
            'validator': type(self.validator).__name__,

            'objective': str(self.hyper_band_config.objective),
            'factor': str(self.hyper_band_config.factor),
            'max_epochs': str(self.hyper_band_config.max_epochs),

            'search_epochs': str(self.search_config.epochs),
            'search_batch_size': str(self.search_config.batch_size),
            'search_callbacks': str(self.search_config.callbacks),
            'search_folds': str(self.search_config.folds),

            'final_fit_epochs': str(self.final_fit_config.epochs),
            'final_fit_batch_size': str(self.final_fit_config.batch_size),
        }

    def get_execution_times(self):
        pre_processing_time = self.data_pre_processor.end_pre_processing - self.data_pre_processor.start_pre_processing
        validation_processing_time = self.validator.end_validation - self.validator.start_validation

        return pre_processing_time, validation_processing_time